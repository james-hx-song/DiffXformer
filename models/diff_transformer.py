import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def lambda_init(layer):
    return 0.8 - 0.6 * math.exp(-0.3 * (layer - 1))


class MultiHeadDiffAttention(nn.Module):
    def __init__(self, config, layer,):
        super().__init__()

        # Note: Diff Transformer splits head dims as: d_k = d / 2h
        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3, bias=False)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.n_head = config.n_head

        self.head_dim = config.n_embed // config.n_head // 2

        self.lambda_init = lambda_init(layer)
        self.lambda_q1 = nn.Parameter(torch.zeros(
            self.head_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(
            self.head_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(
            self.head_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(
            self.head_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))

        self.RMSNorm = nn.RMSNorm(
            2 * self.head_dim, eps=1e-5, elementwise_affine=False)


    def forward(self, x,):
        B, T, C = x.shape
        # Note: C = config.n_embed
        C = C // 2

        qkv = self.c_attn(x)
        q1, k1, q2, k2, v = torch.split(qkv, [C, C, C, C, 2*C], dim=-1)

        q1 = q1.view(B, T, self.n_head, self.head_dim).transpose(
            1, 2)  # (B, n_head, T, head_dim = d_k)
        k1 = k1.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q2 = q2.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k2 = k2.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, 2*self.head_dim).transpose(1, 2)

        lambda_ = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1)) - \
            torch.exp(torch.dot(self.lambda_q2, self.lambda_k2)) + \
            self.lambda_init

        # Scaled dot product attention
        A1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=True)
        A2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=True)

        diff_attn = A1 - lambda_ * A2

        diff_attn = self.RMSNorm(diff_attn)
        diff_attn = (1 - self.lambda_init) * diff_attn

        diff_attn = diff_attn.transpose(1, 2).contiguous().view(B, T, 2*C)

        score = self.c_proj(diff_attn)

        return score


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3, bias=False)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.n_head = config.n_head

        self.head_dim = config.n_embed // config.n_head

        self.RMSNorm = nn.RMSNorm(
            self.head_dim, eps=1e-5, elementwise_affine=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = torch.split(qkv, [C, C, C], dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(
            1, 2)  # (B, n_head, T, head_dim = d_k)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        score = self.RMSNorm(attn)

        score = score.transpose(1, 2).contiguous().view(B, T, C)

        score = self.c_proj(score)

        return score


class GatedFFN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_embed = config.n_embed

        self.W_1 = nn.Linear(config.n_embed, int(
            config.n_embed * 8.0 / 3.0), bias=False)
        self.W_G = nn.Linear(config.n_embed, int(
            config.n_embed * 8.0 / 3.0), bias=False)
        self.W_2 = nn.Linear(int(config.n_embed * 8.0 / 3.0),
                             config.n_embed, bias=False)

        # Swish activation function
        self.SiLU = nn.SiLU()

    def forward(self, x):
        x = self.W_1(x) * self.SiLU(self.W_G(x))
        x = self.W_2(x)

        return x


class Block(nn.Module):
    def __init__(self, config, layer):
        super().__init__()

        if config.is_diff:
            self.attn = MultiHeadDiffAttention(config, layer)
        else:
            self.attn = MultiHeadAttention(config)
        self.ffn = GatedFFN(config)

        self.RMSNorm = nn.RMSNorm(
            config.n_embed, eps=1e-5, elementwise_affine=False)

    def forward(self, x):
        x = self.attn(self.RMSNorm(x)) + x
        x = self.ffn(self.RMSNorm(x)) + x

        return x


class DifferentialTransformer(nn.Module):
    def __init__(self, config,):
        super().__init__()

        self.config = config

        self.blocks = nn.ModuleList([Block(config, i+1)
                                    for i in range(config.n_layer)])
        self.wte = nn.Embedding(
            config.n_vocab, config.n_embed)  # Token embeddings
        # Positional Embeddings
        self.wpe = nn.Embedding(config.n_ctx, config.n_embed)

        self.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)

        # Weight Sharing Scheme per Vaswani et al (2017)
        self.lm_head.weight = self.wte.weight

    def forward(self, x):
        assert x.size(
            1) <= self.config.n_ctx, "Context length exceeds model's maximum context length"
        x = self.wte(x) + self.wpe(torch.arange(x.size(1), device=x.device))

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(x)
        return logits


if __name__ == "__main__":
    from config import StableLMConfig, VANILLA_CONFIG_ARGS, DIFF_CONFIG_ARGS

    # config = ToyTransConfig()
    # model = MultiHeadDiffAttention(config, 1)

    # x = torch.randn(1, config.n_ctx, config.n_embed)
    # output = model(x)
    # model = DifferentialTransformer(StableLMConfig())
    # print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = DifferentialTransformer(
        StableLMConfig(**VANILLA_CONFIG_ARGS["830M"]))
    x = torch.randint(0, 100, (1, 16))
    print(x.shape)
    output = model(x)
    print(output.shape)
