import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtune as tt
import math


def lambda_init(layer):
    return 0.8 - 0.6 * math.exp(-0.3 * (layer - 1))


class MultiHeadDiffAttention(nn.Module):
    def __init__(self, config, layer):
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
        
        self.rope = tt.modules.RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=config.n_ctx, base=10000)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = torch.split(qkv, [C, C, C], dim=-1)

        q = q.view(B, T, 2 * self.n_head, self.head_dim)
        k = k.view(B, T, 2 * self.n_head, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        q1, q2 = torch.split(q, [self.n_head, self.n_head], dim=-2)
        k1, k2 = torch.split(k, [self.n_head, self.n_head], dim=-2)

        q1, q2 = q1.transpose(1, 2), q2.transpose(1, 2)
        k1, k2 = k1.transpose(1, 2), k2.transpose(1, 2)

        v = v.view(B, T, self.n_head, 2*self.head_dim).transpose(1, 2)

        lambda_ = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1)) - \
            torch.exp(torch.dot(self.lambda_q2, self.lambda_k2)) + \
            self.lambda_init

        A1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=True)
        A2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=True)

        diff_attn = A1 - lambda_ * A2
        diff_attn = self.RMSNorm(diff_attn)
        diff_attn = (1 - self.lambda_init) * diff_attn

        diff_attn = diff_attn.transpose(1, 2).contiguous().view(B, T, C)
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
        
        self.rope = tt.modules.RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=config.n_ctx, base=10000)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = torch.split(qkv, [C, C, C], dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        score = self.RMSNorm(attn)
        score = score.transpose(1, 2).contiguous().view(B, T, C)
        score = self.c_proj(score)

        return score


class GatedFFN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_embed = config.n_embed
        self.W_1 = nn.Linear(config.n_embed, int(config.n_embed * 8.0 / 3.0), bias=False)
        self.W_G = nn.Linear(config.n_embed, int(config.n_embed * 8.0 / 3.0), bias=False)
        self.W_2 = nn.Linear(int(config.n_embed * 8.0 / 3.0), config.n_embed, bias=False)
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
        self.RMSNorm = nn.RMSNorm(config.n_embed, eps=1e-5, elementwise_affine=False)

    def forward(self, x):
        x = self.attn(self.RMSNorm(x)) + x
        x = self.ffn(self.RMSNorm(x)) + x
        return x


class TransModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.is_diff:
            print("Using Differential Transformer")
        else:
            print("Using Vanilla Transformer")
        
        self.projection = nn.Linear(in_features=83, out_features=config.n_embed)

        self.blocks = nn.ModuleList([Block(config, i+1) for i in range(config.n_layer)])

        # Removed embedding layer
        # Removed weight tying since there's no embedding now
        self.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)

    def forward(self, x):
        # Now x is expected to be of shape (B, T, n_embed) directly.
        assert x.size(1) <= self.config.n_ctx, "Context length exceeds model's maximum context length"

        x = self.projection(x)
        
        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(x)
        return logits


if __name__ == "__main__":
    from configs.config import LMConfig, ToyTransConfig, LM_ARGS
    model = TransModel(LMConfig(**LM_ARGS["204M"]))
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
