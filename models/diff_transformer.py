import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.lambda_q1 = nn.Parameter(torch.zeros(config.n_embed, dtype=torch.float32).normal_(mean=0.0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(config.n_embed, dtype=torch.float32).normal_(mean=0.0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(config.n_embed, dtype=torch.float32).normal_(mean=0.0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(config.n_embed, dtype=torch.float32).normal_(mean=0.0, std=0.1))

        # self.RMSNorm = nn.RMSNorm(config.n_embed, config.n_ctx,)

        self.register_buffer("mask", torch.tril(torch.ones(config.n_ctx, config.n_ctx)).view(1, 1, config.n_ctx, config.n_ctx))

    def forward(self, x,):
        B, T, C = x.shape
        # Note: C = config.n_embed
        C = C // 2

        qkv = self.c_attn(x)
        q1, k1, q2, k2, v = torch.split(qkv, [C, C, C, C, 2*C], dim=-1)

        q1 = q1.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim = d_k)
        k1 = k1.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q2 = q2.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k2 = k2.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, 2*self.head_dim).transpose(1, 2)

        lambda_ = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1)) - torch.exp(torch.dot(self.lambda_q2, self.lambda_k2)) + self.lambda_init

        # Scaled dot product (Naive)
        attn1 = q1 @ k1.transpose(-1, -2) * (self.head_dim ** -0.5)
        attn2 = q2 @ k2.transpose(-1, -2) * (self.head_dim ** -0.5)

        attn_score = F.softmax(attn1 - lambda_ * attn2, dim=-1)

        diff_attn = attn_score @ v # (B, n_head, T, head_dim)

        diff_attn = (1 - self.lambda_init) * diff_attn

        diff_attn = diff_attn.transpose(1, 2).contiguous().view(B, T, 2*C)

        score = self.c_proj(diff_attn)

        return score


        
        
if __name__ == "__main__":
    from config import ToyTransConfig

    config = ToyTransConfig()
    model = MultiHeadDiffAttention(config, 1)

    x = torch.randn(1, config.n_ctx, config.n_embed)
    output = model(x)











        


        








