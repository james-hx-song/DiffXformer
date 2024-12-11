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
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))

        self.RMSNorm = nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)

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

        v = v.view(B, T, self.n_head, 2 * self.head_dim).transpose(1, 2)

        lambda_ = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1)) - torch.exp(torch.dot(self.lambda_q2, self.lambda_k2)) + self.lambda_init

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
        self.RMSNorm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=False)

        self.rope = tt.modules.RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=config.n_ctx, base=10000)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = torch.split(qkv, [C, C, C], dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)

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

        self.projection = nn.Linear(in_features=20, out_features=config.n_embed)

        self.blocks = nn.ModuleList([Block(config, i + 1) for i in range(config.n_layer)])

        # Removed embedding layer
        # Removed weight tying since there's no embedding now
        self.lm_head = nn.Linear(config.n_embed, 1, bias=False)

    def forward(self, x):
        # Now x is expected to be of shape (B, T, n_embed) directly.
        assert x.size(1) <= self.config.n_ctx, "Context length exceeds model's maximum context length"

        x = self.projection(x)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(x)
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids, max_length=200, eos_token_id=None, return_logits=False):
        """
        Simple greedy generation:
        - input_ids: a starting prompt (batch_size=1 for simplicity)
        - max_length: maximum number of tokens to generate
        """
        
        assert eos_token_id is not None, "Please provide an EOS token ID"
        
        if return_logits:
            logits_list = []
        for _ in range(max_length):
            # Ensure we don't exceed context window
            if input_ids.size(1) > self.config.n_ctx:
                break

            # Get logits for the current sequence
            logits = self.forward(input_ids)  # [B, T, Vocab]
            next_token_logits = logits[:, -1, :]  # Take the last position
            if return_logits:
                logits_list.append(next_token_logits)
            # Greedy: take argmax
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [B,1]
            # Append the next token to the sequence
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
            # Optional: stop if we hit an EOS token (depends on your vocab)
            if next_token_id.item() == eos_token_id:
                break

        return input_ids, (logits_list if return_logits else None)
    
    
    @torch.no_grad()
    def beam_search(
        self,
        input_ids,
        beam_width=5,
        max_length=50,
        eos_token_id=None
    ):
        """
        Beam search decoding.
        - input_ids: initial prompt (batch_size=1)
        - beam_width: number of beams to keep
        - max_length: max tokens to generate
        - tokenizer (optional): for debugging/printing partial results
        - eos_token_id (optional): if provided, stop beam when eos is generated
        """

        # Initialize beams: Each beam is (sequence, score)
        # score is the log probability of the sequence so far
        beams = [(input_ids, 0.0)]

        for step in range(max_length):
            
            # If all beams have ended with EOS (if eos_token_id provided), break early
            if eos_token_id is not None and all((beam[0][0, -1].item() == eos_token_id for beam in beams)):
                break

            # Prepare all sequences in a single batch
            all_input_ids = torch.cat([beam[0] for beam in beams], dim=0)  # Shape: [beam_width, seq_len]

            # Run the model to get the logits for the last token
            logits = self.forward(all_input_ids)  # [beam_width, seq_len, vocab]
            next_token_logits = logits[:, -1, :]  # [beam_width, vocab]

            # Convert logits to log probs
            log_probs = F.log_softmax(next_token_logits, dim=-1)  # [beam_width, vocab]

            # For each beam, get top candidates
            # We'll expand each beam into beam_width candidates
            # Then we'll sort all candidates and pick top beam_width overall
            candidate_beams = []
            for i, (seq, score) in enumerate(beams):
                # Get log_probs for this beam
                beam_log_probs = log_probs[i]

                # Top candidates for this beam
                top_log_probs, top_ids = torch.topk(beam_log_probs, beam_width)
                for log_p, token_id in zip(top_log_probs, top_ids):
                    new_score = score + log_p.item()
                    new_seq = torch.cat([seq, token_id.view(1,1)], dim=1)
                    candidate_beams.append((new_seq, new_score))

            # Sort candidate beams by score (descending order)
            candidate_beams.sort(key=lambda x: x[1], reverse=True)

            # Keep top beam_width
            beams = candidate_beams[:beam_width]

        # After finishing:
        # The first beam in 'beams' is most likely the highest scoring one
        best_seq, best_score = beams[0]

        return best_seq, best_score


# if __name__ == "__main__":
#     from configs.config import LMConfig, ToyTransConfig, LM_ARGS

#     # config = ToyTransConfig()
#     # model = MultiHeadDiffAttention(config, 1)

#     # x = torch.randn(1, config.n_ctx, config.n_embed)
#     # output = model(x)
#     # model = DifferentialTransformer(StableLMConfig())
#     # print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
#     # model = TransModel(StableLMConfig(**CONFIG_ARGS["830M"]))
#     model = TransModel(LMConfig(**LM_ARGS["204M"]))

#     print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
#     # x = torch.randint(0, 100, (1, 16))
#     # print(x.shape)
#     # output = model(x)
#     # print(output.shape)
