from dataclasses import dataclass

@dataclass
class ToyTransConfig:
    n_embed: int = 32
    n_head: int = 4
    n_ctx: int = 64
    n_layer: int = 4
    n_vocab: int = 49152
    is_diff: bool = False

@dataclass
class StableLMConfig:
    n_embed: int = 3072
    n_head: int = 12
    n_layer: int = 28
    n_ctx: int = 4096
    n_vocab: int = 100288
    is_diff: bool = False

@dataclass
class LMConfig:
    n_embed: int = 768
    n_head: int = 12
    n_layer: int = 24
    n_ctx: int = 1024
    n_vocab: int = 49152 # Tokenizer we are using 
    is_diff: bool = True
    
LM_ARGS = {
    "17M": dict(n_embed=256, n_layer=6, n_head=4, n_ctx=512),
    "122M": dict(n_embed=768, n_layer=12, n_head=12),
    "204M": dict(n_embed=960, n_layer=20, n_head=12),   
    "312M": dict(n_embed=960, n_layer=24, n_head=12),
}

CONFIG_ARGS = {
    "830M": dict(n_embed=1536, n_layer=24, n_head=8),
    "1.4B": dict(n_embed=2048, n_layer=24, n_head=8),
    "2.8B": dict(n_embed=2560, n_layer=32, n_head=10),
    "6.8B": dict(n_embed=4096, n_layer=32, n_head=16),
    "13.1B": dict(n_embed=5120, n_layer=40, n_head=20),
}
