from dataclasses import dataclass

@dataclass
class ToyTransConfig:
    n_embed: int = 32
    n_head: int = 4
    n_ctx: int = 16
    n_layer: int = 4
    n_vocab: int = 100288
    is_diff: bool = False

@dataclass
class StableLMConfig:
    n_embed: int = 3072
    n_head: int = 12
    n_layer: int = 28
    n_ctx: int = 4096
    n_vocab: int = 100288
    is_diff: bool = False

CONFIG_ARGS = {
    "830M": dict(n_embed=1536, n_layer=24, n_head=8),
    "1.4B": dict(n_embed=2048, n_layer=24, n_head=8),
    "2.8B": dict(n_embed=2560, n_layer=32, n_head=10),
    "6.8B": dict(n_embed=4096, n_layer=32, n_head=16),
    "13.1B": dict(n_embed=5120, n_layer=40, n_head=20),
}
