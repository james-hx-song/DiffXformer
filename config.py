from dataclasses import dataclass

@dataclass
class ToyTransConfig:
    n_embed: int = 32
    n_head: int = 4
    n_ctx: int = 16

@dataclass
class StableLM3BConfig:
    n_embed: int = 3072
    n_head: int = 12
    n_layer: int = 28
    n_ctx: int = 4096
    n_vocab: int = 100288

