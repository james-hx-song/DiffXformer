from dataclasses import dataclass

@dataclass
class ToyTransConfig:
    n_embed: int = 32
    n_head: int = 4
    n_ctx: int = 16

