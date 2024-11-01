from dataclasses import dataclass


@dataclass
class BaseEmbeddingConfig:
    input_dim: int