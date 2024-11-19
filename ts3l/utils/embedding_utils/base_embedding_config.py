from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BaseEmbeddingConfig:
    input_dim: int
    output_dim: int = field(default=None) # type: ignore