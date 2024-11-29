from .base_embedding_config import BaseEmbeddingConfig
from dataclasses import dataclass, field
from typing import List

@dataclass
class IdentityEmbeddingConfig(BaseEmbeddingConfig):
    def __post_init__(self):
        self.name = "identity"
        self.output_dim = self.input_dim