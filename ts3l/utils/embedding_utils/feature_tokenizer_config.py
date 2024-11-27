from .base_embedding_config import BaseEmbeddingConfig
from dataclasses import dataclass, field
from typing import List

@dataclass
class FTEmbeddingConfig(BaseEmbeddingConfig):
    emb_dim: int = field(default=128)
    cont_nums: int = field(default=0)
    cat_cardinality: List[int] = field(default_factory=lambda: [])
    required_token_dim: int = field(default=1)
    
    def __set_output_dim(self):
        if self.required_token_dim == 1:
            self.output_dim = self.emb_dim * (self.cont_nums + len(self.cat_cardinality) + 1)
        else:
            self.output_dim = self.cont_nums + len(self.cat_cardinality)
            
    def __post_init__(self):
        self.name = "feature_tokenizer"
        self.__set_output_dim()
        if not self.required_token_dim in [1, 2]:
            raise ValueError(f"{self.required_token_dim} is not a valid value. Choices are: [1, 2]")