
from dataclasses import dataclass, field
from .base_backbone_config import BaseBackboneConfig
from typing import Optional

@dataclass
class TransformerBackboneConfig(BaseBackboneConfig):
    d_model: Optional[int] = field(default=None)
    ffn_factor: float = field(default=2.0)
    hidden_dim: int = field(default=256)
    encoder_depth: int = field(default=3)
    n_head: int = field(default=2)
    
    def __post_init__(self):
        self.name = "transformer"
        
        if self.d_model is None:
            raise TypeError("__init__ missing 1 required positional argument: 'd_model'")