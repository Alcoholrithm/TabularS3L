from dataclasses import dataclass, field
from .base_backbone_config import BaseBackboneConfig
from typing import Union, List, Optional
from torch import nn

@dataclass
class MLPBackboneConfig(BaseBackboneConfig):
    input_dim: Optional[int] = field(default=None)
    hidden_dims: Union[int, List[int]] = field(default=128)
    output_dim: Optional[int] = field(default=None)
    n_hiddens: int = field(default=2)
    activation: str = field(default='ReLU')
    use_batch_norm: bool = field(default=True)

    def __post_init__(self):
        self.name = "mlp"
        
        if isinstance(self.hidden_dims, int):
            if self.n_hiddens > 1:
                self.hidden_dims = [self.hidden_dims for _ in range(self.n_hiddens - 1)]
            else:
                self.output_dim = self.hidden_dims
                self.hidden_dims = []
        
        if self.input_dim is None:
            raise TypeError("__init__ missing 1 required positional argument: 'input_dim'")
        
        if self.output_dim is None:
            self.output_dim = self.hidden_dims[-1]

        if not hasattr(nn, self.activation):
            raise ValueError(f"{self.activation} is not a valid activation of torch.nn")