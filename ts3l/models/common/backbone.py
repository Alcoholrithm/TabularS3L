import torch
from torch import nn
from ts3l.utils import BaseBackboneConfig

from ts3l.models.common.mlp import MLP
from ts3l.models.common.transformer_encoder import TransformerEncoder

class TS3LBackboneModule(nn.Module):
    def __init__(self, config: BaseBackboneConfig):
        super().__init__()
        self.config = config
        
        self.__set_backbone_network()
    
    def __set_backbone_network(self):
        if self.config.name == "mlp":
            self.backbone = MLP(**self.config.__dict__)
        elif self.config.name == "transformer":
            self.backbone = TransformerEncoder(**self.config.__dict__)
        
        self.output_dim = self.backbone.output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)