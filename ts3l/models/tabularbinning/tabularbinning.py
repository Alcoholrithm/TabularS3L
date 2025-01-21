import torch
from torch import nn

from typing import Tuple

from ts3l.models.common import TS3LModule
from ts3l.models.common import MLP
from ts3l.utils import BaseEmbeddingConfig, BaseBackboneConfig

class TabularBinning(TS3LModule):
    def __init__(self,
                    embedding_config: BaseEmbeddingConfig,
                    backbone_config: BaseBackboneConfig,
                    output_dim: int,
                    decoder_dim: int,
                    decoder_depth: int,
                    dropout_rate: float = 0.04,
                    **kwargs) -> None:
        """Initialize TabularBinning

        Args:
            embedding_config (BaseEmbeddingConfig): Configuration for the embedding layer.
            backbone_config (BaseBackboneConfig): Configuration for the backbone network.
        """
        super(TabularBinning, self).__init__(embedding_config, backbone_config)

        self.decoder = MLP(input_dim = self.backbone_module.output_dim, hidden_dims=decoder_dim, n_hiddens=decoder_depth, dropout_rate=dropout_rate)

        self.head = nn.Linear(self.backbone_module.output_dim, output_dim)

    @property
    def encoder(self) -> nn.Module:
        return self.backbone_module

    def _first_phase_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, x

    
    def _second_phase_step(self, 
                x : torch.Tensor) -> torch.Tensor:
        return x