import torch
from torch import nn

from typing import Tuple

from ts3l.models.common import TS3LModule
from ts3l.utils import BaseEmbeddingConfig, BaseBackboneConfig

class TabularBinning(TS3LModule):
    def __init__(self,
                    embedding_config: BaseEmbeddingConfig,
                    backbone_config: BaseBackboneConfig,
                    **kwargs) -> None:
        """Initialize TabularBinning

        Args:
            embedding_config (BaseEmbeddingConfig): Configuration for the embedding layer.
            backbone_config (BaseBackboneConfig): Configuration for the backbone network.
        """
        super(TabularBinning, self).__init__(embedding_config, backbone_config)

    @property
    def encoder(self) -> nn.Module:
        return self.backbone_module

    def _first_phase_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, x

    
    def _second_phase_step(self, 
                x : torch.Tensor) -> torch.Tensor:
        return x