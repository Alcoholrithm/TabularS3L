from typing import Tuple

import torch
import torch.nn as nn

from ts3l.models.common import TS3LModule
from .vime_semi import VIMESemiSupervised

from ts3l.utils import BaseEmbeddingConfig, BaseBackboneConfig

class VIME(TS3LModule):
    def __init__(self, 
                embedding_config: BaseEmbeddingConfig,  backbone_config: BaseBackboneConfig, hidden_dim: int, output_dim: int, **kwargs):
        """Initialize VIME

        Args:
            input_dim (int): The dimension of the encoder
            hidden_dim (int): The hidden dimension of the predictor
            output_dim (int): The output dimension of the predictor
        """
        super(VIME, self).__init__(embedding_config, backbone_config)
        # self.__encoder = VIMESelfSupervised(self.embedding_module.output_dim, embedding_config.input_dim)
        self.mask_predictor = nn.Linear(self.backbone_module.output_dim, embedding_config.input_dim, bias=True)
        self.feature_predictor = nn.Linear(self.backbone_module.output_dim, embedding_config.input_dim, bias=True)
        
        self.predictor = VIMESemiSupervised(self.backbone_module.output_dim, hidden_dim, output_dim)
        
    @property
    def encoder(self) -> nn.Module:
        return self.backbone_module
        
    def _first_phase_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The first phase step of VIME

        Args:
            x (torch.Tensor): The input batch

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted mask vector and predicted features
        """
        x = self.embedding_module(x)
        x = self.encoder(x)
        mask_output, feature_output = torch.sigmoid(self.mask_predictor(x)), self.feature_predictor(x)
        return mask_output, feature_output
    
    
    def _second_phase_step(self, x: torch.Tensor) -> torch.Tensor:
        """The second phase step of VIME

        Args:
            x (torch.Tensor): The input batch.

        Returns:
            torch.Tensor: The predicted logits of VIME
        """
        x = self.embedding_module(x)
        x = self.encoder(x)
        logits = self.predictor(x)
        return logits