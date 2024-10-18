from typing import Tuple

import torch
import torch.nn as nn

from ts3l.models.common import TS3LModule
from .vime_self import VIMESelfSupervised
from .vime_semi import VIMESemiSupervised

from ts3l.utils import EmbeddingConfig

class VIME(TS3LModule):
    def __init__(self, 
                embedding_config: EmbeddingConfig, hidden_dim: int, output_dim: int, **kwargs):
        """Initialize VIME

        Args:
            input_dim (int): The dimension of the encoder
            hidden_dim (int): The hidden dimension of the predictor
            output_dim (int): The output dimension of the predictor
        """
        super(VIME, self).__init__(embedding_config)
        self.__encoder = VIMESelfSupervised(self.embedding_module.output_dim, embedding_config.input_dim)
        self.predictor = VIMESemiSupervised(self.embedding_module.output_dim, hidden_dim, output_dim)
        
    @property
    def encoder(self) -> nn.Module:
        return self.__encoder
        
    def _first_phase_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The first phase step of VIME

        Args:
            x (torch.Tensor): The input batch

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted mask vector and predicted features
        """
        x = self.embedding_module(x)
        mask_output, feature_output = self.encoder(x)
        return mask_output, feature_output
    
    
    def _second_phase_step(self, x: torch.Tensor) -> torch.Tensor:
        """The second phase step of VIME

        Args:
            x (torch.Tensor): The input batch.

        Returns:
            torch.Tensor: The predicted logits of VIME
        """
        x = self.embedding_module(x)
        x = self.encoder.h(x)
        logits = self.predictor(x)
        return logits