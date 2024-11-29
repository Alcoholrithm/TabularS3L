from typing import Tuple, List

import torch
import torch.nn as nn

from ts3l.models.common import TS3LModule
from ts3l.models.common.reconstruction_head import ReconstructionHead
from .vime_predictor import VIMEPredictor

from ts3l.utils import BaseEmbeddingConfig, BaseBackboneConfig

class VIME(TS3LModule):
    def __init__(self, 
                embedding_config: BaseEmbeddingConfig,  backbone_config: BaseBackboneConfig, num_continuous: int, cat_cardinality: List[int], predictor_dim: int, output_dim: int, **kwargs):
        """Initialize VIME

        Args:
            embedding_config (BaseEmbeddingConfig): Configuration for the embedding layer.
            backbone_config (BaseBackboneConfig): Configuration for the backbone network.
            num_continuous (int): The number of continuous features.
            cat_cardinality (List[int]): The cardinality of categorical features.
            predictor_dim (int): The hidden dimension of the predictor.
            output_dim (int): The output dimension of the predictor.
        """
        super(VIME, self).__init__(embedding_config, backbone_config)

        self.mask_predictor = nn.Linear(self.backbone_module.output_dim, embedding_config.input_dim, bias=True)
        self.feature_predictor = ReconstructionHead(self.backbone_module.output_dim, num_continuous, cat_cardinality)
        self.predictor = VIMEPredictor(self.backbone_module.output_dim, predictor_dim, output_dim)
        
    @property
    def encoder(self) -> nn.Module:
        return self.backbone_module
        
    def _first_phase_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The first phase step of VIME

        Args:
            x (torch.Tensor): The input batch

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The predicted mask vector and predicted features
        """
        x = self.embedding_module(x)
        x = self.encoder(x)
        mask_output, (cat_preds, cont_preds) = torch.sigmoid(self.mask_predictor(x)), self.feature_predictor(x)
        return mask_output, cat_preds, cont_preds
    
    
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