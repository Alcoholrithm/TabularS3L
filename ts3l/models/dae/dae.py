from typing import OrderedDict, Tuple, List

import torch
import torch.nn as nn


from ts3l.models.common import TS3LModule
from ts3l.models.common.reconstruction_head import ReconstructionHead

from ts3l.utils import BaseEmbeddingConfig, BaseBackboneConfig

class DAE(TS3LModule):
    def __init__(
        self,
        embedding_config: BaseEmbeddingConfig,
        backbone_config: BaseBackboneConfig,
        num_continuous: int, 
        cat_cardinality: List[int],
        dropout_rate = 0.04,
        output_dim = 2,
        **kwargs
    ):
        """Implementation of Denoising AutoEncoder.
        DAE processes input data that has been partially corrupted, producing clean data during the self-supervised learning stage. 
        The denoising task enables the model to learn the input distribution and generate latent representations that are robust to corruption. 
        These latent representations can be utilized for a variety of downstream tasks.
        Args:
            embedding_config (BaseEmbeddingConfig): Configuration for the embedding layer.
            backbone_config (BaseBackboneConfig): Configuration for the backbone network.
            num_continuous (int): The number of continuous features.
            cat_cardinality (List[int]): The cardinality of categorical features.
            dropout_rate (float, optional): A hyperparameter that is to control dropout layer. Default is 0.04.
            output_dim (int): The dimensionality of output.
        """
        super(DAE, self).__init__(embedding_config, backbone_config)

        self.mask_predictor = nn.Linear(self.backbone_module.output_dim, embedding_config.input_dim, bias=True)
        self.feature_predictor = ReconstructionHead(self.backbone_module.output_dim, num_continuous, cat_cardinality)

        self.head = nn.Sequential(
            OrderedDict([
                ("head_activation", nn.ReLU(inplace=True)),
                ("head_batchnorm", nn.BatchNorm1d(self.backbone_module.output_dim)),
                ("head_dropout", nn.Dropout(dropout_rate)),
                ("head_linear", nn.Linear(self.backbone_module.output_dim, output_dim))
            ])
        )
    
    @property
    def encoder(self) -> nn.Module:
        return self.backbone_module

    def _first_phase_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.embedding_module(x)
        emb = self.encoder(x)
        mask = torch.sigmoid(self.mask_predictor(emb))
        cat_preds, cont_preds = self.feature_predictor(emb)

        return mask, cat_preds, cont_preds
    
    def _second_phase_step(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_module(x)
        emb = self.encoder(x)
        output = self.head(emb)
        return output
