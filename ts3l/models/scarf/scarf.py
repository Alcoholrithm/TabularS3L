from typing import OrderedDict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ts3l.models.common import TS3LModule
from ts3l.models.common import MLP
from ts3l.utils import BaseEmbeddingConfig, BaseBackboneConfig

class SCARF(TS3LModule):
    def __init__(
        self,
        embedding_config: BaseEmbeddingConfig,
        backbone_config: BaseBackboneConfig,
        pretraining_head_dim: int,
        output_dim: int,
        head_depth: int =2,
        dropout_rate: float = 0.04,
        **kwargs
    ) -> None:
        """Implementation of SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption.
        It consists in an encoder that learns the embeddings.
        It is done by minimizing the contrastive loss of a sample and a corrupted view of it.
        The corrupted view is built by replacing a random set of features by another sample randomly drawn independently.
            Args:
                embedding_config (BaseEmbeddingConfig): Configuration for the embedding layer.
                backbone_config (BaseBackboneConfig): Configuration for the backbone network.
                pretraining_head_dim (int): The dimensionality of the pretraining head.
                output_dim (int): The dimensionality of output.
                head_depth (int, optional): The number of layers of the pretraining head. Defaults to 2.
                dropout_rate (float, optional): A hyperparameter that is to control dropout layer of the head. Default is 0.04.
        """
        super(SCARF, self).__init__(embedding_config, backbone_config)

        self.pretraining_head = MLP(input_dim = self.backbone_module.output_dim, hidden_dims=pretraining_head_dim, n_hiddens=head_depth, dropout_rate=dropout_rate)

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
        
    def _first_phase_step(self, x: torch.Tensor, x_corrupted: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        size = len(x)
        x = torch.concat((x, x_corrupted))
        
        x = self.embedding_module(x)
        emb_anchor = self.encoder(x)
        
        emb_anchor = self.pretraining_head(emb_anchor)
        emb_anchor = F.normalize(emb_anchor, p=2)
        
        return emb_anchor[:size], emb_anchor[size:]
    
    def _second_phase_step(self, x) -> torch.Tensor:
        x = self.embedding_module(x)
        emb = self.encoder(x)
        output = self.head(emb)

        return output