from typing import OrderedDict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ts3l.models.common import TS3LModule
from ts3l.models.common import MLP


class SCARF(TS3LModule):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        encoder_depth=4,
        head_depth=2,
        dropout_rate = 0.04,
    ) -> None:
        """Implementation of SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption.
        It consists in an encoder that learns the embeddings.
        It is done by minimizing the contrastive loss of a sample and a corrupted view of it.
        The corrupted view is built by replacing a random set of features by another sample randomly drawn independently.
            Args:
                input_dim (int): The size of the inputs.
                hidden_dim (int): The dimension of the hidden layers.
                output_dim (int): The dimension of output.
                encoder_depth (int, optional): The number of layers of the encoder MLP. Defaults to 4.
                head_depth (int, optional): The number of layers of the pretraining head. Defaults to 2.
                dropout_rate (float, optional): A hyperparameter that is to control dropout layer. Default is 0.04.
        """
        super(SCARF, self).__init__()

        self.__encoder = MLP(input_dim, hidden_dim, encoder_depth)

        self.pretraining_head = MLP(hidden_dim, hidden_dim, head_depth)

        self.head = nn.Sequential(
            OrderedDict([
                ("head_activation", nn.ReLU(inplace=True)),
                ("head_batchnorm", nn.BatchNorm1d(hidden_dim)),
                ("head_dropout", nn.Dropout(dropout_rate)),
                ("head_linear", nn.Linear(hidden_dim, output_dim))
            ])
        )

    @property
    def encoder(self) -> nn.Module:
        return self.__encoder
        
    def _first_phase_step(self, x: torch.Tensor, x_corrupted: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        emb_anchor = self.encoder(x)
        emb_anchor = self.pretraining_head(emb_anchor)

        emb_anchor = F.normalize(emb_anchor, p=2)
        
        emb_corrupted = self.__encoder(x_corrupted)
        emb_corrupted = self.pretraining_head(emb_corrupted)
        emb_corrupted = F.normalize(emb_corrupted, p=2)

        return emb_anchor, emb_corrupted
    
    def _second_phase_step(self, x) -> torch.Tensor:
        emb = self.encoder(x)
        output = self.head(emb)

        return output
