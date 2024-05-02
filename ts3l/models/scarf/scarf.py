from numpy.typing import NDArray
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

import numpy as np
from ts3l.models.common import MLP


class SCARF(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        encoder_depth=4,
        head_depth=2,
        dropout_rate = 0.04,
    ):
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
        super().__init__()

        self.encoder = MLP(input_dim, hidden_dim, encoder_depth)

        self.pretraining_head = MLP(hidden_dim, hidden_dim, head_depth)

        self.head = nn.Sequential(
            OrderedDict([
                ("head_activation", nn.ReLU(inplace=True)),
                ("head_batchnorm", nn.BatchNorm1d(hidden_dim)),
                ("head_dropout", nn.Dropout(dropout_rate)),
                ("head_linear", nn.Linear(hidden_dim, output_dim))
            ])
        )

            
    def set_first_phase(self):
        self.forward = self.__first_phase_step
    
    def set_second_phase(self):
        self.forward = self.__second_phase_step

    def __first_phase_step(self, x, x_corrupted):

        emb_anchor = self.encoder(x)
        emb_anchor = self.pretraining_head(emb_anchor)

        emb_anchor = F.normalize(emb_anchor, p=2)
        
        emb_corrupted = self.encoder(x_corrupted)
        emb_corrupted = self.pretraining_head(emb_corrupted)
        emb_corrupted = F.normalize(emb_corrupted, p=2)

        return emb_anchor, emb_corrupted
    
    def __second_phase_step(self, x):
        emb = self.encoder(x)
        output = self.head(emb)

        return output
