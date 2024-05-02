from numpy.typing import NDArray
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

import numpy as np

from ts3l.models.common import MLP


class DAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        encoder_depth=4,
        head_depth=2,
        dropout_rate = 0.04,
        output_dim = 2,
    ):
        """Implementation of Denoising AutoEncoder.
        DAE processes input data that has been partially corrupted, producing clean data during the self-supervised learning stage. 
        The denoising task enables the model to learn the input distribution and generate latent representations that are robust to corruption. 
        These latent representations can be utilized for a variety of downstream tasks.
        Args:
            input_dim (int): The size of the inputs
            hidden_dim (int): The dimension of the hidden layers
            encoder_depth (int, optional): The number of layers of the encoder MLP. Defaults to 4.
            head_depth (int, optional): The number of layers of the pretraining head. Defaults to 2.
            dropout_rate (float, optional): The probability of setting the outputs of the dropout layer to zero during training. Defaults to 0.04.
            output_dim (int, 2): The size of the outputs
        """
        super().__init__()

        self.encoder = MLP(input_dim, hidden_dim, encoder_depth, dropout_rate)
        self.mask_predictor_head = MLP(hidden_dim, input_dim, head_depth, dropout_rate)
        self.reconstruction_head = MLP(hidden_dim, input_dim, head_depth, dropout_rate)

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

    def __first_phase_step(self, x):

        emb = self.encoder(x)
        mask = torch.sigmoid(self.mask_predictor_head(emb))
        feature = self.reconstruction_head(emb)

        return mask, feature
    
    def __second_phase_step(self, x):
        emb = self.encoder(x)
        output = self.head(emb)
        return output
