from typing import OrderedDict, Tuple

import torch
import torch.nn as nn


from ts3l.models.common import TS3LModule
from ts3l.models.common import MLP


class DAE(TS3LModule):
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
        super(DAE, self).__init__()

        self.__encoder = MLP(input_dim, hidden_dim, encoder_depth, dropout_rate)
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
    
    @property
    def encoder(self) -> nn.Module:
        return self.__encoder

    def _first_phase_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        mask = torch.sigmoid(self.mask_predictor_head(emb))
        feature = self.reconstruction_head(emb)

        return mask, feature
    
    def _second_phase_step(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        output = self.head(emb)
        return output
