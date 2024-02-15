from numpy.typing import NDArray

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

import numpy as np

class MLP(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim)),
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)


class SCARF(nn.Module):
    def __init__(
        self,
        # sampling_candidate: NDArray[np.float_],
        input_dim,
        emb_dim,
        encoder_depth=4,
        head_depth=2,
        dropout_rate = 0.3,
        # corruption_rate=0.6,
        out_dim = 2,
    ):
        """Implementation of SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption.
        It consists in an encoder that learns the embeddings.
        It is done by minimizing the contrastive loss of a sample and a corrupted view of it.
        The corrupted view is built by remplacing a random set of features by another sample randomly drawn independently.
            Args:
                input_dim (int): size of the inputs
                emb_dim (int): dimension of the embedding space
                encoder_depth (int, optional): number of layers of the encoder MLP. Defaults to 4.
                head_depth (int, optional): number of layers of the pretraining head. Defaults to 2.
                corruption_rate (float, optional): fraction of features to corrupt. Defaults to 0.6.
        """
        super().__init__()

        self.encoder = MLP(input_dim, emb_dim, encoder_depth)

        self.pretraining_head = MLP(emb_dim, emb_dim, head_depth)

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim, out_dim)
        )
        # initialize weights
        self.encoder.apply(self._init_weights)
        self.pretraining_head.apply(self._init_weights)
        
        # self.n_sampling_candidate = sampling_candidate.shape[0]
        # self.sampling_candidate = torch.FloatTensor(sampling_candidate.T)
        
        # self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        # self.corruption_len = int(corruption_rate * input_dim)

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
            
    def do_pretraining(self):
        self.forward = self.pretraining_step
    
    def do_finetunning(self):
        self.forward = self.finetunning_step

    def pretraining_step(self, x, x_corrupted):
        batch_size, m = x.size()

        # compute embeddings
        emb_anchor = self.encoder(x)
        emb_anchor = self.pretraining_head(emb_anchor)

        emb_anchor = F.normalize(emb_anchor, p=2)
        
        emb_corrupted = self.encoder(x_corrupted)
        emb_corrupted = self.pretraining_head(emb_corrupted)
        emb_corrupted = F.normalize(emb_corrupted, p=2)

        return emb_anchor, emb_corrupted
    
    def finetunning_step(self, x):
        emb = self.encoder(x)
        output = self.head(emb)

        return output
