import torch
from torch import nn
from typing import OrderedDict

class MLP(nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
        layers = []
        in_dim = input_dim
        
        for i in range(n_layers - 1):
            layers.append(("linear_%d" % i, torch.nn.Linear(in_dim, hidden_dim)))
            layers.append(("batchnorm_%d" % i, nn.BatchNorm1d(hidden_dim)))
            layers.append(("relu_%d" % i, nn.ReLU(inplace=True)))
            layers.append(("dropout_%d" % i, torch.nn.Dropout(dropout)))
            in_dim = hidden_dim

        layers.append(("linear_n_layers", torch.nn.Linear(in_dim, hidden_dim)))

        super().__init__(OrderedDict(layers))