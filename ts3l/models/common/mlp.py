import torch
from torch import nn
from typing import OrderedDict, List, Union, Optional

class MLP(nn.Sequential):
    """Simple multi-layer perceptron with activation and optional batch normalization layer and dropout layer"""

    def __init__(self, input_dim: int, hidden_dims: Union[int, List[int]], output_dim = None, n_hiddens: Optional[int] = None, activation: str ="ReLU", use_batch_norm: bool =True, dropout_rate: float =0.0, **kwargs):
        layers = []
        in_dim = input_dim
        
        if use_batch_norm:
            batch_norm_layer = nn.BatchNorm1d # type: ignore
        else:
            batch_norm_layer = nn.Identity # type: ignore
        
        if n_hiddens == None:
            if isinstance(hidden_dims, int):
                n_hiddens = 1
            else:
                n_hiddens = len(hidden_dims)
        
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims for _ in range(n_hiddens)] # type: ignore
        
        if output_dim == None:
            output_dim = hidden_dims[-1]

        for i in range(n_hiddens - 1): # type: ignore
            layers.append(("linear_%d" % i, torch.nn.Linear(in_dim, hidden_dims[i])))
            layers.append(("batchnorm_%d" % i, batch_norm_layer(hidden_dims[i]))) # type: ignore
            layers.append((f"{activation}_%d" % i, getattr(nn, activation)()))
            layers.append(("dropout_%d" % i, torch.nn.Dropout(dropout_rate))) # type: ignore
            in_dim = hidden_dims[i]
        
        if len(hidden_dims) > 0 and len(layers) > 0:
            layers.append((f"linear_{n_hiddens - 1}_layers", torch.nn.Linear(hidden_dims[-1], output_dim))) # type: ignore
        else:
            layers.append((f"linear_{n_hiddens - 1}_layers", torch.nn.Linear(input_dim, output_dim))) # type: ignore
        
        self.output_dim = output_dim
        super().__init__(OrderedDict(layers))