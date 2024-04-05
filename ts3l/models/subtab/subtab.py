import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from numpy.typing import NDArray

import itertools

from types import SimpleNamespace
from typing import Dict, Any, Tuple, List, Union

class ShallowEncoder(nn.Module):
    def __init__(self,
                 feat_dim : int,
                 hidden_dim : int,
                 n_subsets : int,
                 overlap_ratio : float,
    ) -> None:
        super().__init__()

        n_column_subset = int(feat_dim / n_subsets)
        n_overlap = int(overlap_ratio * n_column_subset)

        self.net = nn.Sequential(
            nn.Linear(n_column_subset + n_overlap, hidden_dim),
            nn.LeakyReLU(),
        )
        
    def forward(self,
                x : torch.Tensor
    ) -> torch.Tensor:
        return self.net(x)

class ShallowDecoder(nn.Module):
    def __init__(self,
                 hidden_dim : int,
                 out_dim : int
    ) -> None:
        super().__init__()

        self.net = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AutoEncoder(nn.Module):
    def __init__(self,
                 feat_dim : int,
                 hidden_dim : int,
                 n_subsets : int,
                 overlap_ratio : float,
    ) -> None:
        super().__init__()

        self.encoder = ShallowEncoder(feat_dim, hidden_dim, n_subsets, overlap_ratio)
        self.decoder = ShallowDecoder(hidden_dim, feat_dim)

        self.projection_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def encode(self, x : torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, x : torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
    
    def forward(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        projection = self.projection_net(latent)
        projection = F.normalize(projection, p = 2, dim = 1)
        x_recon = self.decode(latent)
        return latent, projection, x_recon

class SubTab(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 
                 n_subsets: int,
                 overlap_ratio: float,
    ) -> None:
        super().__init__()

        self.feat_dim = input_dim
        
        self.n_subsets = n_subsets
        
        self.ae = AutoEncoder(self.feat_dim, hidden_dim, n_subsets, overlap_ratio)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
        self.set_first_phase()
        
    

    def set_first_phase(self) -> None:
        self.forward = self.__first_phase_step
    
    def set_second_phase(self) -> None:
        self.forward = self.__second_phase_step

    def __first_phase_step(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        latents, projections, x_recons = self.ae(x)
        
        return projections, x_recons, x
    
    def __arange_subsets(self, latent: torch.Tensor) -> torch.Tensor:
        no, dim = latent.shape
        samples = int(no / self.n_subsets)
        latent = latent.reshape((self.n_subsets, samples, dim))
        return torch.concat([latent[:, i] for i in range(samples)])
    
    def __second_phase_step(self, 
                  x : torch.Tensor,
                  return_embeddings : bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        latent = self.ae.encode(x)
        latent = self.__arange_subsets(latent)
        
        latent = latent.reshape(x.shape[0] // self.n_subsets, self.n_subsets, -1).mean(1)
        out = self.head(latent)

        if return_embeddings:
            return out, latent
        return out

