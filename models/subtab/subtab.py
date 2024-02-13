import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from numpy.typing import NDArray

import itertools

from types import SimpleNamespace
from typing import Dict, Any, Tuple, List

class ShallowEncoder(nn.Module):
    def __init__(self,
                 feat_dim : int,
                 emb_dim : int,
                 n_subsets : int,
                 overlap_ratio : float,
    ) -> None:
        super().__init__()

        n_column_subset = int(feat_dim / n_subsets)
        n_overlap = int(overlap_ratio * n_column_subset)

        self.net = nn.Sequential(
            nn.Linear(n_column_subset + n_overlap, emb_dim),
            nn.LeakyReLU(),
        )
        
    def forward(self,
                x : torch.Tensor
    ) -> torch.Tensor:
        return self.net(x)

class ShallowDecoder(nn.Module):
    def __init__(self,
                 emb_dim : int,
                 out_dim : int
    ) -> None:
        super().__init__()

        self.net = nn.Linear(emb_dim, out_dim)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AutoEncoder(nn.Module):
    def __init__(self,
                 feat_dim : int,
                 emb_dim : int,
                 n_subsets : int,
                 overlap_ratio : float,
    ) -> None:
        super().__init__()

        self.encoder = ShallowEncoder(feat_dim, emb_dim, n_subsets, overlap_ratio)
        self.decoder = ShallowDecoder(emb_dim, feat_dim)

        self.projection_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
    
    def encode(self, x : torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, x : torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        projection = self.projection_net(latent)
        projection = F.normalize(projection, p = 2, dim = 1)
        x_recon = self.decode(latent)
        return latent, projection, x_recon

class SubTab(nn.Module):

    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 emb_dim: int,
                 
                 n_subsets: int,
                 overlap_ratio: float,
    ) -> None:
        super().__init__()

        self.feat_dim = input_dim
        
        self.n_subsets = n_subsets
        
        self.ae = AutoEncoder(self.feat_dim, emb_dim, n_subsets, overlap_ratio)
        
        self.head = nn.Sequential(
            nn.Linear(emb_dim, out_dim) ## dropout 넣?말
        )
        self.do_pretraining()
        
    

    def do_pretraining(self) -> None:
        self.forward = self.pretraining_step
    
    def do_finetunning(self) -> None:
        self.forward = self.finetunning_step

    def pretraining_step(self, x : torch.Tensor) -> torch.Tensor:

        latents, projections, x_recons = self.ae(x)
        
        return projections, x_recons, x
    
    def arange_subsets(self, latent: torch.FloatTensor) -> torch.FloatTensor:
        no, dim = latent.shape
        samples = int(no / self.n_subsets)
        
        latent = latent.reshape((self.n_subsets, samples, dim))
        return torch.concat([latent[:, i] for i in range(samples)])
    
    def finetunning_step(self, 
                  x : torch.Tensor,
                  return_embeddings : bool = False) -> torch.Tensor:

        latent = self.ae.encode(x)
        latent = self.arange_subsets(latent)
        
        latent = latent.reshape(x.shape[0] // self.n_subsets, self.n_subsets, -1).mean(1)
        out = self.head(latent)

        if return_embeddings:
            return out, latent
        return out

