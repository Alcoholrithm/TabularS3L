import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Tuple, Union
from ts3l.models.common import TS3LModule
from ts3l.functional.subtab import arrange_tensors

from ts3l.utils import BaseEmbeddingConfig, BaseBackboneConfig
from ts3l.models.common import TS3LBackboneModule

class ShallowEncoder(nn.Module):
    def __init__(self,
                 backbone_config: BaseBackboneConfig,
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
                 input_dim : int,
                 backbone_config: BaseBackboneConfig,
                 output_dim : int,
                 hidden_dim : int,
                 n_subsets : int,
                 overlap_ratio : float,
    ) -> None:
        super().__init__()

        self.encoder = ShallowEncoder(backbone_config, input_dim, hidden_dim, n_subsets, overlap_ratio)
        self.decoder = ShallowDecoder(hidden_dim, output_dim)

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

class Projector(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.projection_net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, x):
        x = self.projection_net(x)
        return F.normalize(x, p = 2, dim = 1)
class SubTab(TS3LModule):

    def __init__(self,
                 embedding_config: BaseEmbeddingConfig,
                 backbone_config: BaseBackboneConfig,
                 output_dim: int,
                 projection_dim: int,
                 
                 n_subsets: int,
                 overlap_ratio: float,
                 **kwargs
    ) -> None:
        self.n_column_subset = int(embedding_config.output_dim / n_subsets)
        self.n_overlap = int(overlap_ratio * self.n_column_subset)
        
        super(SubTab, self).__init__(embedding_config, backbone_config)
        
        
        self.n_subsets = n_subsets
        self.projector = Projector(backbone_config.output_dim, projection_dim)
        self.decoder = ShallowDecoder(backbone_config.output_dim, self.embedding_module.output_dim)
        # self.__auto_encoder = AutoEncoder(self.embedding_module.output_dim, embedding_config.input_dim, hidden_dim, n_subsets, overlap_ratio)
        
        self.head = nn.Sequential(
            nn.Linear(backbone_config.output_dim, output_dim)
        )

    def _set_backbone_module(self, backbone_config):

        if hasattr(backbone_config, "input_dim"):
            backbone_config.input_dim = self.n_column_subset + self.n_overlap

        self.backbone_module = TS3LBackboneModule(backbone_config)
    @property
    def encoder(self) -> nn.Module:
        return self.backbone_module
    
    def _first_phase_step(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(x.shape)
        x = self.embedding_module(x)
        # print(x.shape)
        x = self.backbone_module(x)
        # print(x.shape)
        projections = self.projector(x)
        x_recons = self.decoder(x)
        # latents, projections, x_recons = self.__auto_encoder(x)

        return projections, x_recons
    
    def _second_phase_step(self, 
                x : torch.Tensor,
                return_embeddings : bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x = self.embedding_module(x)
        x = self.backbone_module(x)
        
        # latent = self.__auto_encoder.encode(x)
        latent = arrange_tensors(x, self.n_subsets)
        
        latent = latent.reshape(x.shape[0] // self.n_subsets, self.n_subsets, -1).mean(1)
        out = self.head(latent)

        if return_embeddings:
            return out, latent
        return out

