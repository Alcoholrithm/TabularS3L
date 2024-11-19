import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from numpy.typing import NDArray

from typing import Tuple, Union
from ts3l.models.common import TS3LModule
from ts3l.functional.subtab import arrange_tensors

from ts3l.utils import BaseEmbeddingConfig, BaseBackboneConfig
from ts3l.models.common import TS3LBackboneModule

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
                 input_dim : int,
                 output_dim : int,
                 hidden_dim : int,
                 n_subsets : int,
                 overlap_ratio : float,
    ) -> None:
        super().__init__()

        self.encoder = ShallowEncoder(input_dim, hidden_dim, n_subsets, overlap_ratio)
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
                 
                 shuffle: bool,
                 n_subsets: int,
                 overlap_ratio: float,
                 mask_ratio: float,
                 noise_type: str,
                 noise_level: float,
                 
                 **kwargs
    ) -> None:
        """_summary_

        Args:
            embedding_config (BaseEmbeddingConfig): _description_
            backbone_config (BaseBackboneConfig): _description_
            output_dim (int): _description_
            projection_dim (int): _description_
            shuffle (bool): _description_
            n_subsets (int): _description_
            overlap_ratio (float): _description_
            mask_ratio (floats): _description_
            noise_type (str): _description_
            noise_level (float): _description_
            
            transformer backbone이면 input_dim = embedding의 input_dim
            mlp backbone이고 feature_tokenizer 썼을 땐, input_dim == embedding의 input_dim * emb_dim
            mlp backbone이고 identity면 input_dim == embedding의 input_dim
        """
        
        self.shuffle = shuffle
        
        self.n_subsets = n_subsets
        self.overlap_ratio = overlap_ratio
        
        self.mask_ratio = mask_ratio
        self.noise_type = noise_type
        self.noise_level = noise_level
        
        self.input_dim = embedding_config.input_dim
        
        self.emb_dim = embedding_config.emb_dim if hasattr(embedding_config, "emb_dim") else 1
        
        if backbone_config.name == "mlp" and embedding_config.name == "feature_tokenizer": # type: ignore
            self.column_idx = np.arange(self.input_dim * embedding_config.emb_dim) # type: ignore
            
        else:
            self.column_idx = np.arange(self.input_dim)
        
        
        self.n_feature_subset = self.input_dim // self.n_subsets
        self.subset_dim = self.n_feature_subset
        
        # Number of overlapping features between subsets
        self.n_overlap = int(self.overlap_ratio * self.subset_dim) 
        # self.column_idx = np.array(range(self.input_dim * self.emb_dim))
        
        if backbone_config.name == "mlp" and embedding_config.name == "feature_tokenizer": # type: ignore
            self.subset_dim = self.subset_dim * embedding_config.emb_dim # type: ignore
        
        if embedding_config.name == "feature_tokenizer": # type: ignore
            self.__generate_subset_embedding = self.__post_embedding_subset # type: ignore
        else:
            self.__generate_subset_embedding = self.__pre_embedding_subset # type: ignore
        
        super(SubTab, self).__init__(embedding_config, backbone_config)
        
        
        # self.n_subsets = n_subsets
        self.projector = Projector(self.backbone_module.output_dim, projection_dim)
        self.decoder = ShallowDecoder(self.backbone_module.output_dim, self.embedding_module.input_dim)
        # self.__auto_encoder = AutoEncoder(self.embedding_module.output_dim, embedding_config.input_dim, hidden_dim, n_subsets, overlap_ratio)
        
        self.head = nn.Sequential(
            nn.Linear(self.backbone_module.output_dim, output_dim)
        )

    def _set_backbone_module(self, backbone_config):
        
        if hasattr(backbone_config, "input_dim"):
            backbone_config.input_dim = self.subset_dim + self.n_overlap

        self.backbone_module = TS3LBackboneModule(backbone_config)
        
    @property
    def encoder(self) -> nn.Module:
        return self.backbone_module
    
    def __generate_subset_embedding(self,x: torch.Tensor):
        pass
    
    def __post_embedding_subset(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_module(x)
        x = self.__generate_subset(x)
        return x
    
    def __pre_embedding_subset(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__generate_subset(x)
        x = self.embedding_module(x)
        return x
    
    def _first_phase_step(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(x.shape)
        x = self.__generate_subset_embedding(x)
        # x = self.embedding_module(x)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        projections = self.projector(x)
        x_recons = self.decoder(x)
        # latents, projections, x_recons = self.__auto_encoder(x)

        return projections, x_recons
    
    def _second_phase_step(self, 
                x : torch.Tensor,
                return_embeddings : bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x = self.__generate_subset_embedding(x)
        # x = self.embedding_module(x)
        x = self.encoder(x)
        
        # latent = self.__auto_encoder.encode(x)
        latent = arrange_tensors(x, self.n_subsets)
        
        latent = latent.reshape(x.shape[0] // self.n_subsets, self.n_subsets, -1).mean(1)
        out = self.head(latent)

        if return_embeddings:
            return out, latent
        return out
    
    def __generate_noisy_xbar(self, x : torch.Tensor) -> torch.Tensor:
        """Generates a noisy version of the input sample `x`.

        Args:
            x (torch.Tensor): The original sample.

        Returns:
            torch.Tensor: The noisy sample.
        """
        
        no = len(x)
        dim = self.n_feature_subset
        
        # Initialize corruption array
        x_bar = torch.zeros(x.shape).to(x.device)

        # Randomly (and column-wise) shuffle data
        if self.noise_type == "Swap":
            # Generate random permutations for all columns
            permutations = torch.stack([torch.randperm(no).reshape((-1, 1)) for _ in range(dim)], dim=1).to(x.device)
            permutations = permutations.repeat((1, 1, self.emb_dim))
            permutations = permutations.reshape(x.shape)

            # Use advanced indexing to permute the tensor
            x_bar = torch.gather(x, 0, permutations)
            
        elif self.noise_type == "Gaussian":
            noise = torch.normal(mean=0.0, std=self.noise_level, size=x.shape, device=x.device)
            x_bar = x + noise

        return x_bar
    
    def __generate_x_tilde(self, x: torch.Tensor, subset_column_idx: NDArray[np.int32]) -> torch.Tensor:
        """Generates a masked and potentially noisy subset of the input sample `x` based on the provided column indices.

        Args:
            x (torch.Tensor): The original sample.
            subset_column_idx (NDArray[np.int32]): Indices of columns to include in the subset.

        Returns:
            torch.Tensor: The processed subset of the sample.
        """

        x_bar = x[:, subset_column_idx]
        
        x_bar_noisy = self.__generate_noisy_xbar(x_bar)
        
        mask = torch.bernoulli(torch.full(x_bar.shape, self.mask_ratio, device=x_bar.device))
        
        x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
        
        return x_bar
    
    def __generate_subset(self,
                        x : torch.Tensor,
    ) -> torch.Tensor:
        """Generates subsets of features from the input sample `x`, applying shuffling, noise, and masking as configured.

        Args:
            x (torch.Tensor): The batch of samples.

        Returns:
            torch.Tensor: A tensor containing the generated subsets for the batch.
        """

        permuted_order = np.random.permutation(self.n_subsets) if self.shuffle else np.arange(self.n_subsets)

        subset_column_indice_list = [self.column_idx[:(self.subset_dim + self.n_overlap)]]
        subset_column_indice_list.extend([self.column_idx[range((i * self.subset_dim - self.n_overlap), ((i + 1) * self.subset_dim))] for i in range(self.n_subsets)])
        
        
        subset_column_indice = np.array(subset_column_indice_list)
        subset_column_indice = subset_column_indice[permuted_order]
        
        if len(subset_column_indice) == 1:
            subset_column_indice = np.concatenate([subset_column_indice, subset_column_indice])
        
        x_tildes = torch.concat([self.__generate_x_tilde(x, subset_column_indice[i]) for i in range(self.n_subsets)]) # [subset1, subset2, ... ,  subsetN]

        return x_tildes

