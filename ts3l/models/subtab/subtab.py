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

class ShallowDecoder(nn.Module):
    def __init__(self,
                 hidden_dim : int,
                 out_dim : int
    ) -> None:
        super().__init__()

        self.net = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.net(x)

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
        """Initialize SubTab

        Args:
            embedding_config (BaseEmbeddingConfig): Configuration for the embedding layer.
            backbone_config (BaseBackboneConfig): Configuration for the backbone network.
            output_dim (int): The dimensionality of the output.
            projection_dim (int): The dimensionality of the projector.
            shuffle (bool): Whether to shuffle the subsets. 
            n_subsets (int): The number of subsets to generate different views of the data.
            overlap_ratio (float): A hyperparameter that is to control the extent of overlapping between the subsets.
            mask_ratio (floats): Ratio of features to be masked as noise.
            noise_type (str): The type of noise to apply.
            noise_level (float): Intensity of Gaussian noise to be applied.
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
        self.n_overlap = int(self.overlap_ratio * self.n_feature_subset)
        self.n_overlap_dim = self.n_overlap
        
        if backbone_config.name == "mlp" and embedding_config.name == "feature_tokenizer": # type: ignore
            self.subset_dim = self.subset_dim * embedding_config.emb_dim # type: ignore
            self.n_overlap_dim = self.n_overlap_dim * embedding_config.emb_dim # type: ignore
        
        if embedding_config.name == "feature_tokenizer": # type: ignore
            self.__generate_subset_embedding = self.__post_embedding_subset # type: ignore
        else:
            self.__generate_subset_embedding = self.__pre_embedding_subset # type: ignore
        
        super(SubTab, self).__init__(embedding_config, backbone_config)
        
        self.projector = Projector(self.backbone_module.output_dim, projection_dim)
        self.decoder = ShallowDecoder(self.backbone_module.output_dim, self.embedding_module.input_dim)
        self.head = nn.Sequential(
            nn.Linear(self.backbone_module.output_dim, output_dim)
        )

    def _set_backbone_module(self, backbone_config):
        
        if hasattr(backbone_config, "input_dim"):
            backbone_config.input_dim = self.subset_dim + self.n_overlap_dim

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
        x = self.__generate_subset_embedding(x)
        x = self.encoder(x)
        projections = self.projector(x)
        x_recons = self.decoder(x)
        
        return projections, x_recons
    
    def _second_phase_step(self, 
                x : torch.Tensor,
                return_embeddings : bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x = self.__generate_subset_embedding(x)
        x = self.encoder(x)
        
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
        dim = self.n_feature_subset + self.n_overlap
        
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
        
        subset_column_indice_list = [self.column_idx[:(self.subset_dim + self.n_overlap_dim)]]
        subset_column_indice_list.extend([self.column_idx[range((i * self.subset_dim - self.n_overlap_dim), ((i + 1) * self.subset_dim))] for i in range(self.n_subsets)])
        
        subset_column_indice = np.array(subset_column_indice_list)
        subset_column_indice = subset_column_indice[permuted_order]
        
        if len(subset_column_indice) == 1:
            subset_column_indice = np.concatenate([subset_column_indice, subset_column_indice])

        x_tildes = torch.concat([self.__generate_x_tilde(x, subset_column_indice[i]) for i in range(self.n_subsets)]) # [subset1, subset2, ... ,  subsetN]

        return x_tildes