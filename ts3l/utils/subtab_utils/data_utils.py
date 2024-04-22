from typing import Dict, Any, Tuple, Union, Optional
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from dataclasses import asdict
from ts3l.utils.subtab_utils import SubTabConfig
class SubTabDataset(Dataset):
    def __init__(self, X: pd.DataFrame, 
                        Y: Optional[Union[NDArray[np.int_], NDArray[np.float_]]] = None,
                        unlabeled_data: Optional[pd.DataFrame] = None, 
                        is_regression: Optional[bool] = False
                        ) -> None:
        """A dataset class for SubTab that handles labeled and unlabeled data.

        This class is designed to manage data for the SubTab, accommodating both labeled and unlabeled datasets
        for self-supervised learning scenarios. It supports regression and classification tasks.

        Args:
            X (pd.DataFrame): DataFrame containing the features of the labeled data.
            Y (Optional[Union[NDArray[np.int_], NDArray[np.float_]]]): Numpy array containing the labels for the data. 
                Use integers for classification labels and floats for regression targets. Defaults to None.
            unlabeled_data (Optional[pd.DataFrame]): DataFrame containing the features of the unlabeled data, used for 
                self-supervised learning. Defaults to None.
            is_regression (Optional[bool]): Flag indicating whether the task is regression (True) or classification (False).
                Defaults to False.
        """
        if unlabeled_data is not None:
            X = pd.concat([X, unlabeled_data])
            
        self.data = torch.FloatTensor(X.values)
        
        self.label_class = torch.FloatTensor if is_regression else torch.LongTensor
            
        if Y is None:
            self.label = None
        else:
            self.label = self.label_class(Y)
            
            if not is_regression:
                class_counts = [sum((self.label == i)) for i in set(self.label.numpy())]
                num_samples = len(self.label)

                class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
                self.weights = [class_weights[self.label[i]] for i in range(int(num_samples))]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the feature and label tensors for a given index.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The feature tensor and the label 
            tensor (if available) for the given index. If labels are not provided, a placeholder is returned.
        """
        if self.label is None:
            return self.data[idx], torch.LongTensor([0])
        return self.data[idx], self.label[idx]
    
    def __len__(self) -> int:
        """Returns the total number of items in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.data)
    
class SubTabCollateFN(object):
    def __init__(self, config: SubTabConfig) -> None:
        """A collate function for SubTab to generate subsets of features for dataloaders.

        This collate function is designed to prepare batches for the SubTab model by generating subsets of features with
        optional noise and masking. It supports configurable shuffling, overlap, and masking of features.

        Args:
            config (SubTabConfig): The given hyperparameter set for SubTab.
        """
        
        self.config = config
            
        self.shuffle = self.config.shuffle
        
        self.n_subsets = self.config.n_subsets
        self.overlap_ratio = self.config.overlap_ratio
        self.mask_ratio = self.config.mask_ratio
        self.noise_type = self.config.noise_type
        self.noise_level = self.config.noise_level
        
        self.input_dim = self.config.input_dim
        self.n_column_subset = int(self.input_dim / self.n_subsets)
        # Number of overlapping features between subsets
        self.n_overlap = int(self.overlap_ratio * self.n_column_subset)
        self.column_idx = np.array(range(self.input_dim))
    
    def __generate_noisy_xbar(self, x : torch.Tensor) -> torch.Tensor:
        """Generates a noisy version of the input sample `x`.

        Args:
            x (torch.Tensor): The original sample.

        Returns:
            torch.Tensor: The noisy sample.
        """
        no, dim = x.shape
        
        # Initialize corruption array
        x_bar = torch.zeros([no, dim]).to(x.device)

        # Randomly (and column-wise) shuffle data
        if self.noise_type == "Swap":
            x_bar = torch.stack([x[np.random.permutation(no), i] for i in range(dim)], dim=1).to(x.device)
        elif self.noise_type == "Gaussian":
            x_bar = x + torch.normal(torch.zeros(x.shape), torch.full(x.shape, self.noise_level)).to(x.device)

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
        
        mask = torch.distributions.binomial.Binomial(total_count = 1, probs = self.mask_ratio).sample(x_bar.shape).to(x_bar.device)
        
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

        subset_column_indice_list = [self.column_idx[:self.n_column_subset + self.n_overlap]]
        subset_column_indice_list.extend([self.column_idx[range(i * self.n_column_subset - self.n_overlap, (i + 1) * self.n_column_subset)] for i in range(self.n_subsets)])
        
        
        subset_column_indice = np.array(subset_column_indice_list)
        subset_column_indice = subset_column_indice[permuted_order]
        
        if len(subset_column_indice) == 1:
            subset_column_indice = np.concatenate([subset_column_indice, subset_column_indice])
        
        x_tildes = torch.concat([self.__generate_x_tilde(x, subset_column_indice[i]) for i in range(self.n_subsets)]) # [subset1, subset2, ... ,  subsetN]

        return x_tildes
    
    def __call__(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Processes a batch of samples, and generating feature subsets.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the processed feature subsets, the original samples, and the labels.
        """
        y_recons = torch.stack([sample[0] for sample in batch])
        x = self.__generate_subset(y_recons)
        y = torch.tensor([sample[1] for sample in batch])
        
        return x, y_recons, y