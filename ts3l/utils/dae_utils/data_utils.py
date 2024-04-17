from typing import Dict, Any, Tuple, Union, Optional
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from dataclasses import asdict
from ts3l.utils.dae_utils import DAEConfig

class DAEDataset(Dataset):
    def __init__(self, X: pd.DataFrame, 
                        Y: Optional[Union[NDArray[np.int_], NDArray[np.float_]]] = None,
                        unlabeled_data: Optional[pd.DataFrame] = None,
                        is_regression: Optional[bool] = False
                        ) -> None:
        """A dataset class for DenoisingAutoEncoder that handles labeled and unlabeled data.

        This class is designed to manage data for the DenoisingAutoEncoder, accommodating both labeled and unlabeled datasets
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
            return self.data[idx]
        
        return self.data[idx], self.label[idx]
    
    def __len__(self) -> int:
        """Returns the total number of items in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.data)
    
class DAECollateFN(object):
    def __init__(self, config: DAEConfig) -> None:
        """A collate function for DenoisingAutoEncoder to generate noisy x for dataloaders.

        Args:
            config (SubTabConfig): The given hyperparameter set for DenoisingAutoEncoder.
        """
        
        self.config = asdict(config)
        
        self.noise_ratio = self.config["noise_ratio"]
        self.noise_type = self.config["noise_type"]
        self.noise_level = self.config["noise_level"]
    
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
    
    def __generate_x_tilde(self, x: torch.Tensor) -> torch.Tensor:
        """Generates noisy samples for the given x.

        Args:
            x (torch.Tensor): The original sample.

        Returns:
            torch.Tensor: The noisy samples.
        """
        x_bar_noisy = self.__generate_noisy_xbar(x)
        
        mask = torch.distributions.binomial.Binomial(total_count = 1, probs = self.mask_ratio).sample(x_bar.shape).to(x_bar.device)
        
        x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
        
        return x_bar
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates noisy x from the given input x.

        Args:
            x (torch.Tensor): A batch of the original samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the original samples, and the noisy samples.
        """

        x_bar = self.__generate_x_tilde(x)
        
        return x, x_bar