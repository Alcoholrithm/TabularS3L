from typing import Union, Tuple, Optional, List
from numpy.typing import NDArray

from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataclasses import asdict
from ts3l.utils.scarf_utils import SCARFConfig

class SCARFDataset(Dataset):
    def __init__(self, X: pd.DataFrame, 
                        Y: Optional[Union[NDArray[np.int_], NDArray[np.float64]]] = None,
                        unlabeled_data: Optional[pd.DataFrame] = None, 
                        config: Optional[SCARFConfig] = None,
                        continuous_cols: Optional[List] = None, 
                        category_cols: Optional[List] = None,
                        is_regression: Optional[bool] = False,
                        is_second_phase: Optional[bool] = False,
                        ) -> None:
        """A dataset class for SCARF that handles labeled and unlabeled data.

        This class is designed to manage data for the SCARF, accommodating both labeled and unlabeled datasets
        for self-supervised learning scenarios. It supports regression and classification tasks.

        Args:
            X (pd.DataFrame): DataFrame containing the features of the labeled data.
            Y (Union[NDArray[np.int_], NDArray[np.float64]], optional): Numpy array containing the labels for the data. 
                Use integers for classification labels and floats for regression targets. Defaults to None.
            unlabeled_data (pd.DataFrame): DataFrame containing the features of the unlabeled data, used for 
                self-supervised learning. Defaults to None.
            config (SCARFConfig): The given hyperparameter set for SCARF.
            continuous_cols (List, optional): List of continuous columns. Defaults to None.
            category_cols (List, optional): List of categorical columns. Defaults to None.
            is_regression (bool, optional): Flag indicating whether the task is regression (True) or classification (False).
                Defaults to False.
            is_second_phase (bool, optional): Flag indicating whether the dataset is for first phase or second phase learning. 
                Default is False.
        """
        
        if config is not None:
            self.config = config
            
        if unlabeled_data is not None:
            X = pd.concat([X, unlabeled_data])
        
        cat_data = torch.FloatTensor(X[category_cols].values)
        cont_data = torch.FloatTensor(X[continuous_cols].values)
        
        self.data = torch.concat([cat_data, cont_data], dim=1)
        
        self.corruption_rate = self.config.corruption_rate if not is_second_phase else 0.0
        self.corruption_len = int(X.shape[1] * self.corruption_rate)
        if not is_second_phase:
            self.corruption_len = max(1, self.corruption_len)
            
        self.n_sampling_candidate , self.n_features = X.shape

        self.is_regression = is_regression
        
        self.label_class = torch.FloatTensor if is_regression else torch.LongTensor
            
        if Y is None:
            self.label = None
        else:
            self.label = self.label_class(Y)
            
            if self.label_class == torch.LongTensor:
                class_counts = [sum((self.label == i)) for i in set(self.label.numpy())]
                num_samples = len(self.label)

                class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
                self.weights = [class_weights[self.label[i]] for i in range(int(num_samples))]


    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the feature tensor for a given index, along with an label or corrupted version of the feature.

        For first phase learning, this method can return either the original feature tensor or, if corruption is applied, 
        a corrupted version of the feature tensor. For second phase learning, it returns the feature tensor and its corresponding label.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            torch.Tensor: Only the feature tensor for the given index, suitable for test-time inference.
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the original and corrupted feature tensors for first phase learning, 
                or the feature tensor and its corresponding label for second phase learning,
                or the feature tensor and dummy label for test-time inference.
        """
        if self.label is None:
            if self.corruption_len:
                corruption_mask = torch.zeros((self.n_features), dtype=torch.bool, device=self.data.device)
                corruption_idx = torch.randperm(self.n_features, device=self.data.device)[:self.corruption_len]
                corruption_mask[corruption_idx] = True
                
                # Generate random indices for each feature
                x_random = torch.randint(0, self.n_sampling_candidate, (self.n_features,), device=self.data.device)

                # Use advanced indexing to create the corrupted features
                _x_corrupted = self.data[x_random, torch.arange(self.n_features, device=self.data.device)].float()
                
                x_corrupted = torch.where(corruption_mask, _x_corrupted, self.data[idx])
                
                return self.data[idx], x_corrupted
            return self.data[idx], torch.tensor(-1)
        else:
            return self.data[idx], self.label[idx]

    def __len__(self):
        """Returns the total number of items in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.data)
