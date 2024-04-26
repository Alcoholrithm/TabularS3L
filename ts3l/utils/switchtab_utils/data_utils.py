
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset

import numpy as np
from ts3l.utils.switchtab_utils import SwitchTabConfig

class SwitchTabDataset(Dataset):
    def __init__(self,
                X: pd.DataFrame,
                Y: Optional[Union[NDArray[np.int_], NDArray[np.float_]]] = None, 
                config: Optional[SwitchTabConfig] = None, 
                unlabeled_data: Optional[pd.DataFrame] = None, 
                u_label: Optional[Any] = -1,
                continuous_cols: Optional[List] = None, 
                category_cols: Optional[List] = None, 
                is_second_phase: Optional[bool] = False,
                is_regression: Optional[bool] = False, 
        ) -> None:
        """A dataset class for SwitchTab that handles labeled and unlabeled data.

        This class is designed to manage data for the SwitchTab, accommodating both labeled and unlabeled datasets
        for self- and semi-supervised learning scenarios. It supports regression and classification tasks.

        Args:
            X (pd.DataFrame): DataFrame containing the features of the labeled data.
            Y (Union[NDArray[np.int_], NDArray[np.float_]], optional): Numpy array containing the labels for the data. 
                Use integers for classification labels and floats for regression targets. Defaults to None.
            config (SwitchTabConfig): The given hyperparameter set for SwitchTab.
            unlabeled_data (pd.DataFrame): DataFrame containing the features of the unlabeled data, used for 
                self-supervised learning. Defaults to None.
            u_label (int, optional): The specifier for unlabeled sample. Defaults to -1.
            continuous_cols (List, optional): List of continuous columns. Defaults to None.
            category_cols (List, optional): List of categorical columns. Defaults to None.
            is_second_phase (bool): The flag that determines whether the dataset is for first phase or second phase learning. Default is False.
            is_regression (bool, optional): Flag indicating whether the task is regression (True) or classification (False).
                Defaults to False.
        """
        
        if config is not None:
            self.config = config
            
        if unlabeled_data is not None:
            X = pd.concat([X, unlabeled_data])
        
        cat_data = torch.FloatTensor(X[category_cols].values)
        cont_data = torch.FloatTensor(X[continuous_cols].values)
        
        self.data = torch.concat([cat_data, cont_data], dim=1)
        
        self.u_label = u_label
        self.label = None
        self.label_class = torch.FloatTensor if is_regression else torch.LongTensor
        
        if not is_second_phase:
            self.corruption_rate = self.config.corruption_rate if not is_second_phase else 0.0
            self.corruption_len = int(X.shape[1] * self.corruption_rate)
            self.n_sampling_candidate, self.n_features = X.shape
            
        if Y is not None:
            
            self.label = self.label_class(Y)
            
            if unlabeled_data is not None:
                self.label = torch.concat((self.label, self.label_class([u_label for _ in range(len(unlabeled_data))])), dim=0)
            
            if not is_regression:
                class_counts = [sum((self.label == i)) for i in set(self.label.numpy())]
                num_samples = len(self.label)

                class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
                self.weights = [class_weights[self.label[i]] for i in range(int(num_samples))]
            else:
                self.weights = [1.0 for _ in range(len(X))]
                if unlabeled_data is not None:
                    unlabeled_weight = len(X) / len(unlabeled_data)
                    self.weights.extend([unlabeled_weight for _ in range(len(unlabeled_data))])
        

        if not is_second_phase:
            self.__getitem = self.__first_phase_get_item # type: ignore
            self.idx_arr = np.arange(len(self.data))
            
        else:
            self.__getitem = self.__second_phase_get_item # type: ignore

    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return the feature tensor for a given index, along with an optional label or corrupted version of the tensor.

        For first phase learning, this method returns the original and a corrupted feature tensor pair with optional labels.
        For second phase learning, it returns the feature tensor and its corresponding label.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A pair of original and corrupted feature tensors with optional labels.
            Tuple[torch.Tensor, torch.Tensor]: A feature tensor and its corresponding label for second phase learning.
        """
        return self.__getitem(idx)
    
    def __first_phase_get_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the original and a corrupted feature tensor pair with optional labels for the first phase learning

        Args:
            idx (int): The index of the data to sample

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A pair of original and corrupted feature tensors with optional labels.
        """
        x_1 = self.data[idx]
        
        idx2 = np.random.choice(self.idx_arr)
        while idx2 == idx:
            idx2 = np.random.choice(self.idx_arr)
        x_2 = self.data[idx2]
        
        if self.label is not None:
            y_1, y_2 = self.label[idx], self.label[idx2] # type: ignore
        else:
            y_1, y_2 = self.label_class((self.u_label,)), self.label_class((self.u_label,))
        
        if self.corruption_len:
            xc_1 = self.__generate_corrupted_sample(x_1)
            xc_2 = self.__generate_corrupted_sample(x_2)
            return x_1, xc_1, y_1, x_2, xc_2, y_2
        else:
            return x_1, x_1, y_1, x_2, x_2, y_2

    def __second_phase_get_item(self, idx) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return a input and label pair for the second phase learning

        Args:
            idx (int): The index of the data to sample

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the feature tensor and corresponding label.
        """
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]
            
    def __len__(self):
        """Return the length of the dataset
        """
        return len(self.data)
    
    def __generate_corrupted_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Return a corrupted version of the feature tensor for a given sample.

        Args:
            x (torch.Tensor): A feature tensor to be corrupted.

        Returns:
            torch.Tensor: A corrupted feature tensor.
        """
        corruption_mask = torch.zeros((self.n_features), dtype=torch.bool)
        corruption_idx = torch.randperm(self.n_features)[:self.corruption_len]

        corruption_mask[corruption_idx] = True
        
        x_random = torch.randint(0, self.n_sampling_candidate, corruption_mask.shape)
        _x_corrupted = torch.FloatTensor([self.data[:, i][x_random[i]] for i in range(self.n_features)])
        x_corrupted = torch.where(corruption_mask, _x_corrupted, x)
        return x_corrupted
    
class SwitchTabFirstPhaseCollateFN(object):
    """A callable class designed for batch processing, specifically tailored for the first phase learning with SwitchTab. 
    It restructures the batch by concatenating certain elements to form new tensors.
    This class is meant to be used as a collate function in a DataLoader, where it efficiently organizes batch data 
    for training during first phase learning.
    """
    def __call__(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process the input batch to generate tensors by concatenating specific elements.

        Args:
            batch (Tuple): The batch to process. The structure of each element is expected to be 
                (x1, x1_corrupted, y1, x2, x2_corrupted, y2).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three tensors:
                - xs (Tensor): Concatenation of the x1 and x2 from each tuple in the batch.
                - xcs (Tensor): Concatenation of the x1_corrupted and x2_corrupted.
                - ys (Tensor): Concatenation of the y1 and y2.
        """
        xs = torch.concat([torch.stack([x[0] for x in batch]), torch.stack([x[3] for x in batch])])
        xcs = torch.concat([torch.stack([x[1] for x in batch]), torch.stack([x[4] for x in batch])])
        ys = torch.concat([torch.stack([x[2] for x in batch]), torch.stack([x[5] for x in batch])])
        return xs, xcs, ys