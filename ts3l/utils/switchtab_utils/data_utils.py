
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset

import numpy as np
from dataclasses import asdict
from ts3l.utils.vime_utils import VIMEConfig

class SwitchTabDataset(Dataset):
    def __init__(self,
                X: pd.DataFrame,
                Y: Optional[Union[NDArray[np.int_], NDArray[np.float_]]] = None, 
                config: Optional[VIMEConfig] = None, 
                unlabeled_data: Optional[pd.DataFrame] = None, 
                u_label: Optional[Any] = -1,
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
            config (Dict[str, Any]): The given hyperparameter set for SwitchTab.
            unlabeled_data (pd.DataFrame): DataFrame containing the features of the unlabeled data, used for 
                self-supervised learning. Defaults to None.
            u_label (int, optional): The specifier for unlabeled sample. Defaults to -1.
            is_second_phase (bool): The flag that determines whether the dataset is for first phase or second phase learning. Default is False.
            is_regression (bool, optional): Flag indicating whether the task is regression (True) or classification (False).
                Defaults to False.
        """
        
        if unlabeled_data is not None:
            X = pd.concat([X, unlabeled_data])
        
        self.data = torch.FloatTEnsor(X.values)
        self.u_label = u_label
        self.label = None
        self.label_class = torch.FloatTensor if is_regression else torch.LongTensor
        
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
            self.__getitem = self.__first_phase_get_item
            self.idx_arr = np.arange(len(self.data))
            
        else:
            self.__getitem = self.__second_phase_get_item

        if config is not None:
            self.config = config

    
    def __getitem__(self, idx: int) -> Dict[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieves the feature and label tensors for a given index. The structure of the label 
            varies depending on the learning phase: 

            - In the first phase, the label is a tuple containing a mask and the original feature tensor.
            - In the second phase, the label is either a token indicating unlabeled samples or the actual label for labeled samples.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            Dict[Any, torch.Tensor]: The feature tensor and the label for the given index. 
            For first phase learning, label is a tuple of mask and original feature tensor.
            For second phase learning, label is a one of a token for unlabeled samples or a label of it.
        """
        return self.__getitem(idx)
    
    def __first_phase_get_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return input and label pairs for the first phase learning

        Args:
            idx (int): The index of the data to sample

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the feature vector and corresponding label.
        """
        x_1 = self.data[idx]
        y_1 = self.label[idx]
        
        idx2 = np.random.choice(self.idx_arr)
        while idx2 == idx:
            idx2 = np.random.choice(self.idx_arr)
        x_2 = self.data[idx2]
        y_2 = self.label[idx2]
        
        return x_1, x_2, y_1, y_2

    def __second_phase_get_item(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a input and label pair for the second phase learning

        Args:
            idx (int): The index of the data to sample

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the feature vector and corresponding label.
        """
        return self.data[idx], self.label[idx]
            
    def __len__(self):
        """Return the length of the dataset
        """
        return len(self.data)