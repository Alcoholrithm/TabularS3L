from typing import Tuple, Union, Optional, List
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class SubTabDataset(Dataset):
    def __init__(self, X: pd.DataFrame, 
                        Y: Optional[Union[NDArray[np.int_], NDArray[np.float64]]] = None,
                        unlabeled_data: Optional[pd.DataFrame] = None,
                        continuous_cols: Optional[List] = None, 
                        category_cols: Optional[List] = None,
                        is_regression: Optional[bool] = False
                        ) -> None:
        """A dataset class for SubTab that handles labeled and unlabeled data.

        This class is designed to manage data for the SubTab, accommodating both labeled and unlabeled datasets
        for self-supervised learning scenarios. It supports regression and classification tasks.

        Args:
            X (pd.DataFrame): DataFrame containing the features of the labeled data.
            Y (Optional[Union[NDArray[np.int_], NDArray[np.float64]]]): Numpy array containing the labels for the data. 
                Use integers for classification labels and floats for regression targets. Defaults to None.
            unlabeled_data (Optional[pd.DataFrame]): DataFrame containing the features of the unlabeled data, used for 
                self-supervised learning. Defaults to None.
            continuous_cols (List, optional): List of continuous columns. Defaults to None.
            category_cols (List, optional): List of categorical columns. Defaults to None.
            is_regression (Optional[bool]): Flag indicating whether the task is regression (True) or classification (False).
                Defaults to False.
        """
        if unlabeled_data is not None:
            X = pd.concat([X, unlabeled_data])
        
        cat_data = torch.FloatTensor(X[category_cols].values)
        cont_data = torch.FloatTensor(X[continuous_cols].values)
        
        self.data = torch.concat([cat_data, cont_data], dim=1)
        
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