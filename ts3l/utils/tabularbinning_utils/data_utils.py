import pandas as pd
from typing import List, Union, Optional, Tuple
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset

import numpy as np
from ts3l.utils.tabularbinning_utils import TabularBinningConfig

class TabularBinningDataset(Dataset):
    def __init__(self, config: TabularBinningConfig,
                        X: pd.DataFrame, 
                        Y: Optional[Union[NDArray[np.int_], NDArray[np.float64]]] = None,
                        unlabeled_data: Optional[pd.DataFrame] = None, 
                        continuous_cols: Optional[List] = None, 
                        category_cols: Optional[List] = None,
                        is_regression: Optional[bool] = False,
                        is_second_phase: Optional[bool] = False,
                        ) -> None:
        """A dataset class for TabularBinning that handles labeled and unlabeled data.

        This class is designed to manage data for the TabularBinning, accommodating both labeled and unlabeled datasets
        for self-supervised learning scenarios. It supports regression and classification tasks.

        Args:
            X (pd.DataFrame): DataFrame containing the features of the labeled data.
            Y (Union[NDArray[np.int_], NDArray[np.float64]], optional): Numpy array containing the labels for the data. 
                Use integers for classification labels and floats for regression targets. Defaults to None.
            unlabeled_data (pd.DataFrame): DataFrame containing the features of the unlabeled data, used for 
                self-supervised learning. Defaults to None.
            config (TabularBinningConfig): The given hyperparameter set for TabularBinning.
            continuous_cols (List, optional): List of continuous columns. Defaults to None.
            category_cols (List, optional): List of categorical columns. Defaults to None.
            is_regression (bool, optional): Flag indicating whether the task is regression (True) or classification (False).
                Defaults to False.
            is_second_phase (bool, optional): Flag indicating whether the dataset is for first phase or second phase learning. 
                Default is False.
        """
        
        self.n_bins = config.n_bins
        self.pretext_task = config.pretext_task

        if unlabeled_data is not None:
            X = pd.concat([X, unlabeled_data])
        
        cat_data = torch.FloatTensor(X[category_cols].values)
        cont_data = torch.FloatTensor(X[continuous_cols].values)
        
        self.data = torch.concat([cat_data, cont_data], dim=1)

        self.is_regression = is_regression
        self.is_second_phase = is_second_phase

        if not self.is_second_phase:
            self.binned_data = self.__get_binned_data(self.data)

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


    def __binning_feature(self, feature: torch.Tensor) -> NDArray[np.int_]:
        if len(torch.unique(feature)) < self.n_bins:
            bins = feature.unique()
            return np.digitize(feature, bins=bins[1:], right=False)
        else:
            bins = np.percentile(feature, np.arange(0, 100, step= 100 / self.n_bins))
            bins[-1] = np.inf
            return np.digitize(feature, bins=bins[1:], right=False)
    
    def __get_binned_data(self, data: torch.Tensor):
        binned_features = []
        for idx in range(data.shape[1]):
            binned_features.append(self.__binning_feature(data[:, idx]))

        binned_data = torch.from_numpy(np.stack(binned_features, axis=-1)).type(torch.int64)

        if self.pretext_task == "BinRecon":
            binned_data = binned_data.type(torch.float32)
            mean = binned_data.mean(0, keepdim=True)[0]
            std = binned_data.std(0, keepdim=True)[0]
            binned_data = (binned_data - mean) / (std + 1e-10)
        
        return binned_data

    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the feature tensor for a given index, along with an label.

        In the first phase of learning, this method return the original feature tensor together with its binned version as the label.
        In the second phase of learning, it returns the feature tensor alongside its corresponding label.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
        torch.Tensor: Only the feature tensor for the given index, suitable for test-time inference.
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the original feature tensor and its binned version for the first phase learning, 
                or the feature tensor and its corresponding label for second phase learning,
                or the feature tensor and dummy label for test-time inference.
        """

        if self.is_second_phase:
            if self.label is None:
                return self.data[idx], torch.tensor(-1)
            else:
                return self.data[idx], self.label[idx]
        else:
            return self.data[idx], self.binned_data[idx]

    def __len__(self):
        """Returns the total number of items in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.data)