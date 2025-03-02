import pandas as pd
from typing import List, Union, Optional, Tuple, Any, Callable
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset
from torch import Tensor

import numpy as np
from ts3l.utils.tabularbinning_utils import TabularBinningConfig


class TabularBinningDataset(Dataset):
    n_bin: int
    pretext_task: str
    data: Tensor
    binned_data: Optional[Tensor]
    is_regression: bool
    is_second_phase: bool
    label_class: Any  # torch.FloatTensor or torch.LongTensor
    label: Optional[Tensor]
    weights: Optional[List[float]]

    def __init__(self, config: TabularBinningConfig,
                 X: pd.DataFrame,
                 Y: Optional[Union[NDArray[np.int_],
                                   NDArray[np.float64]]] = None,
                 unlabeled_data: Optional[pd.DataFrame] = None,
                 continuous_cols: List[str] = [],
                 category_cols: List[str] = [],
                 is_regression: bool = False,
                 is_second_phase: bool = False,
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

        self.n_bin = config.n_bin
        self.pretext_task = config.pretext_task

        if unlabeled_data is not None:
            X = pd.concat([X, unlabeled_data], copy=False)

        self.data = torch.from_numpy(
            X[category_cols + continuous_cols].values).float()

        self.is_regression = is_regression
        self.is_second_phase = is_second_phase

        if not self.is_second_phase:
            self.binned_data = self.__get_binned_data(self.data)

        self.label_class = torch.FloatTensor if is_regression else torch.LongTensor

        if Y is not None:
            self.label = self.label_class(Y)

            if self.label_class == torch.LongTensor:
                _, counts = torch.unique(self.label, return_counts=True)
                num_samples = len(self.label)
                class_weights = num_samples / counts
                self.weights = class_weights[self.label]
        else:
            self.label = None

    def __binning_feature(self, feature: Tensor) -> Tensor:
        """Discretizes feature data into a specified number of bins.

        Args:
            feature (torch.Tensor): Feature vector to be discretized

        Returns:
            torch.Tensor: Discretized feature vector with values ranging from 0 to (n_bin-1)
        """
        unique_vals = torch.unique(feature)
        if len(unique_vals) < self.n_bin:
            bins = unique_vals
            return torch.bucketize(feature, bins[1:], right=False)
        else:
            percentiles = torch.linspace(0, 100, self.n_bin + 1)[1:-1]
            bins = torch.quantile(feature, percentiles / 100)
            bins = torch.cat([bins, torch.tensor([float('inf')])])
            return torch.bucketize(feature, bins, right=False)

    def __get_binned_data(self, data: Tensor) -> Tensor:
        """Performs binning on all features and applies normalization if required.

        Args:
            data (torch.Tensor): Original feature matrix of shape (n_samples, n_features)

        Returns:
            torch.Tensor: Discretized and normalized feature matrix
        """
        binned_features = torch.stack([self.__binning_feature(data[:, idx])
                                       for idx in range(data.shape[1])], dim=1)

        if self.pretext_task == "BinRecon":
            binned_features = binned_features.float()
            mean = binned_features.mean(dim=0, keepdim=True)
            std = binned_features.std(dim=0, keepdim=True)
            binned_features = (binned_features - mean) / (std + 1e-10)

        return binned_features

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
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
            return self.data[idx], self.binned_data[idx]  # type: ignore

    def __len__(self) -> int:
        """Returns the total number of items in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.data)


class TabularBinningFirstPhaseCollateFN(object):
    p_m: float
    constant_x_bar: Optional[Tensor]
    __noise_generator: Callable[[Tensor], Tensor]
    __custom_call: Callable[[
        List[Tuple[Tensor, Tensor]]], Tuple[Tensor, Tensor]]

    """A callable class designed for batch processing, specifically tailored for the first phase learning with TabularBinning.
    This class is meant to be used as a collate function in a DataLoader, where it efficiently organizes batch data
    for training during first phase learning.
    """

    def __init__(self, config: TabularBinningConfig,
                 constant_x_bar: Optional[NDArray[np.float64]] = None) -> None:
        """Initialize the collate function for first phase learning.

        Args:
            config (TabularBinningConfig): Configuration object for TabularBinning
            constant_x_bar (np.NDArray, optional): Feature-wise mean values for constant noise.
                Required when mask_type is 'constant'.

        Raises:
            ValueError: When mask_type is 'constant' and constant_x_bar is None
        """

        if config.pretext_task == "BinXent":
            self.__custom_call = self.__return_flatten_label

        if config.mask_type == "constant" and constant_x_bar is None:
            raise ValueError(
                "constant_x_bar cannot be None when the mask type is constant")

        self.p_m = config.p_m

        if config.mask_type == "constant":
            self.__noise_generator = self.__constant_noise_generator
            self.constant_x_bar = torch.as_tensor(constant_x_bar).unsqueeze(0)
        else:
            self.__noise_generator = self.__random_noise_generator

    def __mask_generator(self, x: Tensor) -> Tensor:
        """Generates a mask vector for feature corruption.

        Args:
            x (torch.Tensor): Input feature tensor

        Returns:
            torch.Tensor: Binary mask tensor of zeros and ones
        """
        return torch.bernoulli(torch.full_like(x, self.p_m))

    def __random_noise_generator(self, x: Tensor) -> Tensor:
        """Generates randomly shuffled noise samples.

        Args:
            x (torch.Tensor): Original feature tensor

        Returns:
            torch.Tensor: Feature-wise randomly shuffled tensor
        """
        no, dim = x.shape
        idx = torch.stack([torch.randperm(no) for _ in range(dim)], dim=1)
        return x[idx, torch.arange(dim)]

    def __constant_noise_generator(self, x: Tensor) -> Tensor:
        """Generates constant noise using mean values.

        Args:
            x (torch.Tensor): Original feature tensor

        Returns:
            torch.Tensor: Tensor filled with feature-wise mean values
        """
        return self.constant_x_bar.repeat(x.size(0), 1).to(dtype=x.dtype, device=x.device)  # type: ignore

    def __call__(
            self, batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        return self.__custom_call(batch)

    def __x_generator(self, batch: List[Tuple[Tensor, Tensor]]) -> Tensor:
        """Generates masked input by combining original and noise data.

        Args:
            batch (Tuple): Batch of (feature, label) pairs

        Returns:
            torch.Tensor: Feature tensor with applied masking and noise
        """
        x = torch.stack([x for x, _ in batch])
        mask = self.__mask_generator(x)
        x_bar = self.__noise_generator(x)

        return x * (1 - mask) + x_bar * mask

    def __custom_call(self,  # type: ignore
                      batch: List[Tuple[Tensor, Tensor]]
                      ) -> Tuple[Tensor, Tensor]:
        """Default batch processing function.

        Args:
            batch (Tuple): Batch of (feature, label) pairs

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (transformed features, labels)
        """
        return self.__x_generator(batch), torch.stack([y for _, y in batch])

    def __return_flatten_label(
            self, batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Batch processing function for BinXent learning.

        Args:
            batch (Tuple): Batch of (feature, label) pairs

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (transformed features, flattened labels)
        """
        return self.__x_generator(batch), torch.cat([y for _, y in batch])
