from typing import Union
from numpy.typing import NDArray

from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SCARFDataset(Dataset):
    def __init__(self, X: pd.DataFrame, 
                        Y: Union[NDArray[np.int_], NDArray[np.float_]] = None,
                        corruption_len: int = 0,
                        is_regression: bool = False
                        ) -> None:

        self.data = torch.FloatTensor(X.values)
        
        self.corruption_len = corruption_len
        self.n_sampling_candidate , self.n_features = X.shape

        if is_regression:
            self.label_class = torch.FloatTensor
        else:
            self.label_class = torch.LongTensor
            
        if Y is None:
            self.label = None
        else:
            self.label = self.label_class(Y)
            
            if self.label_class == torch.LongTensor:
                class_counts = [sum((self.label == i)) for i in set(self.label.numpy())]
                num_samples = len(self.label)

                class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
                self.weights = [class_weights[self.label[i]] for i in range(int(num_samples))]


    def __getitem__(self, idx):
        
        if self.label is None:
            if self.corruption_len:
                corruption_mask = torch.zeros((self.n_features), dtype=torch.bool)
                corruption_idx = torch.randperm(self.n_features)[:self.corruption_len]

                corruption_mask[corruption_idx] = True
                
                
                x_random = torch.randint(0, self.n_sampling_candidate, corruption_mask.shape)
                x_corrupted = torch.FloatTensor([self.data[:, i][x_random[i]] for i in range(self.n_features)])
                x_corrupted = torch.where(corruption_mask, x_corrupted, self.data[idx])
                return self.data[idx], x_corrupted
            return self.data[idx]
        else:
            return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
