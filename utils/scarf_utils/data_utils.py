from typing import Union
from numpy.typing import NDArray

from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SCARFDataset(Dataset):
    def __init__(self, X: pd.DataFrame, 
                        Y: Union[NDArray[np.int_], NDArray[np.float_]],
                        label_class: Union[torch.FloatTensor, torch.LongTensor] = torch.LongTensor
                        ) -> None:

        self.data = torch.FloatTensor(X)
        
        assert label_class in [torch.LongTensor, torch.FloatTensor], 'The label_class must be one of the following: "torch.LongTensor", or "Torch.FloatTensor"'
        
        if Y is None:
            self.label = None
        else:
            self.label = label_class(Y)
            
            if label_class == torch.LongTensor:
                class_counts = [sum((self.label == i)) for i in set(self.label.numpy())]
                num_samples = len(self.label)

                class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
                self.weights = [class_weights[self.label[i]] for i in range(int(num_samples))]


    def __getitem__(self, idx):
        
        if self.label is None:
            return self.data[idx]
        else:
            return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
