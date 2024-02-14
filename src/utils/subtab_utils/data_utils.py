from typing import Dict, Any, Tuple, Union
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class SubTabDataset(Dataset):
    def __init__(self, X: pd.DataFrame, 
                        Y: Union[NDArray[np.int_], NDArray[np.float_]] = None,
                        is_regression: bool = False
                        ) -> None:
        
        self.data = torch.FloatTensor(X.values)
        
        if is_regression:
            self.label_class = torch.FloatTensor
        else:
            self.label_class = torch.LongTensor
            
        if Y is None:
            self.label = None
        else:
            self.label = self.label_class(Y)
            
            if not is_regression:
                class_counts = [sum((self.label == i)) for i in set(self.label.numpy())]
                num_samples = len(self.label)

                class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
                self.weights = [class_weights[self.label[i]] for i in range(int(num_samples))]
    
    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, Union[torch.FloatTensor, torch.LongTensor]]:
        if self.label is None:
            return self.data[idx], [0]
        return self.data[idx], self.label[idx]
    
    def __len__(self) -> int:
        return len(self.data)
    
class SubTabCollateFN(object):
    def __init__(self, data_hparams: Dict[str, Any]) -> None:
        self.shuffle = data_hparams["shuffle"] if "shuffle" in data_hparams.keys() else False
        
        self.n_subsets = data_hparams["n_subsets"]
        self.overlap_ratio = data_hparams["overlap_ratio"]
        self.mask_ratio = data_hparams["mask_ratio"]
        self.noise_type = data_hparams["noise_type"]
        self.noise_level = data_hparams["noise_level"]
        
        self.n_column = data_hparams["n_column"]
        self.n_column_subset = int(self.n_column / self.n_subsets)
        # Number of overlapping features between subsets
        self.n_overlap = int(self.overlap_ratio * self.n_column_subset)
        self.column_idx = np.array(range(self.n_column))
    
    def generate_noisy_xbar(self, x : torch.FloatTensor) -> torch.Tensor:
        """Generates noisy version of the samples x
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
    
    def generate_x_tilde(self, x: torch.FloatTensor, subset_column_idx: NDArray[np.int32]) -> torch.FloatTensor:
        x_bar = x[:, subset_column_idx]
        x_bar_noisy = self.generate_noisy_xbar(x_bar)
        
        mask = torch.distributions.binomial.Binomial(total_count = 1, probs = self.mask_ratio).sample(x_bar.shape).to(x_bar.device)
        
        x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
        
        return x_bar
    
    def generate_subset(self,
                        x : torch.Tensor,
    ) -> torch.FloatTensor:

        permuted_order = np.random.permutation(self.n_subsets) if self.shuffle else range(self.n_subsets)

        subset_column_indice = [self.column_idx[:self.n_column_subset + self.n_overlap]]
        subset_column_indice.extend([self.column_idx[range(i * self.n_column_subset - self.n_overlap, (i + 1) * self.n_column_subset)] for i in range(self.n_subsets)])
        
        
        subset_column_indice = np.array(subset_column_indice)
        subset_column_indice = subset_column_indice[permuted_order]
        
        if len(subset_column_indice) == 1:
            subset_column_indice = np.concatenate([subset_column_indice, subset_column_indice])
        
        x_tildes = torch.concat([self.generate_x_tilde(x, subset_column_indice[i]) for i in range(self.n_subsets)]) # [subset1, subset2, ... ,  subsetN]

        return x_tildes
    
    def __call__(self, batch):
        y_recons = torch.stack([sample[0] for sample in batch])
        x = self.generate_subset(y_recons)
        y = torch.tensor([sample[1] for sample in batch])
        
        return x, y_recons, y