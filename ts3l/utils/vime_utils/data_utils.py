import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset

import numpy as np
from dataclasses import asdict
from ts3l.utils.vime_utils import VIMEConfig

class VIMEDataset(Dataset):
    def __init__(self,
                X: pd.DataFrame,
                Y: Optional[Union[NDArray[np.int_], NDArray[np.float_]]] = None, 
                config: Optional[VIMEConfig] = None, 
                unlabeled_data: Optional[pd.DataFrame] = None, 
                continuous_cols: Optional[List] = None, 
                category_cols: Optional[List] = None, 
                u_label: Any = -1, 
                is_second_phase: Optional[bool] = False,
                is_regression: Optional[bool] = False, 
        ) -> None:
        """A dataset class for VIME that handles labeled and unlabeled data.

        This class is designed to manage data for the VIME, accommodating both labeled and unlabeled datasets
        for self- and semi-supervised learning scenarios. It supports regression and classification tasks.

        Args:
            X (pd.DataFrame): DataFrame containing the features of the labeled data.
            Y (Union[NDArray[np.int_], NDArray[np.float_]], optional): Numpy array containing the labels for the data. 
                Use integers for classification labels and floats for regression targets. Defaults to None.
            config (Dict[str, Any]): The given hyperparameter set for VIME.
            unlabeled_data (pd.DataFrame): DataFrame containing the features of the unlabeled data, used for 
                self-supervised learning. Defaults to None.
            continuous_cols (List, optional): List of continuous columns. Defaults to None.
            category_cols (List, optional): List of categorical columns. Defaults to None.
            u_label (int, optional): The specifier for unlabeled sample. Defaults to -1.
            is_second_phase (bool): The flag that determines whether the dataset is for first phase or second phase learning. Default is False.
            is_regression (bool, optional): Flag indicating whether the task is regression (True) or classification (False).
                Defaults to False.
        """
        
        self.label = None
        
        if not is_second_phase:
            self.__getitem = self.__first_phase_get_item
            
        else:
            self.__getitem = self.__second_phase_get_item
            
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
        
        if unlabeled_data is not None:
            X = pd.concat([X, unlabeled_data])
            
        self.cont_data = torch.FloatTensor(X[continuous_cols].values)
        self.cat_data = torch.FloatTensor(X[category_cols].values)
        
        self.continuous_cols = continuous_cols
        self.category_cols = category_cols

        if config is not None:
            self.config = config
        
        self.u_label = u_label
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
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
    
    def __mask_generator(self, p_m, x):
        """Generate mask vector.
        
        Args:
            - p_m: corruption probability
            - x: feature matrix
            
        Returns:
            - mask: binary mask matrix 
        """
        mask = np.random.binomial(1, p_m, x.shape)

        return np.expand_dims(mask, axis=0)

    def __pretext_generator(self, m, x, empirical_dist) -> Tuple[torch.LongTensor, torch.FloatTensor]:  
        """Generate corrupted samples.
        
        Args:
            m: mask matrix
            x: feature matrix
            
        Returns:
            m_new: final mask matrix after corruption
            x_tilde: corrupted feature matrix
        """
        
        # Parameters
        dim = x.shape[0]
        # Randomly (and column-wise) shuffle data
        x_bar = np.zeros([1, dim])

        rand_idx = np.random.randint(0, len(empirical_dist), size=dim)
        
        x_bar = np.array([empirical_dist[rand_idx[i], i] for i in range(dim)])
        
        # Corrupt samples
        x_tilde = x * (1-m) + x_bar * m  
        # Define new mask matrix
        m_new = 1 * (x != x_tilde)
        
        if dim != 1:
            return m_new.squeeze(), x_tilde.squeeze()
        else:
            return m_new.squeeze().reshape(1), x_tilde.squeeze().reshape(1)
    
    def __generate_x_tildes(self, cat_samples: torch.Tensor, cont_samples:torch.Tensor) -> torch.Tensor:
        """Generate x_tilde for consistency regularization

        Args:
            cat_samples (torch.Tensor): The categorical features to generate x_tilde
            cont_samples (torch.Tensor): The continuous features to generate x_tilde

        Returns:
            torch.Tensor: x_tilde for consistency regularization
        """
        m_unlab = self.__mask_generator(self.config.p_m, cat_samples)
        dcat_m_label, cat_x_tilde = self.__pretext_generator(m_unlab, cat_samples, self.cat_data)
        
        m_unlab = self.__mask_generator(self.config.p_m, cont_samples)
        cont_m_label, cont_x_tilde = self.__pretext_generator(m_unlab, cont_samples, self.cont_data)
        x_tilde = torch.concat((cat_x_tilde, cont_x_tilde)).float()
        
        return x_tilde
    
    def __first_phase_get_item(self, idx: int) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """Return a input and label pair

        Args:
            idx (int): The index of the data to sample

        Returns:
            Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]: A pair of input and label for first phase learning
        """
        cat_samples = self.cat_data[idx]
        m_unlab = self.__mask_generator(self.config.p_m, cat_samples)
        cat_m_label, cat_x_tilde = self.__pretext_generator(m_unlab, cat_samples, self.cat_data)

        cont_samples = self.cont_data[idx]
        m_unlab = self.__mask_generator(self.config.p_m, cont_samples)
        cont_m_label, cont_x_tilde = self.__pretext_generator(m_unlab, cont_samples, self.cont_data)

        m_label = torch.concat((cat_m_label, cont_m_label)).float()
        x_tilde = torch.concat((cat_x_tilde, cont_x_tilde)).float()

        x = torch.concat((cat_samples, cont_samples))

        return {
                "input" : x_tilde,
                "label" : (m_label, x)
                }

    def __second_phase_get_item(self, idx) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """Return a input and label pair

        Args:
            idx (int): The index of the data to sample

        Returns:
            Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]: A pair of input and label for second phase learning
        """
        cat_samples = self.cat_data[idx]
        cont_samples = self.cont_data[idx]
        x = torch.concat((cat_samples, cont_samples)).squeeze()
        if self.label is not None:
            
            if self.label[idx] == self.u_label:
                _xs = [x]
                _xs.extend([self.__generate_x_tildes(cat_samples, cont_samples) for _ in range(self.config.K)])
                xs = torch.stack(_xs)
                
                return {
                    "input" : xs,
                    "label" : self.label_class([self.u_label for _ in range(len(xs))])
                }
            else:
                return {
                    "input" : x.unsqueeze(0),
                    "label" : self.label[idx].unsqueeze(0)
                }
        else:
            return {
                    "input" : x,
                    "label" : self.u_label
            }
            
    def __len__(self):
        """Return the length of the dataset
        """
        return len(self.cat_data)

class VIMESemiSLCollateFN(object):
    """A callable class designed for batch processing, specifically tailored for the second phase learning with VIME. 
    It consolidates a batch of samples into a single dictionary with concatenated inputs and labels, suitable for model input.

    This class is meant to be used as a collate function in a DataLoader, where it efficiently organizes batch data 
    for training during second phase learning.
    """
    def __call__(self, batch):
        return {
            'input': torch.concat([x['input'] for x in batch], dim=0),
            'label': torch.concat([x['label'] for x in batch], dim=0)
        }