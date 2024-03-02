import pandas as pd
from typing import Dict, Any, List, Union
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset

import numpy as np

class VIMEDataset(Dataset):
    def __init__(self,
                X: pd.DataFrame,
                Y: Union[NDArray[np.int_], NDArray[np.float_]] = None, 
                data_hparams: Dict[str, Any] = None, 
                unlabeled_data: pd.DataFrame = None, 
                continous_cols: List = None, 
                category_cols: List = None, 
                u_label = -1, 
                is_second_phase: bool = False,
                is_regression: bool = False, 
                is_test: bool = False
        ) -> None:
        """Initialize the VIME dataset

        Args:
            X (pd.DataFrame): The features of the labeled data
            Y (Union[NDArray[np.int_], NDArray[np.float_]]): The label of the labeled data
            data_hparams (Dict[str, Any]): The hyperparameters for consistency regularization
            unlabeled_data (pd.DataFrame, optional): The features of the unlabeled data. Defaults to None.
            continous_cols (List, optional): The list of continuous columns. Defaults to None.
            category_cols (List, optional): The list of categorical columns. Defaults to None.
            u_label (int, optional): The specifier for unlabeled sample. Defaults to -1.
            is_second_phase (bool): The flag that de... Default is False.
            is_regression (bool): Default is False.
            is_test (bool, optional): The flag that determines whether the dataset is for testing or not. Defaults to False.
        """
        
        if not is_second_phase:
            self.__getitem = self.__first_phase_get_item__
            
        else:
            self.__getitem = self.__second_phase_get_item__
            if is_regression:
                self.label_class = torch.FloatTensor
            else:
                self.label_class = torch.LongTensor
            
            if is_test is False:
            
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
            
        self.cont_data = torch.FloatTensor(X[continous_cols].values)
        self.cat_data = torch.FloatTensor(X[category_cols].values)
        
        self.continuous_cols = continous_cols
        self.category_cols = category_cols

        self.data_hparams = data_hparams
        
        self.u_label = u_label
        self.is_test = is_test
    
    def __getitem__(self, idx: int):
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

    def __pretext_generator(self, m, x, empirical_dist):  
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
    
    def __generate_x_tildes(self, cat_samples: torch.FloatTensor, cont_samples:torch.FloatTensor) -> torch.FloatTensor:
        """Generate x_tilde for consistency regularization

        Args:
            cat_samples (torch.FloatTensor): The categorical features to generate x_tilde
            cont_samples (torch.FloatTensor): The continuous features to generate x_tilde

        Returns:
            torch.FloatTensor: x_tilde for consistency regularization
        """
        m_unlab = self.__mask_generator(self.data_hparams["p_m"], cat_samples)
        dcat_m_label, cat_x_tilde = self.__pretext_generator(m_unlab, cat_samples, self.cat_data)
        
        m_unlab = self.__mask_generator(self.data_hparams["p_m"], cont_samples)
        cont_m_label, cont_x_tilde = self.__pretext_generator(m_unlab, cont_samples, self.cont_data)
        x_tilde = torch.concat((cat_x_tilde, cont_x_tilde)).float()
        
        return x_tilde
    
    def __first_phase_get_item__(self, idx: int):
        """Return a input and label pair

        Args:
            idx (int): The index of the data to sample

        Returns:
            Dict[str, Any]: A pair of input and label for self-supervised learning
        """
        cat_samples = self.cat_data[idx]
        m_unlab = self.__mask_generator(self.data_hparams["p_m"], cat_samples)
        cat_m_label, cat_x_tilde = self.__pretext_generator(m_unlab, cat_samples, self.cat_data)

        cont_samples = self.cont_data[idx]
        m_unlab = self.__mask_generator(self.data_hparams["p_m"], cont_samples)
        cont_m_label, cont_x_tilde = self.__pretext_generator(m_unlab, cont_samples, self.cont_data)

        m_label = torch.concat((cat_m_label, cont_m_label)).float()
        x_tilde = torch.concat((cat_x_tilde, cont_x_tilde)).float()

        x = torch.concat((cat_samples, cont_samples))

        return {
                "input" : x_tilde,
                "label" : (m_label, x)
                }

    def __second_phase_get_item__(self, idx):
        """Return a input and label pair

        Args:
            idx (int): The index of the data to sample

        Returns:
            Dict[str, Any]: A pair of input and label for semi-supervised learning
        """
        cat_samples = self.cat_data[idx]
        cont_samples = self.cont_data[idx]
        x = torch.concat((cat_samples, cont_samples)).squeeze()
        if self.is_test is False:
            
            if self.label[idx] == self.u_label:
                xs = [x]
                
                xs.extend([self.__generate_x_tildes(cat_samples, cont_samples) for _ in range(self.data_hparams["K"])])

                xs = torch.stack(xs)
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
    def __call__(self, batch):
        return {
            'input': torch.concat([x['input'] for x in batch], dim=0),
            'label': torch.concat([x['label'] for x in batch], dim=0)
        }