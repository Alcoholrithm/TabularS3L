from typing import Dict, Any, Tuple, Union, Type

from .base_module import TS3LLightining
from ts3l.models import SubTab
from ts3l.utils.subtab_utils import JointLoss

import torch
from ts3l.utils.subtab_utils import SubTabConfig

class SubTabLightning(TS3LLightining):
    
    def __init__(self, config: SubTabConfig) -> None:
        """Initialize the pytorch lightining module of SubTab

        Args:
            config (SubTabConfig): The configuration of SubTabLightning.
        """
        super(SubTabLightning, self).__init__(config)

    def _initialize(self, config: Dict[str, Any]):
        """Initializes the model with specific hyperparameters and sets up various components of SubTabLightning.

        Args:
            config (Dict[str, Any]): The given hyperparameter set for SubTab. 
        """
        self.first_phase_loss = JointLoss(
                                        config["tau"],
                                        config["n_subsets"],
                                        config["use_contrastive"],
                                        config["use_distance"],
                                        use_cosine_similarity = config["use_cosine_similarity"]
                                        )

        self.n_subsets = config["n_subsets"]

        del config["tau"],
        del config["use_contrastive"]
        del config["use_distance"]
        del config["use_cosine_similarity"]
        del config["shuffle"]
        del config["mask_ratio"]
        del config["noise_type"]
        del config["noise_level"]
        
        self._init_model(SubTab, config)
    
    def __get_recon_label(self, label: torch.Tensor) -> torch.Tensor:
        """Duplicates the input label tensor across the batch dimension to match the number of subsets for reconstruction loss.

        Args:
            label (torch.Tensor): The input tensor representing the label to be duplicated.

        Returns:
            torch.Tensor: A tensor with the input `label` duplicated `self.n_subsets` times along the batch dimension.
        """
        recon_label = label
        for _ in range(1, self.n_subsets):
            recon_label = torch.concat((recon_label, label), dim = 0)
        return recon_label
    
    def __arange_subsets(self, projections: torch.Tensor) -> torch.Tensor:
        """
        Rearranges the projections tensor into a sequence of subsets for first phase loss.
        This method takes a tensor of projections and reshapes it into a format where each subset of projections is concatenated along the first dimension. 
        The original tensor, which is assumed to represent a series of projections, is first reshaped into a tensor of shape `(n_subsets, samples, dim)` 
        where `n_subsets` is the number of subsets, `samples` is the batch_size, and `dim` is the embedding dimension. 
        The method then concatenates these subsets along the zeroth dimension to produce a sequence of projections suitable for first phase loss.

        Args:
            projections (torch.Tensor): A tensor of shape `(no, dim)` where `no` is the total number of observations and `dim` is the embedding dimension. 

        Returns:
            torch.Tensor: A reshaped tensor where subsets of projections are concatenated along the first dimension. 
                                The resulting shape is `(no, dim)`, maintaining the original dimensionality but rearranging the order of observations to align with subset divisions.
        """
        no, dim = projections.shape
        samples = int(no / self.n_subsets)
        
        projections = projections.reshape((self.n_subsets, samples, dim))
        return torch.concat([projections[:, i] for i in range(samples)])

    def _get_first_phase_loss(self, batch:Tuple[torch.FloatTensor, torch.Tensor, torch.Tensor]) -> torch.FloatTensor:
        """Calculate the first phase loss

        Args:
            batch (Tuple[torch.FloatTensor, torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.FloatTensor: The final loss of first phase step
        """
        x, y_recons, y = batch
        projections, x_recons, x = self.model(x)
        
        recon_label = self.__get_recon_label(y_recons)
        
        projections = self.__arange_subsets(projections)
        
        total_loss, contrastive_loss, recon_loss, dist_loss = self.first_phase_loss(projections, x_recons, recon_label)

        return total_loss
    
    def _get_second_phase_loss(self, batch:Dict[str, Any]):
        """Calculate the second phase loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of second phase step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        x, _, y = batch
        y_hat = self(x).squeeze()
        
        loss = self.loss_fn(y_hat, y)
        
        return loss, y, y_hat
    
    def predict_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        """The perdict step of SubTab

        Args:
            batch (Dict[str, Any]): The input batch
            batch_idx (int): For compatibility, do not use

        Returns:
            torch.FloatTensor: The predicted output (logit)
        """
        x, _, _ = batch
        y_hat = self(x)
        return y_hat