from typing import Dict, Any, Tuple, Union, Type

from .base_module import TS3LLightining
from ts3l.models import SubTab
from ts3l.models.subtab import JointLoss

import torch
from ts3l.utils.subtab_utils import SubTabConfig
from ts3l import functional as F
from ts3l.utils import BaseConfig
class SubTabLightning(TS3LLightining):
    
    def __init__(self, config: SubTabConfig) -> None:
        """Initialize the pytorch lightining module of SubTab

        Args:
            config (SubTabConfig): The configuration of SubTabLightning.
        """
        super(SubTabLightning, self).__init__(config)

    def _initialize(self, config: BaseConfig):
        """Initializes the model with specific hyperparameters and sets up various components of SubTabLightning.

        Args:
            config (Dict[str, Any]): The given hyperparameter set for SubTab. 
        """
        if not isinstance(config, SubTabConfig):
            raise TypeError(f"Expected SubTabConfig, got {type(config)}")
        
        self.joint_loss_fn = JointLoss(
                                        config.tau,
                                        config.n_subsets,
                                        config.use_contrastive,
                                        config.use_distance,
                                        use_cosine_similarity = config.use_cosine_similarity
                                        )

        self.n_subsets = config.n_subsets
        
        self._init_model(SubTab, config)

    def _get_first_phase_loss(self, batch:Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Calculate the first phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.Tensor: The final loss of first phase step
        """
        projections, x_recons = F.subtab.first_phase_step(self.model, batch)
        
        x_originals, _ = batch
        
        total_loss, contrastive_loss, recon_loss, dist_loss = F.subtab.first_phase_loss(projections, x_recons, x_originals, self.n_subsets, self.joint_loss_fn)
        
        return total_loss
    
    def _get_second_phase_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        """Calculate the second phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.FloatTensor: The final loss of second phase step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        y_hat = F.subtab.second_phase_step(self.model, batch)
        
        _, y = batch
        
        task_loss = F.subtab.second_phase_loss(y, y_hat, self.task_loss_fn)
        
        return task_loss, y, y_hat
    
    def set_second_phase(self, freeze_encoder: bool = True) -> None:
        """Set the module to fine-tuning
        
        Args:
            freeze_encoder (bool): If True, the encoder will be frozen during fine-tuning. Otherwise, the encoder will be trainable.
                                    Default is True.
        """
        return super().set_second_phase(freeze_encoder)
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """The perdict step of SubTab

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The input batch
            batch_idx (int): For compatibility, do not use

        Returns:
            torch.FloatTensor: The predicted output (logit)
        """
        
        return F.subtab.second_phase_step(self.model, batch)