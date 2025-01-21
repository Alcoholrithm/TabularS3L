from typing import Dict, Any, Type, Tuple, Union
import torch
from torch import nn

from .base_module import TS3LLightining
from ts3l.models import TabularBinning
from ts3l.utils.tabularbinning import TabularBinningConfig
from ts3l import functional as F
from ts3l.utils import BaseConfig

class TabularBinningLightning(TS3LLightining):
    
    def __init__(self, config: TabularBinningConfig) -> None:
        """Initialize the pytorch lightining module of TabularBinning.

        Args:
            config (TabularBinningConfig): The configuration of TabularBinningLightning.
        """
        super(TabularBinningLightning, self).__init__(config)
    
    def _initialize(self, config: BaseConfig) -> None:
        """Initializes the model with specific hyperparameters and sets up various components of TabularBinningLightning.

        Args:
            config (Dict[str, Any]): The given hyperparameter set for TabularBinning. 
        """
        if not isinstance(config, TabularBinningConfig):
            raise TypeError(f"Expected TabularBinningConfig, got {type(config)}")
        
        self._init_model(TabularBinning, config)
    
    def _get_first_phase_loss(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Calculate the first phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.FloatTensor: The final loss of first phase step
        """
        
        return batch[0]
    
    def _get_second_phase_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the second phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]: The input batch

        Returns:
            torch.FloatTensor: The final loss of second phase step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        
        return batch[0], batch[0], batch[0]
    
    def set_second_phase(self, freeze_encoder: bool = False) -> None:
        """Set the module to fine-tuning
        
        Args:
            freeze_encoder (bool): If True, the encoder will be frozen during fine-tuning. Otherwise, the encoder will be trainable.
                                    Default is False.
        """
        return super().set_second_phase(freeze_encoder)
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
        ) -> torch.Tensor:
        """The predict step of TabularBinning

        Args:
            batch (torch.Tensor): The input batch
            batch_idx (int): Only for compatibility, do not use

        Returns:
            torch.Tensor: The predicted output (logit).
        """

        return batch[0]