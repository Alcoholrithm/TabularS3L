from typing import Dict, Any, Type, Tuple, Union
import torch
from torch import nn

from .base_module import TS3LLightining
from ts3l.models import SwitchTab
from ts3l.utils.switchtab_utils import SwitchTabConfig
from ts3l import functional as F
from ts3l.utils import BaseConfig

class SwitchTabLightning(TS3LLightining):
    
    def __init__(self, config: SwitchTabConfig) -> None:
        """Initialize the pytorch lightining module of SwitchTab

        Args:
            config (SwitchTabConfig): The configuration of SwitchTabLightning.
        """
        super(SwitchTabLightning, self).__init__(config)
    
    def _initialize(self, config: BaseConfig) -> None:
        """Initializes the model with specific hyperparameters and sets up various components of SwitchTabLightning.

        Args:
            config (Dict[str, Any]): The given hyperparameter set for SwitchTab. 
        """
        if not isinstance(config, SwitchTabConfig):
            raise TypeError(f"Expected SwitchTabConfig, got {type(config)}")
        
        self.u_label = config.u_label
        self.alpha = config.alpha
        
        self.reconstruction_loss_fn = nn.MSELoss()
        
        self._init_model(SwitchTab, config)
    
    def _get_first_phase_loss(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Calculate the first phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.FloatTensor: The final loss of first phase step
        """
        
        x_hat, labeled_y, labeled_y_hat = F.switchtab.first_phase_step(self.model, batch, self.u_label)

        xs, _, _ = batch
        
        size = len(batch)
        
        xs = torch.concat(
        [torch.concat([xs[:size], xs[:size]]), torch.concat([xs[size:], xs[size:]])]
    )
        
        recon_loss, task_loss = F.switchtab.first_phase_loss(xs, x_hat, labeled_y, labeled_y_hat, self.reconstruction_loss_fn, self.task_loss_fn)
    
        return recon_loss + self.alpha * task_loss
    
    def _get_second_phase_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the second phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]: The input batch

        Returns:
            torch.FloatTensor: The final loss of second phase step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        _, y = batch
        y_hat = F.switchtab.second_phase_step(self.model, batch)
        task_loss = F.switchtab.second_phase_loss(y, y_hat, self.task_loss_fn)
        
        return task_loss, y, y_hat
    
    def set_second_phase(self, freeze_encoder: bool = False) -> None:
        """Set the module to fine-tuning
        
        Args:
            freeze_encoder (bool): If True, the encoder will be frozen during fine-tuning. Otherwise, the encoder will be trainable.
                                    Default is False.
        """
        return super().set_second_phase(freeze_encoder)
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """The predict step of SwitchTab

        Args:
            batch (torch.Tensor): The input batch
            batch_idx (int): Only for compatibility, do not use

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: The predicted output (logit) or (logit, salient_feature)
        """
        y_hat = F.switchtab.second_phase_step(self.model, batch)

        return y_hat
        
    def return_salient_feature(self, flag: bool) -> None:
        """Configures the model to either return or not return salient features based on the provided flag.

        Args:
            flag (bool): A boolean flag that determines the behavior of returning salient features.
                        If True, the model will include salient features in its output; if False, it will not.
        """
        self.model.return_salient_feature(flag)