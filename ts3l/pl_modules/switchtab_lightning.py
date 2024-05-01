from typing import Dict, Any, Type, Tuple, Union
import torch
from torch import nn

from .base_module import TS3LLightining
from ts3l.models import SwitchTab
from ts3l.utils.switchtab_utils import SwitchTabConfig

class SwitchTabLightning(TS3LLightining):
    
    def __init__(self, config: SwitchTabConfig) -> None:
        """Initialize the pytorch lightining module of SwitchTab

        Args:
            config (SwitchTabConfig): The configuration of SwitchTabLightning.
        """
        super(SwitchTabLightning, self).__init__(config)
    
    def _initialize(self, config: Dict[str, Any]) -> None:
        """Initializes the model with specific hyperparameters and sets up various components of SwitchTabLightning.

        Args:
            config (Dict[str, Any]): The given hyperparameter set for SwitchTab. 
        """
        
        self.u_label = config["u_label"]
        self.alpha = config["alpha"]
        del config["u_label"]
        del config["alpha"]
        del config["corruption_rate"]
        
        self.first_phase_reconstruction_loss = nn.MSELoss()
        
        self._init_model(SwitchTab, config)
    
    def _get_first_phase_loss(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Calculate the first phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.FloatTensor: The final loss of first phase step
        """
        xs, xcs, ys = batch
        size = len(batch)
        
        xs = torch.concat([torch.concat([xs[:size], xs[:size]]), torch.concat([xs[size:], xs[size:]])])
        
        cls_idx = (ys != self.u_label)
        
        x_hat, y_hat = self.model(xcs)
        
        recon_loss = self.first_phase_reconstruction_loss(x_hat, xs)
        
        cls_loss = 0
        if sum(cls_idx) > 0:
            cls_loss = self.loss_fn(y_hat[cls_idx].squeeze(), ys[cls_idx])
        
        return recon_loss + self.alpha * cls_loss
    
    def _get_second_phase_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the second phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]: The input batch

        Returns:
            torch.FloatTensor: The final loss of second phase step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        x, y = batch

        y_hat = self.model(x).squeeze()

        loss = self.loss_fn(y_hat, y)
        
        return loss, y, y_hat
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """The predict step of SwitchTab

            Args:
                batch (torch.Tensor): The input batch
                batch_idx (int): Only for compatibility, do not use

            Returns:
                Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: The predicted output (logit) or (logit, salient_feature)
            """
            out = self(batch)

            return out
        
    def return_salient_feature(self, flag: bool) -> None:
        """Configures the model to either return or not return salient features based on the provided flag.

        Args:
            flag (bool): A boolean flag that determines the behavior of returning salient features.
                        If True, the model will include salient features in its output; if False, it will not.
        """
        self.model.return_salient_feature(flag)