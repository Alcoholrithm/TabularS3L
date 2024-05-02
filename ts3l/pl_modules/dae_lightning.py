from typing import Dict, Any, Type, Tuple
import torch
from torch import nn

from .base_module import TS3LLightining
from ts3l.models import DAE
from ts3l.utils.dae_utils import DAEConfig

class DAELightning(TS3LLightining):
    
    def __init__(self, config: DAEConfig) -> None:
        """Initialize the pytorch lightining module of DAE

        Args:
            config (DAEConfig): The configuration of DAELightning.
        """
        super(DAELightning, self).__init__(config)
    
    def _initialize(self, config: Dict[str, Any]) -> None:
        """Initializes the model with specific hyperparameters and sets up various components of DAELightning.

        Args:
            config (Dict[str, Any]): The given hyperparameter set for DAE. 
        """
        
        self.mask_loss_weight = config["mask_loss_weight"]
        del config["mask_loss_weight"]
        
        self.num_categoricals, self.num_continuous = config["num_categoricals"], config["num_continuous"]
        del config["num_categoricals"]
        del config["num_continuous"]
        
        del config["noise_type"]
        del config["noise_level"]
        del config["noise_ratio"]
        
        self.first_phase_mask_loss = nn.BCELoss()
        self.first_phase_feature_loss1 = nn.CrossEntropyLoss()
        self.first_phase_feature_loss2 = nn.MSELoss()

        self._init_model(DAE, config)
    
    def _get_first_phase_loss(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Calculate the first phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.FloatTensor: The final loss of first phase step
        """
        x, x_bar, mask = batch
        mask_output, feature_output = self.model(x_bar)
        
        # print(mask, mask_output)
        mask_loss = self.first_phase_mask_loss(mask_output, mask)
        feature_loss1, feature_loss2 = 0, 0
        if self.num_categoricals > 0:
            feature_loss1 = self.first_phase_feature_loss1(feature_output[:, :self.num_categoricals], x[:, :self.num_categoricals])
        if self.num_continuous > 0:
            feature_loss2 = self.first_phase_feature_loss2(feature_output[:, self.num_categoricals:], x[:, self.num_categoricals:])
        final_loss = mask_loss * self.mask_loss_weight + feature_loss1 + feature_loss2

        return final_loss
    
    def _get_second_phase_loss(self, batch:Dict[str, Any]):
        """Calculate the second phase loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of second phase step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        x, y = batch

        y_hat = self.model(x).squeeze()

        loss = self.loss_fn(y_hat, y)
        
        return loss, y, y_hat
    
    def predict_step(self, batch, batch_idx: int
        ) -> torch.FloatTensor:
            """The predict step of DAE

            Args:
                batch (torch.Tensor): The input batch
                batch_idx (int): Only for compatibility, do not use

            Returns:
                torch.FloatTensor: The predicted output (logit)
            """
            y_hat = self(batch)

            return y_hat