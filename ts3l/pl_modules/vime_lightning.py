from typing import Dict, Any, Type
import torch
from torch import nn

from .base_module import TS3LLightining
from ts3l.models import VIME
from ts3l.utils.vime_utils import VIMEConfig

class VIMELightning(TS3LLightining):
    
    def __init__(self, config: VIMEConfig) -> None:
        """Initialize the pytorch lightining module of VIME

        Args:
            config (VIMEConfig): The configuration of VIMELightning.
        """
        super(VIMELightning, self).__init__(config)
    
    def _initialize(self, config: Dict[str, Any]) -> None:
        """Initializes the model with specific hyperparameters and sets up various components of VIMELightning.

        Args:
            config (Dict[str, Any]): The given hyperparameter set for VIME. 
        """
        
        self.alpha1 = config["alpha1"]
        self.alpha2 = config["alpha2"]
        del config["alpha1"]
        del config["alpha2"]
        
        self.beta = config["beta"]
        del config["beta"]
        
        self.K = config["K"]
        self.consistency_len = self.K + 1
        del config["K"]
        
        self.num_categoricals, self.num_continuous = config["num_categoricals"], config["num_continuous"]
        del config["num_categoricals"]
        del config["num_continuous"]
        
        self.u_label = config["u_label"]
        del config["u_label"]
        
        del config["p_m"]
        
        self.first_phase_mask_loss = nn.BCELoss()
        self.first_phase_feature_loss1 = nn.CrossEntropyLoss()
        self.first_phase_feature_loss2 = nn.MSELoss()
        
        self.consistency_loss = nn.MSELoss()

        self._init_model(VIME, config)
    
    def _get_first_phase_loss(self, batch:Dict[str, Any]):
        """Calculate the first phase loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of first phase step
        """
        mask_output, feature_output = self.model(batch["input"])
        
        mask_loss = self.first_phase_mask_loss(mask_output, batch["label"][0])
        feature_loss1, feature_loss2 = 0, 0
        if self.num_categoricals > 0:
            feature_loss1 = self.first_phase_feature_loss1(feature_output[:, :self.num_categoricals], batch["label"][1][:, :self.num_categoricals])
        if self.num_continuous > 0:
            feature_loss2 = self.first_phase_feature_loss2(feature_output[:, self.num_categoricals:], batch["label"][1][:, self.num_categoricals:])
        final_loss = mask_loss + self.alpha1 * feature_loss1 + self.alpha2 * feature_loss2

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
        x = batch["input"]
        y = batch["label"]
        
        unsupervised_loss = 0
        unlabeled = x[y == self.u_label]
        
        if len(unlabeled) > 0:
            u_y_hat = self.model(unlabeled)
            target = u_y_hat[::self.consistency_len]
            target = target.repeat(1, self.K).reshape((-1, u_y_hat.shape[-1]))
            preds = torch.stack([u_y_hat[i, :] for i in range(len(u_y_hat)) if i % self.consistency_len != 0], dim = 0)
            unsupervised_loss += self.consistency_loss(preds, target)
        
        labeled_x = x[y != self.u_label]
        labeled_y = y[y != self.u_label]

        y_hat = self.model(labeled_x).squeeze()

        supervised_loss = self.loss_fn(y_hat, labeled_y)
        loss = supervised_loss + self.beta * unsupervised_loss
        
        return loss, labeled_y, y_hat
    
    def predict_step(self, batch, batch_idx: int
        ) -> torch.FloatTensor:
            """The predict step of VIME

            Args:
                batch (Dict[str, Any]): The input batch
                batch_idx (int): Only for compatibility, do not use

            Returns:
                torch.FloatTensor: The predicted output (logit)
            """
            y_hat = self(batch["input"])

            return y_hat