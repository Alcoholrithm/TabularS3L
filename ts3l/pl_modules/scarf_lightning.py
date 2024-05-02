from typing import Dict, Any, Type
import torch

from .base_module import TS3LLightining
from ts3l.models import SCARF
from ts3l.utils.scarf_utils import NTXentLoss, SCARFConfig

class SCARFLightning(TS3LLightining):
    
    def __init__(self, config: SCARFConfig) -> None:
        """Initialize the pytorch lightining module of SCARF

        Args:
            config (SubTabConfig): The configuration of SCARFLightning.
        """
        super(SCARFLightning, self).__init__(config)

    def _initialize(self, config: Dict[str, Any]):
        """Initializes the model with specific hyperparameters and sets up various components of SCARFLightning.

        Args:
            config (Dict[str, Any]): The given hyperparameter set for SCARF. 
        """
        self.first_phase_loss = NTXentLoss(config["tau"])
        del config["tau"]
        del config["corruption_rate"]
        
        self._init_model(SCARF, config)
    
    def _get_first_phase_loss(self, batch):
        """Calculate the first phase loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of first phase step
        """
        x, x_corrupted = batch
        emb_anchor, emb_corrupted = self.model(x, x_corrupted)

        loss = self.first_phase_loss(emb_anchor, emb_corrupted)

        return loss
    
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
        y_hat = self(x).squeeze()

        loss = self.loss_fn(y_hat, y)
        
        return loss, y, y_hat
    
    def predict_step(self, batch, batch_idx: int
    ) -> torch.FloatTensor:
        """The perdict step of SCARF

        Args:
            batch (Dict[str, Any]): The input batch
            batch_idx (int): For compatibility, do not use

        Returns:
            torch.FloatTensor: The predicted output (logit)
        """
        
        y_hat = self(batch)

        return y_hat