from typing import Dict, Any, Tuple
import torch

from .base_module import TS3LLightining
from ts3l.models import SCARF
from ts3l.models.scarf import NTXentLoss
from ts3l.utils.scarf_utils import SCARFConfig
from ts3l import functional as F
from ts3l.utils import BaseConfig

class SCARFLightning(TS3LLightining):

    def __init__(self, config: SCARFConfig) -> None:
        """Initialize the pytorch lightining module of SCARF

        Args:
            config (SubTabConfig): The configuration of SCARFLightning.
        """
        super(SCARFLightning, self).__init__(config)

    def _initialize(self, config: BaseConfig):
        """Initializes the model with specific hyperparameters and sets up various components of SCARFLightning.

        Args:
            config (Dict[str, Any]): The given hyperparameter set for SCARF.
        """
        if not isinstance(config, SCARFConfig):
            raise TypeError(f"Expected SCARFConfig, got {type(config)}")
        
        self.contrastive_loss = NTXentLoss(config.tau)

        self._init_model(SCARF, config)

    def _get_first_phase_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Calculate the first phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.Tensor: The final loss of first phase step
        """

        emb_anchor, emb_corrupted = F.scarf.first_phase_step(self.model, batch)

        loss = F.scarf.first_phase_loss(
            emb_anchor, emb_corrupted, self.contrastive_loss
        )

        return loss

    def _get_second_phase_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the second phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.Tensor: The final loss of second phase step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        _, y = batch
        y_hat = F.scarf.second_phase_step(self.model, batch)
        task_loss = F.scarf.second_phase_loss(y, y_hat, self.task_loss_fn)

        return task_loss, y, y_hat

    def set_second_phase(self, freeze_encoder: bool = False) -> None:
        """Set the module to fine-tuning
        
        Args:
            freeze_encoder (bool): If True, the encoder will be frozen during fine-tuning. Otherwise, the encoder will be trainable.
                                    Default is False.
        """
        return super().set_second_phase(freeze_encoder)
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """The perdict step of SCARF

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The input batch
            batch_idx (int): For compatibility, do not use

        Returns:
            torch.Tensor: The predicted output (logit)
        """

        y_hat = F.scarf.second_phase_step(self.model, batch)

        return y_hat
