from typing import Dict, Any, Type, Tuple
import torch
from torch import nn

from .base_module import TS3LLightining
from ts3l.models import DAE
from ts3l.utils.dae_utils import DAEConfig
from ts3l import functional as F
from ts3l.utils import BaseConfig

class DAELightning(TS3LLightining):

    def __init__(self, config: DAEConfig) -> None:
        """Initialize the pytorch lightining module of DAE

        Args:
            config (DAEConfig): The configuration of DAELightning.
        """
        super(DAELightning, self).__init__(config)

    def _initialize(self, config: BaseConfig) -> None:
        """Initializes the model with specific hyperparameters and sets up various components of DAELightning.

        Args:
            config (Dict[str, Any]): The given hyperparameter set for DAE.
        """
        if not isinstance(config, DAEConfig):
            raise TypeError(f"Expected DAEConfig, got {type(config)}")
        
        self.mask_loss_weight = config.mask_loss_weight

        self.num_categoricals, self.num_continuous = (
            len(config.cat_cardinality),
            config.num_continuous,
        )

        self.mask_loss_fn = nn.BCELoss()
        self.categorical_feature_loss = nn.CrossEntropyLoss()
        self.continuous_feature_loss = nn.MSELoss()

        self._init_model(DAE, config)

    def _get_first_phase_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Calculate the first phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.FloatTensor: The final loss of first phase step
        """
        x, _, mask = batch
        
        mask_preds, cat_preds, cont_preds = F.dae.first_phase_step(self.model, batch)
        
        mask_loss, feature_loss = F.dae.first_phase_loss(
            x[:, : self.num_categoricals],
            x[:, self.num_categoricals :],
            mask,
            cat_preds, 
            cont_preds,
            mask_preds,
            self.mask_loss_fn,
            self.categorical_feature_loss,
            self.continuous_feature_loss,
        )
        
        return mask_loss * self.mask_loss_weight + feature_loss

    def _get_second_phase_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        """Calculate the second phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.FloatTensor: The final loss of second phase step
            torch.Tensor: The ground truth label
            torch.Tensor: The predicted label
        """
        _, y = batch

        y_hat = F.dae.second_phase_step(self.model, batch)

        loss = F.dae.second_phase_loss(y, y_hat, self.task_loss_fn)

        return loss, y, y_hat

    def set_second_phase(self, freeze_encoder: bool = True) -> None:
        """Set the module to fine-tuning
        
        Args:
            freeze_encoder (bool): If True, the encoder will be frozen during fine-tuning. Otherwise, the encoder will be trainable.
                                    Default is True.
        """
        return super().set_second_phase(freeze_encoder)
    
    def predict_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        """The predict step of DAE

        Args:
            batch (torch.Tensor): The input batch
            batch_idx (int): Only for compatibility, do not use

        Returns:
            torch.FloatTensor: The predicted output (logit)
        """
        y_hat = self(batch)

        return y_hat
