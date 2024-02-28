from typing import Dict, Any, Type
import torch
from torch import nn
from ts3l.utils import BaseScorer

from .base_module import TS3LLightining
from ts3l.models import SCARF
from ts3l.utils.scarf_utils import NTXentLoss

class SCARFLightning(TS3LLightining):
    
    def __init__(self,         
                 model_hparams: Dict[str, Any],
                 optim: torch.optim = torch.optim.AdamW,
                 optim_hparams: Dict[str, Any] = {
                                                    "lr" : 0.0001,
                                                    "weight_decay" : 0.00005
                                                },
                 scheduler: torch.optim.lr_scheduler = None,
                 scheduler_hparams: Dict[str, Any] = {},
                 loss_fn: nn.Module = nn.CrossEntropyLoss,
                 loss_hparams: Dict[str, Any] = {},
                 scorer: Type[BaseScorer] = None,
                 random_seed: int = 0
                ):
        super(SCARFLightning, self).__init__(
            model_hparams,
            optim,
            optim_hparams,
            scheduler,
            scheduler_hparams,
            loss_fn,
            loss_hparams,
            scorer,
            random_seed
        )

    def _initialize(self, model_hparams: Dict[str, Any]):
        self.model = SCARF(**model_hparams)
        self.pretraining_loss = NTXentLoss()
    
    def _check_model_hparams(self, model_hparams: Dict[str, Any]):
        pass
    
    def _get_first_phase_loss(self, batch):
        
        x, x_corrupted = batch
        emb_anchor, emb_corrupted = self.model(x, x_corrupted)

        loss = self.pretraining_loss(emb_anchor, emb_corrupted)

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
        y_hat = self(x)

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