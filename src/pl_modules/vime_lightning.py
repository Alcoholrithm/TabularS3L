from typing import Dict, Any, Type
import torch
from torch import nn
from utils import BaseScorer

from .base_module import TS3LLightining
from models import VIME
from copy import deepcopy

class VIMELightning(TS3LLightining):
    
    def __init__(self, 
                 model_hparams: Dict[str, Any],
                 optim: torch.optim,
                 optim_hparams: Dict[str, Any],
                 scheduler: torch.optim.lr_scheduler,
                 scheduler_hparams: Dict[str, Any],
                 num_categoricals: int,
                 num_continuous: int,
                 u_label: Any,
                 loss_fn: nn.Module,
                 loss_hparams: Dict[str, Any],
                 scorer: Type[BaseScorer],
                 random_seed: int = 0):
        
        super(VIMELightning, self).__init__(
                 model_hparams,
                 optim,
                 optim_hparams,
                 scheduler,
                 scheduler_hparams,
                 loss_fn,
                 loss_hparams,
                 scorer,
                 random_seed)
        
        self.num_categoricals, self.num_continuous = num_categoricals, num_continuous
        self.u_label = u_label
        
        self.pretraining_mask_loss = nn.BCELoss()
        self.pretraining_feature_loss1 = nn.CrossEntropyLoss()
        self.pretraining_feature_loss2 = nn.MSELoss()
        
        self.consistency_loss = nn.MSELoss()

    def _initialize(self, model_hparams: Dict[str, Any]):
        hparams = deepcopy(model_hparams)
        
        self.alpha1 = hparams["alpha1"]
        self.alpha2 = hparams["alpha2"]
        del hparams["alpha1"]
        del hparams["alpha2"]
        
        self.beta = hparams["beta"]
        del hparams["beta"]
        
        self.K = hparams["K"]
        self.consistency_len = self.K + 1
        del hparams["K"]
        
        self.model = VIME(**hparams)
        
    def get_pretraining_loss(self, batch:Dict[str, Any]):
        """Calculate the pretraining loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of pretraining step
        """
        mask_output, feature_output = self.model.pretraining_step(batch["input"])
        
        mask_loss = self.pretraining_mask_loss(mask_output, batch["label"][0])
        feature_loss1, feature_loss2 = 0, 0
        if self.num_categoricals > 0:
            feature_loss1 = self.pretraining_feature_loss1(feature_output[:, :self.num_categoricals], batch["label"][1][:, :self.num_categoricals])
        if self.num_continuous > 0:
            feature_loss2 = self.pretraining_feature_loss2(feature_output[:, self.num_categoricals:], batch["label"][1][:, self.num_categoricals:])
        final_loss = mask_loss + self.alpha1 * feature_loss1 + self.alpha2 * feature_loss2

        return final_loss
    
    def get_finetunning_loss(self, batch:Dict[str, Any]):
        """Calculate the finetunning loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of finetunning step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        x = batch["input"]
        y = batch["label"]
        
        unsupervised_loss = 0
        unlabeled = x[y == self.u_label]

        if len(unlabeled) > 0:
            u_y_hat = self.model.finetunning_step(unlabeled)
            target = u_y_hat[::self.consistency_len]
            target = target.repeat(1, self.K).reshape((-1, u_y_hat.shape[-1]))
            preds = torch.stack([u_y_hat[i, :] for i in range(len(u_y_hat)) if i % self.consistency_len != 0], dim = 0)
            unsupervised_loss += self.consistency_loss(preds, target)
        
        labeled_x = x[y != self.u_label].squeeze()
        labeled_y = y[y != self.u_label].squeeze()

        y_hat = self.model.finetunning_step(labeled_x).squeeze()

        supervised_loss = self.loss_fn(y_hat, labeled_y)
        
        loss = supervised_loss + self.beta * unsupervised_loss
        
        return loss, labeled_y, y_hat