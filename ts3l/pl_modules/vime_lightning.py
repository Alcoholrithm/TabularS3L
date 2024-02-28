from typing import Dict, Any, Type
import torch
from torch import nn
from ts3l.utils import BaseScorer

from .base_module import TS3LLightining
from ts3l.models import VIME
from copy import deepcopy

class VIMELightning(TS3LLightining):
    
    def __init__(self, 
                 model_hparams: Dict[str, Any],
                 optim: torch.optim = torch.optim.AdamW,
                 optim_hparams: Dict[str, Any] = {
                                                    "lr" : 0.0001,
                                                    "weight_decay" : 0.00005
                                                },
                 scheduler: torch.optim.lr_scheduler = None,
                 scheduler_hparams: Dict[str, Any] = {},
                 num_categoricals: int = 0,
                 num_continuous: int = 0,
                 u_label: Any = -1,
                 loss_fn: nn.Module = nn.CrossEntropyLoss,
                 loss_hparams: Dict[str, Any] = {},
                 scorer: Type[BaseScorer] = None,
                 random_seed: int = 0
    ):
        
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
        
        self.first_phase_mask_loss = nn.BCELoss()
        self.first_phase_feature_loss1 = nn.CrossEntropyLoss()
        self.first_phase_feature_loss2 = nn.MSELoss()
        
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
    
    def _check_model_hparams(self, model_hparams: Dict[str, Any]):
        requirements = [
            "alpha1", "alpha2", "beta", "K", "encoder_dim", "predictor_hidden_dim", "predictor_output_dim"
        ]
        
        missings = []
        for requirement in requirements:
            if not requirement in model_hparams.keys():
                missings.append(requirement)
        
        if len(missings) == 1:
            raise KeyError("model_hparams requires {%s}" % missings[0])
        elif len(missings) > 1:
            raise KeyError("model_hparams requires {%s, and %s}" % (', '.join(missings[:-1]), missings[-1]))
    
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
        
        labeled_x = x[y != self.u_label].squeeze()
        labeled_y = y[y != self.u_label].squeeze()

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