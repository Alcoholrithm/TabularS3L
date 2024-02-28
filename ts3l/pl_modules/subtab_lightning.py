from typing import Dict, Any, Tuple, Union, Type

from .base_module import TS3LLightining
from ts3l.models import SubTab
from ts3l.utils.subtab_utils import JointLoss
from copy import deepcopy
import torch
from torch import nn
from ts3l.utils import BaseScorer

class SubTabLightning(TS3LLightining):
    
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
        super(SubTabLightning, self).__init__(
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
        
        hparams = deepcopy(model_hparams)
        del hparams["batch_size"]
        del hparams["tau"],
        del hparams["use_contrastive"]
        del hparams["use_distance"]
        del hparams["use_cosine_similarity"]
        
        self.model = SubTab(**hparams)
        
        self.second_phase_loss = JointLoss(model_hparams["batch_size"],
                                          model_hparams["tau"],
                                          model_hparams["n_subsets"],
                                          model_hparams["use_contrastive"],
                                          model_hparams["use_distance"],
                                          use_cosine_similarity = model_hparams["use_cosine_similarity"])

        self.n_subsets = model_hparams["n_subsets"]
    
    def _check_model_hparams(self, model_hparams: Dict[str, Any]):
        pass
    
    def __get_recon_label(self, x: torch.FloatTensor) -> torch.FloatTensor:
        recon_label = x
        for _ in range(1, self.n_subsets):
            recon_label = torch.concat((recon_label, x), dim = 0)
        return recon_label
    
    def __arange_subsets(self, projections: torch.FloatTensor) -> torch.FloatTensor:
        no, dim = projections.shape
        samples = int(no / self.n_subsets)
        
        projections = projections.reshape((self.n_subsets, samples, dim))
        return torch.concat([projections[:, i] for i in range(samples)])

    def _get_first_phase_loss(self, batch:Tuple[torch.FloatTensor, Union[torch.FloatTensor, torch.LongTensor]]):
        """Calculate the first phase loss

        Args:
            batch (Tuple[torch.FloatTensor, Union[torch.FloatTensor, torch.LongTensor]]): The input batch

        Returns:
            torch.FloatTensor: The final loss of first phase step
        """
        x, y_recons, y = batch
        projections, x_recons, x = self.model(x)
        
        recon_label = self.__get_recon_label(y_recons)
        
        projections = self.__arange_subsets(projections)
        
        total_loss, contrastive_loss, recon_loss, dist_loss = self.second_phase_loss(projections, x_recons, recon_label)

        return total_loss
    
    def _get_second_phase_loss(self, batch:Dict[str, Any]):
        """Calculate the second phase loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of second phase step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        x, _, y = batch
        y_hat = self(x)
        
        loss = self.loss_fn(y_hat, y)
        
        return loss, y, y_hat
    
    def predict_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        x, _, _ = batch
        y_hat = self(x)
        return y_hat