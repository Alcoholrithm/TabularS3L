from typing import Dict, Any, Tuple, Union

from base_module import TS3LLightining
from models import SubTab
from utils.subtab_utils import JointLoss
from copy import deepcopy
import torch

class SubTabLightning(TS3LLightining):
    
    def __init__(self, *args, **kwargs):
        super(SubTabLightning, self).__init__(*args, **kwargs)

    def _initialize(self, model_hparams: Dict[str, Any]):
        
        hparams = deepcopy(model_hparams)
        del hparams["batch_size"]
        del hparams["tau"],
        del hparams["use_contrastive"]
        del hparams["use_distance"]
        del hparams["use_cosine_similarity"]
        
        self.model = SubTab(**hparams)
        
        self.pretraining_loss = JointLoss(model_hparams["batch_size"],
                                          model_hparams["tau"],
                                          model_hparams["n_subsets"],
                                          model_hparams["use_contrastive"],
                                          model_hparams["use_distance"],
                                          use_cosine_similarity = model_hparams["use_cosine_similarity"])

        self.n_subsets = model_hparams["n_subsets"]
        
    def get_recon_label(self, x: torch.FloatTensor) -> torch.FloatTensor:
        recon_label = x
        for _ in range(1, self.n_subsets):
            recon_label = torch.concat((recon_label, x), dim = 0)
        return recon_label
    
    def arange_subsets(self, projections: torch.FloatTensor) -> torch.FloatTensor:
        no, dim = projections.shape
        samples = int(no / self.n_subsets)
        
        projections = projections.reshape((self.n_subsets, samples, dim))
        return torch.concat([projections[:, i] for i in range(samples)])

    def get_pretraining_loss(self, batch:Tuple[torch.FloatTensor, Union[torch.FloatTensor, torch.LongTensor]]):
        """Calculate the pretraining loss

        Args:
            batch (Tuple[torch.FloatTensor, Union[torch.FloatTensor, torch.LongTensor]]): The input batch

        Returns:
            torch.FloatTensor: The final loss of pretraining step
        """
        x, y_recons, y = batch
        projections, x_recons, x = self.model(x)
        
        recon_label = self.get_recon_label(y_recons)
        
        projections = self.arange_subsets(projections)
        
        total_loss, contrastive_loss, recon_loss, dist_loss = self.pretraining_loss(projections, x_recons, recon_label)

        return total_loss
    
    def get_finetunning_loss(self, batch:Dict[str, Any]):
        """Calculate the finetunning loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of finetunning step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        x, _, y = batch
        y_hat = self(x)
        
        loss = self.loss_fn(y_hat, y)
        
        return loss, y, y_hat