from typing import Dict, Any

from base_module import TS3LLightining
from models import SCARF
from utils.scarf_utils import NTXentLoss

class SCARFLightning(TS3LLightining):
    
    def __init__(self, *args, **kwargs):
        super(SCARFLightning, self).__init__(*args, **kwargs)

    def _initialize(self, model_hparams: Dict[str, Any]):
        self.model = SCARF(**model_hparams)
        self.pretraining_loss = NTXentLoss()
    
    def get_pretraining_loss(self, batch):
        
        x, y = batch
        emb_anchor, emb_corrupted = self.model(x)

        loss = self.pretraining_loss(emb_anchor, emb_corrupted)

        return loss
    
    def get_finetunning_loss(self, batch:Dict[str, Any]):
        """Calculate the finetunning loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of finetunning step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        
        return loss, y, y_hat