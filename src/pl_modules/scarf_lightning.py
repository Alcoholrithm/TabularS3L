from typing import Dict, Any

from base_module import TS3LLightining
from models import SCARF
class SCARFLightning(TS3LLightining):
    
    def __init__(self, *args, **kwargs):
        super(SCARFLightning, self).__init__(*args, **kwargs)

    def _initialize(self, model_hparams: Dict[str, Any]):
        self.model = SCARF(**model_hparams)
    
    def get_pretraining_loss(self, batch:Dict[str, Any]):
        """Calculate the pretraining loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of pretraining step
        """
        pass
    
    def get_finetunning_loss(self, batch:Dict[str, Any]):
        """Calculate the finetunning loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of finetunning step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        pass