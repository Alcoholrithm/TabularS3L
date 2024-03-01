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
        """Initialize the pytorch lightining module of SCARF

        Args:
            model_hparams (Dict[str, Any]): The hyperparameters of SCARF. 
                                            It must have following keys: {
                                                                            "input_dim": int, the input dimension of SCARF.
                                                                            "out_dim": int, the output dimension of SCARF.
                                                                            'emb_dim': int, the embedding dimension of encoder of SCARF.
                                                                            'encoder_depth': int, The depth of encoder of SCARF.
                                                                            'head_depth': int, The depth of head of SCARF.
                                                                            'dropout_rate': float, A hyperparameter that is to control dropout layer.
                                                                        }
            optim (torch.optim): The optimizer for training. Defaults to torch.optim.AdamW.
            optim_hparams (Dict[str, Any]): The hyperparameters of the optimizer. Defaults to { "lr" : 0.0001, "weight_decay" : 0.00005 }.
            scheduler (torch.optim.lr_scheduler): The scheduler for training. Defaults to None.
            scheduler_hparams (Dict[str, Any]): The hyperparameters of the scheduler. Defaults to {}.
            loss_fn (nn.Module): The loss function for SCARF. Defaults to nn.CrossEntropyLoss.
            loss_hparams (Dict[str, Any]): The hyperparameters of the loss function. Defaults to {}.
            scorer (BaseScorer): The scorer to measure the performance. Defaults to None.
            random_seed (int, optional): The random seed. Defaults to 0.
        """
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
        
        self.save_hyperparameters()

    def _initialize(self, model_hparams: Dict[str, Any]):
        """Initializes the model with specific hyperparameters and sets up various components of SCARFLightning.

        Args:
            model_hparams (Dict[str, Any]): The given hyperparameter set for SCARF. 
        """
        self.model = SCARF(**model_hparams)
        self.first_phase_loss = NTXentLoss()
    
    def _check_model_hparams(self, model_hparams: Dict[str, Any]) -> None:
        """Checks whether the provided hyperparameter set for SCARF is valid by ensuring all required hyperparameters are present.

        This method verifies the presence of all necessary hyperparameters in the `model_hparams` dictionary. 
        It is designed to ensure that the hyperparameter set provided for the SCARF model includes all required keys. 
        The method raises a KeyError if any required hyperparameter is missing.

        Args:
            model_hparams (Dict[str, Any]): The given hyperparameter set for SCARF. 
            This dictionary must include keys for all necessary hyperparameters, 
            which are 'input_dim', 'out_dim', 'emb_dim', 'encoder_depth', 'head_depth', and 'dropout_rate'.

        Raises:
            KeyError: Raised with a message specifying the missing hyperparameter(s) if any required hyperparameters are missing from `model_hparams`. 
            The message will list all missing hyperparameters, formatted appropriately depending on the number missing.
        """
        requirements = [
            "input_dim", "out_dim", "emb_dim", "encoder_depth", "head_depth", "dropout_rate"
        ]
        
        missings = []
        for requirement in requirements:
            if not requirement in model_hparams.keys():
                missings.append(requirement)
        
        if len(missings) == 1:
            raise KeyError("model_hparams requires {%s}" % missings[0])
        elif len(missings) > 1:
            raise KeyError("model_hparams requires {%s, and %s}" % (', '.join(missings[:-1]), missings[-1]))
    
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