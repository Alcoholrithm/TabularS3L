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
        """Initialize the pytorch lightining module of SubTab

        Args:
            model_hparams (Dict[str, Any]): The hyperparameters of SubTab. 
                                            It must have following keys: {
                                                                            "input_dim": int, the input dimension of SubTab.
                                                                            "out_dim": int, the output dimension of SubTab.
                                                                            'emb_dim': int, the embedding dimension of encoder of SubTab.
                                                                            'tau': float, A hyperparameter that is to scale similarity between 
                                                                                            projections during the first phase.
                                                                            'use_cosine_similarity': bool, A hyperparameter that is to select whether using cosine similarity 
                                                                                                            or dot similarity when calculating similarity between projections 
                                                                                                            during the first phase.
                                                                            'use_contrastive': bool, A hyperparameter that is to select using contrastive loss or not during 
                                                                                                    the first phase.
                                                                            "use_distance": bool, A hyperparameter that is to select using distance loss or not during
                                                                                                    the first phase.
                                                                            "n_subsets": int, The number of subsets to generate different views of the data. 
                                                                            "overlap_ratio": float, A hyperparameter that is to control the extent of overlapping between 
                                                                                                    the subsets.
                                                                        }
            optim (torch.optim): The optimizer for training. Defaults to torch.optim.AdamW.
            optim_hparams (Dict[str, Any]): The hyperparameters of the optimizer. Defaults to { "lr" : 0.0001, "weight_decay" : 0.00005 }.
            scheduler (torch.optim.lr_scheduler): The scheduler for training. Defaults to None.
            scheduler_hparams (Dict[str, Any]): The hyperparameters of the scheduler. Defaults to {}.
            loss_fn (nn.Module): The loss function for SubTab. Defaults to nn.CrossEntropyLoss.
            loss_hparams (Dict[str, Any]): The hyperparameters of the loss function. Defaults to {}.
            scorer (BaseScorer): The scorer to measure the performance. Defaults to None.
            random_seed (int, optional): The random seed. Defaults to 0.
        """
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
        
        self.save_hyperparameters()

    def _initialize(self, model_hparams: Dict[str, Any]):
        """Initializes the model with specific hyperparameters and sets up various components of SubTabLightning.

        Args:
            model_hparams (Dict[str, Any]): The given hyperparameter set for SubTab. 
        """
        hparams = deepcopy(model_hparams)
        del hparams["tau"],
        del hparams["use_contrastive"]
        del hparams["use_distance"]
        del hparams["use_cosine_similarity"]
        
        self.model = SubTab(**hparams)
        
        self.first_phase_loss = JointLoss(
                                          model_hparams["tau"],
                                          model_hparams["n_subsets"],
                                          model_hparams["use_contrastive"],
                                          model_hparams["use_distance"],
                                          use_cosine_similarity = model_hparams["use_cosine_similarity"])

        self.n_subsets = model_hparams["n_subsets"]
    
    def _check_model_hparams(self, model_hparams: Dict[str, Any]) -> None:
        """Checks whether the provided hyperparameter set for SubTab is valid by ensuring all required hyperparameters are present.

        This method verifies the presence of all necessary hyperparameters in the `model_hparams` dictionary. 
        It is designed to ensure that the hyperparameter set provided for the SubTab model includes all required keys. 
        The method raises a KeyError if any required hyperparameter is missing.

        Args:
            model_hparams (Dict[str, Any]): The given hyperparameter set for SubTab. 
            This dictionary must include keys for all necessary hyperparameters, 
            which are 'input_dim', 'out_dim', 'emb_dim', 'tau', 'use_cosine_similarity', 'use_contrastive', 
                        'use_distance', 'n_subsets', and 'overlap_ratio'.

        Raises:
            KeyError: Raised with a message specifying the missing hyperparameter(s) if any required hyperparameters are missing from `model_hparams`. 
            The message will list all missing hyperparameters, formatted appropriately depending on the number missing.
        """
        requirements = [
            "input_dim", "out_dim", "emb_dim", "tau", "use_cosine_similarity", "use_contrastive", "use_distance", "n_subsets", "overlap_ratio"
        ]
        
        missings = []
        for requirement in requirements:
            if not requirement in model_hparams.keys():
                missings.append(requirement)
        
        if len(missings) == 1:
            raise KeyError("model_hparams requires {%s}" % missings[0])
        elif len(missings) > 1:
            raise KeyError("model_hparams requires {%s, and %s}" % (', '.join(missings[:-1]), missings[-1]))
    
    def __get_recon_label(self, label: torch.FloatTensor) -> torch.FloatTensor:
        """Duplicates the input label tensor across the batch dimension to match the number of subsets for reconstruction loss.

        Args:
            label (torch.FloatTensor): The input tensor representing the label to be duplicated.

        Returns:
            torch.FloatTensor: A tensor with the input `label` duplicated `self.n_subsets` times along the batch dimension.
        """
        recon_label = label
        for _ in range(1, self.n_subsets):
            recon_label = torch.concat((recon_label, label), dim = 0)
        return recon_label
    
    def __arange_subsets(self, projections: torch.FloatTensor) -> torch.FloatTensor:
        """
        Rearranges the projections tensor into a sequence of subsets for first phase loss.
        This method takes a tensor of projections and reshapes it into a format where each subset of projections is concatenated along the first dimension. 
        The original tensor, which is assumed to represent a series of projections, is first reshaped into a tensor of shape `(n_subsets, samples, dim)` 
        where `n_subsets` is the number of subsets, `samples` is the batch_size, and `dim` is the embedding dimension. 
        The method then concatenates these subsets along the zeroth dimension to produce a sequence of projections suitable for first phase loss.

        Args:
            projections (torch.FloatTensor): A tensor of shape `(no, dim)` where `no` is the total number of observations and `dim` is the embedding dimension. 

        Returns:
            torch.FloatTensor: A reshaped tensor where subsets of projections are concatenated along the first dimension. 
                                The resulting shape is `(no, dim)`, maintaining the original dimensionality but rearranging the order of observations to align with subset divisions.
        """
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
        
        total_loss, contrastive_loss, recon_loss, dist_loss = self.first_phase_loss(projections, x_recons, recon_label)

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
        y_hat = self(x).squeeze()
        
        loss = self.loss_fn(y_hat, y)
        
        return loss, y, y_hat
    
    def predict_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        """The perdict step of SubTab

        Args:
            batch (Dict[str, Any]): The input batch
            batch_idx (int): For compatibility, do not use

        Returns:
            torch.FloatTensor: The predicted output (logit)
        """
        x, _, _ = batch
        y_hat = self(x)
        return y_hat