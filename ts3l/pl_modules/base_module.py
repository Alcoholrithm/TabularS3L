from typing import Dict, Any, Type
from ts3l.utils import BaseScorer

from abc import ABC, abstractmethod

import torch
from torch import nn

import pytorch_lightning as pl

class TS3LLightining(ABC, pl.LightningModule):
    """The pytorch lightning module of TabularS3L
    """
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
    ) -> None:
        """Initialize the pytorch lightining module of TabularS3L

        Args:
            model_hparams (Dict[str, Any]): The hyperparameters of TabularS3L.
            optim (torch.optim): The optimizer for training. Defaults to torch.optim.AdamW.
            optim_hparams (Dict[str, Any]): The hyperparameters of the optimizer. Defaults to { "lr" : 0.0001, "weight_decay" : 0.00005 }.
            scheduler (torch.optim.lr_scheduler): The scheduler for training. Defaults to None.
            scheduler_hparams (Dict[str, Any]): The hyperparameters of the scheduler. Defaults to {}.
            loss_fn (nn.Module): The loss function for VIME. Defaults to nn.CrossEntropyLoss.
            loss_hparams (Dict[str, Any]): The hyperparameters of the loss function. Defaults to {}.
            scorer (BaseScorer): The scorer to measure the performance. Defaults to None.
            random_seed (int, optional): The random seed. Defaults to 0.
        """
        super(TS3LLightining, self).__init__()
        
        
        pl.seed_everything(random_seed)
        
        self._initialize(model_hparams)
        
        self.optim = getattr(torch.optim, optim)
        self.optim_hparams = optim_hparams

        self.sched = getattr(torch.optim.lr_scheduler, scheduler) if scheduler is not None else None
        self.scheduler_hparams = scheduler_hparams
        
        self.loss_fn = loss_fn(**loss_hparams)
        
        self.scorer = scorer
            
        self.set_first_phase()

        self.first_phase_step_outputs = []
        self.second_phase_step_outputs = []
        
        self.save_hyperparameters()
    
    @abstractmethod
    def _initialize(self, model_hparams: Dict[str, Any]):
        pass

    @abstractmethod
    def _check_model_hparams(self, model_hparams: Dict[str, Any]):
        pass
    
    def configure_optimizers(self):
        """Configure the optimizer
        """
        self.optimizer = self.optim(self.parameters(), **self.optim_hparams)
        if self.sched is None:
            return [self.optimizer]
        self.scheduler = self.sched(self.optimizer, **self.scheduler_hparams)
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'} ]

    def set_first_phase(self) -> None:
        """Set the module to pretraining
        """
        self.model.set_first_phase()
        self.training_step = self._first_phase_step
        self.on_validation_start = self._on_first_phase_validation_start
        self.validation_step = self._first_phase_step
        self.on_validation_epoch_end = self._first_phase_validation_epoch_end

    def set_second_phase(self) -> None:
        """Set the module to finetunning
        """
        self.model.set_second_phase()
        self.training_step = self._second_phase_step
        self.on_validation_start = self._on_second_phase_validation_start
        self.validation_step = self._second_phase_step
        self.on_validation_epoch_end = self._second_phase_validation_epoch_end

    def forward(self,
                batch:Dict[str, Any]
    ) -> torch.FloatTensor:
        """Do forward pass for given input

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The output of forward pass
        """
        return self.model(batch)
    

    @abstractmethod
    def _get_first_phase_loss(self, batch:Dict[str, Any]):
        """Calculate the first phase loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of first phase step
        """
        pass
    
    def _first_phase_step(self,
                      batch,
                      batch_idx: int
    ) -> Dict[str, Any]:
        """The first phase step of TabularS3L

        Args:
            batch (Dict[str, Any]): The input batch
            batch_idx (int): Only for compatibility

        Returns:
            Dict[str, Any]: The loss of the first phase step
        """

        loss = self._get_first_phase_loss(batch)
        self.first_phase_step_outputs.append({
            "loss" : loss
        })
        return {
            "loss" : loss
        }

    def _on_first_phase_validation_start(self):
        """Log the training loss of the first_phase
        """
        if len(self.first_phase_step_outputs) > 0:
            train_loss = torch.Tensor([out["loss"] for out in self.first_phase_step_outputs]).cpu().mean()
            
            self.log("train_loss", train_loss, prog_bar = True)
            
            self.first_phase_step_outputs = []    
        return super().on_validation_start() 
    
    def _first_phase_validation_epoch_end(self) -> None:
        """Log the validation loss of the first phase
        """
        val_loss = torch.Tensor([out["loss"] for out in self.first_phase_step_outputs]).cpu().mean()

        self.log("val_loss", val_loss, prog_bar = True)
        self.first_phase_step_outputs = []
        return super().on_validation_epoch_end()

    @abstractmethod
    def _get_second_phase_loss(self, batch:Dict[str, Any]):
        """Calculate the second phase loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of second phase step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        pass
        
    
    def _second_phase_step(self,
                      batch,
                      batch_idx: int = 0
    ) -> Dict[str, Any]:
        """The second phase step of TabularS3L

        Args:
            batch (Dict[str, Any]): The input batch
            batch_idx (int): Only for compatibility

        Returns:
            Dict[str, Any]: The loss of the second phase step
        """
        loss, y, y_hat = self._get_second_phase_loss(batch)

        self.second_phase_step_outputs.append(
            {
            "loss" : loss,
            "y" : y,
            "y_hat" : y_hat
        }
        )
        return {
            "loss" : loss
        }
    
    def _on_second_phase_validation_start(self):
        """Log the training loss and the performance of the second phase
        """
        if len(self.second_phase_step_outputs) > 0:
            train_loss = torch.Tensor([out["loss"] for out in self.second_phase_step_outputs]).cpu().mean()
            y = torch.cat([out["y"] for out in self.second_phase_step_outputs]).cpu().detach().numpy()
            y_hat = torch.cat([out["y_hat"] for out in self.second_phase_step_outputs]).cpu().detach().numpy()
            
            train_score = self.scorer(y, y_hat)
            
            self.log("train_loss", train_loss, prog_bar = True)
            self.log("train_" + self.scorer.__name__, train_score, prog_bar = True)
            self.second_phase_step_outputs = []   
            
        return super().on_validation_start()
    
    def _second_phase_validation_epoch_end(self) -> None:
        """Log the validation loss and the performance of the second phase
        """
        val_loss = torch.Tensor([out["loss"] for out in self.second_phase_step_outputs]).cpu().mean()

        y = torch.cat([out["y"] for out in self.second_phase_step_outputs]).cpu().numpy()
        y_hat = torch.cat([out["y_hat"] for out in self.second_phase_step_outputs]).cpu().numpy()
        val_score = self.scorer(y, y_hat)

        self.log("val_" + self.scorer.__name__, val_score, prog_bar = True)
        self.log("val_loss", val_loss, prog_bar = True)
        self.second_phase_step_outputs = []      
        return super().on_validation_epoch_end()
    
    @abstractmethod
    def predict_step(self, batch, batch_idx: int
    ) -> torch.FloatTensor:
        """The perdict step

        Args:
            batch (Dict[str, Any]): The input batch
            batch_idx (int): Only for compatibility

        Returns:
            torch.FloatTensor: The predicted output (logit)
        """
        pass