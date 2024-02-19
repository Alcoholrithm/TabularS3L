from typing import Dict, Any, Type
from ts3l.utils import BaseScorer

from abc import ABC, abstractmethod

import torch
from torch import nn

import pytorch_lightning as pl

class TS3LLightining(ABC, pl.LightningModule):
    """The pytorch lightning module of VIME
    """
    def __init__(self,
                 model_hparams: Dict[str, Any],
                 optim: torch.optim,
                 optim_hparams: Dict[str, Any],
                 scheduler: torch.optim.lr_scheduler,
                 scheduler_hparams: Dict[str, Any],
                 loss_fn: nn.Module,
                 loss_hparams: Dict[str, Any],
                 scorer: Type[BaseScorer],
                 random_seed: int = 0
    ) -> None:
        """Initialize the pytorch lightining module of VIME

        Args:
            model_hparams (Dict[str, Any]): The hyperparameters of VIME
            optim (torch.optim): The optimizer for training
            optim_hparams (Dict[str, Any]): The hyperparameters of the optimizer
            scheduler (torch.optim.lr_scheduler): The scheduler for training
            scheduler_hparams (Dict[str, Any]): The hyperparameters of the scheduler
            num_categoricals (int): The number of categorical features
            num_continuous (int): The number of continuous features
            u_label (Any): The specifier for unlabeled data.
            loss_fn (nn.Module): The loss function of pytorch
            scorer (BaseScorer): The scorer to measure the performance
            random_seed (int, optional): The random seed. Defaults to 0.
        """
        super(TS3LLightining, self).__init__()
        
        
        pl.seed_everything(random_seed)
        
        self._initialize(model_hparams)
        
        self.optim = getattr(torch.optim, optim)
        self.optim_hparams = optim_hparams

        self.scheduler = getattr(torch.optim.lr_scheduler, scheduler)
        self.scheduler_hparams = scheduler_hparams
        
        self.loss_fn = loss_fn(**loss_hparams)
        
        self.scorer = scorer
            
        self.do_pretraining()

        self.pretraining_step_outputs = []
        self.finetunning_step_outputs = []
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
        if len(self.scheduler_hparams) == 0:
            return [self.optimizer]
        self.scheduler = self.scheduler(self.optimizer, **self.scheduler_hparams)
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'} ]

    def do_pretraining(self) -> None:
        """Set the module to pretraining
        """
        self.model.do_pretraining()
        self.training_step = self.pretraining_step
        self.on_validation_start = self.on_pretraining_validation_start
        self.validation_step = self.pretraining_step
        self.on_validation_epoch_end = self.pretraining_validation_epoch_end

    def do_finetunning(self) -> None:
        """Set the module to finetunning
        """
        self.model.do_finetunning()
        self.training_step = self.finetuning_step
        self.on_validation_start = self.on_finetunning_validation_start
        self.validation_step = self.finetuning_step
        self.on_validation_epoch_end = self.finetuning_validation_epoch_end

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
    def get_pretraining_loss(self, batch:Dict[str, Any]):
        """Calculate the pretraining loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of pretraining step
        """
        pass
    
    def pretraining_step(self,
                      batch,
                      batch_idx: int
    ) -> Dict[str, Any]:
        """Pretraining step of VIME

        Args:
            batch (Dict[str, Any]): The input batch
            batch_idx (int): For compatibility, do not use

        Returns:
            Dict[str, Any]: The loss of the pretraining step
        """

        loss = self.get_pretraining_loss(batch)
        self.pretraining_step_outputs.append({
            "loss" : loss
        })
        return {
            "loss" : loss
        }

    def on_pretraining_validation_start(self):
        """Log the training loss of the pretraining
        """
        if len(self.pretraining_step_outputs) > 0:
            train_loss = torch.Tensor([out["loss"] for out in self.pretraining_step_outputs]).cpu().mean()
            
            self.log("train_loss", train_loss, prog_bar = True)
            
            self.pretraining_step_outputs = []    
        return super().on_validation_start() 
    
    def pretraining_validation_epoch_end(self) -> None:
        """Log the validation loss of the pretraining
        """
        val_loss = torch.Tensor([out["loss"] for out in self.pretraining_step_outputs]).cpu().mean()

        self.log("val_loss", val_loss, prog_bar = True)
        self.pretraining_step_outputs = []
        return super().on_validation_epoch_end()

    @abstractmethod
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
        
    
    def finetuning_step(self,
                      batch,
                      batch_idx: int = 0
    ) -> Dict[str, Any]:
        """Finetunning step of VIME

        Args:
            batch (Dict[str, Any]): The input batch
            batch_idx (int): For compatibility, do not use

        Returns:
            Dict[str, Any]: The loss of the finetunning step
        """
        loss, y, y_hat = self.get_finetunning_loss(batch)
        self.finetunning_step_outputs.append(
            {
            "loss" : loss,
            "y" : y,
            "y_hat" : y_hat
        }
        )
        return {
            "loss" : loss
        }
    
    def on_finetunning_validation_start(self):
        """Log the training loss and the performance of the finetunning
        """
        if len(self.finetunning_step_outputs) > 0:
            train_loss = torch.Tensor([out["loss"] for out in self.finetunning_step_outputs]).cpu().mean()
            y = torch.cat([out["y"] for out in self.finetunning_step_outputs]).cpu().detach().numpy()
            y_hat = torch.cat([out["y_hat"] for out in self.finetunning_step_outputs]).cpu().detach().numpy()
            
            train_score = self.scorer(y, y_hat)
            
            self.log("train_loss", train_loss, prog_bar = True)
            self.log("train_" + self.scorer.__name__, train_score, prog_bar = True)
            self.finetunning_step_outputs = []   
            
        return super().on_validation_start()
    
    def finetuning_validation_epoch_end(self) -> None:
        """Log the validation loss and the performance of the finetunning
        """
        val_loss = torch.Tensor([out["loss"] for out in self.finetunning_step_outputs]).cpu().mean()

        y = torch.cat([out["y"] for out in self.finetunning_step_outputs]).cpu().numpy()
        y_hat = torch.cat([out["y_hat"] for out in self.finetunning_step_outputs]).cpu().numpy()
        val_score = self.scorer(y, y_hat)

        self.log("val_" + self.scorer.__name__, val_score, prog_bar = True)
        self.log("val_loss", val_loss, prog_bar = True)
        self.finetunning_step_outputs = []      
        return super().on_validation_epoch_end()
    
    @abstractmethod
    def predict_step(self, batch, batch_idx: int
    ) -> torch.FloatTensor:
        """The perdict step

        Args:
            batch (Dict[str, Any]): The input batch
            batch_idx (int): For compatibility, do not use

        Returns:
            torch.FloatTensor: The predicted output (logit)
        """
        pass