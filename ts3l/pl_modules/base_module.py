from typing import Dict, Any, Type
from ts3l.utils import RegressionMetric, ClassificationMetric

from abc import ABC, abstractmethod

import torch
from torch import nn

import pytorch_lightning as pl

from dataclasses import asdict
from ts3l.utils import BaseConfig
from typing import Type
    

    
class TS3LLightining(ABC, pl.LightningModule):
    """The pytorch lightning module of TabularS3L
    """
    def __init__(self, config: Type[BaseConfig]) -> None:
        """Initialize the pytorch lightining module of TabularS3L

        Args:
            config (Type[BaseConfig]): The configuration of TS3LLightining.
        """
        super(TS3LLightining, self).__init__()
        
        config = asdict(config)
        
        self.random_seed = config["random_seed"]
        del config["random_seed"]
        
        pl.seed_everything(self.random_seed)
        
        self.optim = getattr(torch.optim, config["optim"])
        del config["optim"]
        self.optim_hparams = config["optim_hparams"]
        del config["optim_hparams"]
        
        self.sched = getattr(torch.optim.lr_scheduler, config["scheduler"]) if config["scheduler"] is not None else None
        del config["scheduler"]
        self.scheduler_hparams = config["scheduler_hparams"]
        del config["scheduler_hparams"]
        
        self.loss_fn = getattr(torch.nn, config["loss_fn"])(**config["loss_hparams"])
        del config["loss_fn"]
        del config["loss_hparams"]
        
        self.__configure_metric(config["task"], config["metric"], config["metric_hparams"])
        del config["task"]
        del config["metric"]
        del config["metric_hparams"]
        
        self._initialize(config)
        
        self.set_first_phase()

        self.first_phase_step_outputs = []
        self.second_phase_step_outputs = []
        
        self.save_hyperparameters()
    
    @abstractmethod
    def _initialize(self, config: Dict[str, Any]) -> None:
        pass
    
    def __configure_metric(self, task, metric, metric_hparams):
        
        if task == "regression":
            self.metric = RegressionMetric(metric, metric_hparams)
        else:
            self.metric = ClassificationMetric(metric, metric_hparams)
            
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
            train_loss = torch.Tensor([out["loss"] for out in self.second_phase_step_outputs]).detach().mean()
            y = torch.cat([out["y"] for out in self.second_phase_step_outputs]).detach()
            y_hat = torch.cat([out["y_hat"] for out in self.second_phase_step_outputs]).detach()
            
            train_score = self.metric(y_hat, y)
            
            self.log("train_loss", train_loss, prog_bar = True)
            self.log("train_" + self.metric.__name__, train_score, prog_bar = True)
            self.second_phase_step_outputs = []   
            
        return super().on_validation_start()
    
    def _second_phase_validation_epoch_end(self) -> None:
        """Log the validation loss and the performance of the second phase
        """
        val_loss = torch.Tensor([out["loss"] for out in self.second_phase_step_outputs]).mean()

        
        y = torch.cat([out["y"] for out in self.second_phase_step_outputs])
        y_hat = torch.cat([out["y_hat"] for out in self.second_phase_step_outputs])
        val_score = self.metric(y_hat, y)

        self.log("val_" + self.metric.__name__, val_score, prog_bar = True)
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