from dataclasses import dataclass, field

from typing import Dict, Any
from torch import optim, nn
import torchmetrics
import sklearn

@dataclass
class BaseConfig:
    """ Configuration class for initializing components of the TabularS3L Lightning Module, including optimizers, 
    learning rate schedulers, and loss functions, along with their respective hyperparameters.

    Attributes:
        task (str): Specify whether the problem is regression or classification.
        optim (str): Name of the optimizer to be used. Must be an attribute of 'torch.optim'. Default is 'AdamW'.
        optim_hparams (Dict[str, Any]): Hyperparameters for the optimizer. Default is {'lr': 0.0001, 'weight_decay': 0.00005}.
        scheduler (str): Name of the learning rate scheduler to be used. Must be an attribute of 'torch.optim.lr_scheduler' or None. Default is None.
        scheduler_hparams (Dict[str, Any]): Hyperparameters for the scheduler. Default is None, indicating no scheduler is used.
        loss_fn (str): Name of the loss function to be used. Must be an attribute of 'torch.nn'.
        loss_hparams (Dict[str, Any]): Hyperparameters for the loss function. Default is empty dictionary.
        metric (str): Name of the metric to be used. Must be an attribute of 'torchmetrics.functional' or 'sklearn.metrics'. Default is None.
        metric_hparams (Dict[str, Any]): Hyperparameters for the metric. Default is an empty dictionary.
        random_seed (int): Seed for random number generators to ensure reproducibility. Defaults to 42.

    Raises:
        ValueError: If the specified 'optim' is not a valid optimizer in 'torch.optim'.
        ValueError: If the specified 'scheduler' is not None and is not a valid scheduler in 'torch.optim.lr_scheduler'.
        
        ValueError: If the 'loss_fn' attribute is None, indicating that a loss function must be specified.
        ValueError: If the specified 'loss_fn' is not None and is not a valid loss function in 'torch.nn'.
        
        ValueError: If the 'metric' attribute is None, indicating that a metric must be specified.
        ValueError: If the specified 'metric' is not a valid metric in 'torchmetrics' or 'sklearn.metrics'.
        
        ValueError: If the 'task' attribute is None, indicating that a task must be specified.
        ValueError: If the specified 'task' is not a valid task in ['regression', 'classification']'.
        
    """
    task: str = field(default=None)
    
    optim: str = field(default="AdamW")
    
    optim_hparams: Dict[str, Any] = field(
                                            default_factory=lambda: {
                                                    "lr" : 0.0001,
                                                    "weight_decay" : 0.00005
                                            }
                                        )
    
    scheduler: str = field(default=None)
    
    scheduler_hparams: Dict[str, Any] = field(default = None)
    
    loss_fn: str = field(default=None)
    
    loss_hparams: Dict[str, Any] = field(default_factory=dict)
    
    metric: str = field(default=None)
    
    metric_hparams: Dict[str, Any] = field(default_factory=dict)
    
    random_seed: int = field(default=42)
    
    def __post_init__(self):
        if self.task is None:
            raise ValueError("The task of the problem must be specified in the 'task' attribute.")
        elif (type(self.task) is not str or (self.task != "regression" and self.task != "classification")):
            raise ValueError(f"{self.task} is not a valid task. Choices are: ['regression', 'classification']")
        
        if type(self.optim) is not str or not hasattr(optim, self.optim):
            raise ValueError(f"{self.optim} is not a valid optimizer in torch.optim")
        
        if self.scheduler is not None and (type(self.scheduler) is not str or not hasattr(optim.lr_scheduler, self.scheduler)):
            raise ValueError(f"{self.scheduler} is not a valid scheduler in torch.optim.lr_scheduler")
        
        if self.loss_fn is None:
            raise ValueError("A loss function must be specified in the 'loss_fn' attribute.")
        elif type(self.loss_fn) is not str or not hasattr(nn, self.loss_fn):
            raise ValueError(f"{self.loss_fn} is not a valid loss function in torch.nn")
        
        if self.metric is None:
            raise ValueError("A metric must be specified in the 'metric' attribute.")
        elif (type(self.metric) is not str or (not hasattr(torchmetrics.functional, self.metric) and not hasattr(sklearn.metrics, self.metric))):
            raise ValueError(f"{self.metric} is not a valid metric in torchmetrics.functional or sklearn.metrics")