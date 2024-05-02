from dataclasses import dataclass, field

from typing import Dict, Any, Optional
from torch import optim, nn
import torchmetrics
import sklearn

import sys
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

@dataclass
class BaseConfig:
    """ Configuration class for initializing components of the TabularS3L Lightning Module, including optimizers, 
    learning rate schedulers, and loss functions, along with their respective hyperparameters.

    Attributes:
        task (str): Specify whether the problem is regression or classification.
        input_dim (int): The dimension of the input.
        output_dim (int): The dimension of output.
        loss_fn (str): Name of the loss function to be used. Must be an attribute of 'torch.nn'.
        optim (str): Name of the optimizer to be used. Must be an attribute of 'torch.optim'. Default is 'AdamW'.
        optim_hparams (Dict[str, Any]): Hyperparameters for the optimizer. Default is {'lr': 0.0001, 'weight_decay': 0.00005}.
        scheduler (str): Name of the learning rate scheduler to be used. Must be an attribute of 'torch.optim.lr_scheduler' or None. Default is None.
        scheduler_hparams (Dict[str, Any]): Hyperparameters for the scheduler. Default is None, indicating no scheduler is used.
        loss_hparams (Dict[str, Any]): Hyperparameters for the loss function. Default is empty dictionary.
        metric (str): Name of the metric to be used. Must be an attribute of 'torchmetrics.functional' or 'sklearn.metrics'. Default is None.
        metric_hparams (Dict[str, Any]): Hyperparameters for the metric. Default is an empty dictionary.
        initialization (str): The way to initialize neural network parameters. Default is 'kaiming_uniform'.
        random_seed (int): Seed for random number generators to ensure reproducibility. Defaults to 42.

    Raises:
        ValueError: If the specified 'optim' is not a valid optimizer in 'torch.optim'.
        ValueError: If the specified 'scheduler' is not None and is not a valid scheduler in 'torch.optim.lr_scheduler'.

        ValueError: If the specified 'loss_fn' is not None and is not a valid loss function in 'torch.nn'.
        
        ValueError: If the specified 'metric' is not a valid metric in 'torchmetrics' or 'sklearn.metrics'.
        
        ValueError: If the specified 'task' is not a valid task in ['regression', 'classification']'.
        
    """
    task: str
    
    input_dim: int
    
    output_dim: int
    
    loss_fn: str
    
    loss_hparams: Dict[str, Any] = field(default_factory=dict)
    
    metric: Optional[str] = field(default=None)
    
    metric_hparams: Dict[str, Any] = field(default_factory=dict)
    
    optim: str = field(default="AdamW")
    
    optim_hparams: Dict[str, Any] = field(
                                            default_factory=lambda: {
                                                    "lr" : 0.0001,
                                                    "weight_decay" : 0.00005
                                            }
                                        )
    
    scheduler: Optional[str] = field(default=None)
    
    scheduler_hparams: Optional[Dict[str, Any]] = field(default = None)
    
    initialization: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'uniform', 'normal'] = "kaiming_uniform"
    
    random_seed: int = field(default=42)
    
    def __post_init__(self):

        if (type(self.task) is not str or (self.task != "regression" and self.task != "classification")):
            raise ValueError(f"{self.task} is not a valid task. Choices are: ['regression', 'classification']")
        
        if type(self.optim) is not str or not hasattr(optim, self.optim):
            raise ValueError(f"{self.optim} is not a valid optimizer in torch.optim")
        
        if self.scheduler is not None and (type(self.scheduler) is not str or not hasattr(optim.lr_scheduler, self.scheduler)):
            raise ValueError(f"{self.scheduler} is not a valid scheduler in torch.optim.lr_scheduler")

        if type(self.loss_fn) is not str or not hasattr(nn, self.loss_fn):
            raise ValueError(f"{self.loss_fn} is not a valid loss function in torch.nn")
        
        if self.metric is None:
            if self.task == "regression":
                self.metric = "mean_squared_error"
            else:
                self.metric = "accuracy_score"
                
        elif (type(self.metric) is not str or (not hasattr(torchmetrics.functional, self.metric) and not hasattr(sklearn.metrics, self.metric))):
            raise ValueError(f"{self.metric} is not a valid metric in torchmetrics.functional or sklearn.metrics")