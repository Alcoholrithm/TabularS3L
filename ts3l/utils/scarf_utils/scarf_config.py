from dataclasses import dataclass, field
from ts3l.utils import BaseConfig

from typing import Any, List

@dataclass
class SCARFConfig(BaseConfig):
    """ Configuration class for initializing components of the SCARFLightning Module, including hyperparameters of SCARF,
    optimizers, learning rate schedulers, and loss functions, along with their respective hyperparameters.

    Inherits Attributes:
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
        random_seed (int): Seed for random number generators to ensure reproducibility. Defaults to 42.
        
    New Attributes:
        hidden_dim (int): The dimension of hidden layer. Default is 256.
        encoder_depth (bool):  The depth of encoder. Default is 4.
        head_depth (bool): The depth of head. Default is 2.
        dropout_rate (bool): A hyperparameter that is to control dropout layer. Default is 0.04.
        tau (float): A hyperparameter that is to scale similarity between views during the first phase.
        corruption_rate (float): The proportion of features to be corrupted, simulating noisy conditions for robustness. 
                For the second phase dataset, it should be 0. Defaults to 0.0.

    Raises:
        ValueError: Inherited from 'BaseConfig' to indicate that a configuration for the task, optimizer, scheduler, loss function, or metric is either invalid or not specified.
        
    """
    
    hidden_dim: int = field(default=256)
    
    encoder_depth: int = field(default=4)
    
    head_depth: int = field(default=2)
    
    dropout_rate: float = field(default=0.04)
    
    tau: float = field(default=0.1)
    
    corruption_rate: float = field(default=0)
    
    
    def __post_init__(self):
        super().__post_init__()