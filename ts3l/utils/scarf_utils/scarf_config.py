from dataclasses import dataclass, field
from ts3l.utils import BaseConfig

from typing import Any, List

@dataclass
class SCARFConfig(BaseConfig):
    """ Configuration class for initializing components of the SCARFLightning Module, including hyperparameters of SCARF,
    optimizers, learning rate schedulers, and loss functions, along with their respective hyperparameters.

    Inherits Attributes:
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
        
    New Attributes:
        input_dim (int): The dimension of the input.
        hidden_dim (int): The dimension of hidden layer. Default is 256.
        output_dim (int): The dimension of output.
        encoder_depth (bool):  The depth of encoder. Default is 4.
        head_depth (bool): The depth of head. Default is 2.
        dropout_rate (bool): A hyperparameter that is to control dropout layer. Default is 0.04.
        tau (float): A hyperparameter that is to scale similarity between views during the first phase.

    Raises:
        ValueError: Inherited from 'BaseConfig' to indicate that a configuration for the task, optimizer, scheduler, loss function, or metric is either invalid or not specified.
        
        ValueError: Raised if 'input_dim' or 'output_dim' are not specified, indicating these dimensions must be defined.                    
    """
    
    input_dim: int = field(default=None)
    
    hidden_dim: int = field(default=256)
    
    output_dim: int = field(default=None)
    
    encoder_depth: int = field(default=4)
    
    head_depth: int = field(default=2)
    
    dropout_rate: float = field(default=0.04)
    
    tau: float = field(default=0.1)
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.input_dim is None:
            raise ValueError("The dimension of input must be specified in the 'input_dim' attribute.")
        
        if self.output_dim is None:
            raise ValueError("The dimension of output must be specified in the 'output_dim' attribute.")