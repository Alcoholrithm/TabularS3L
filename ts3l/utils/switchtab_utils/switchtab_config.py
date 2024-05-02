from dataclasses import dataclass, field
from ts3l.utils import BaseConfig

from typing import Any, List, Optional

@dataclass
class SwitchTabConfig(BaseConfig):
    """ Configuration class for initializing components of the SwitchTabLightning Module, including hyperparameters of SwitchTab,
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
        hidden_dim (int): The hidden dimension of predictor. Default is 256.
        encoder_depth (int): The depth of encoder. Default is 3.
        n_head (int): The number of heads in the encoder. Default is 2.
        u_label (Any): The special token for unlabeled samples.
        corruption_rate (float): The proportion of features to be corrupted. Default is 0.3.
        alpha (float): A hyperparameter that is to control the trade-off between the reconstruction loss and task loss during first phase. Default is 1.0.
        dropout_rate (float): A hyperparameter that is to control dropout layer. Default is 0.3.
        ffn_factor (float): Multiple by which the dimension of feed forward network in the transformer scales the input. Defaults to 2.0
        category_dims (List[int]): The cardinality of categorical features. Default is an empty list.
    Raises:
        ValueError: Inherited from `BaseConfig` to indicate that a configuration for the task, optimizer, scheduler, loss function, or metric is either invalid or not specified.
    """
    
    hidden_dim: int = field(default=256)
    
    encoder_depth: int = field(default=3)
    
    n_head: int = field(default=2)
    
    u_label: Any = field(default=-1)
    
    corruption_rate: float = field(default=0.3)
    
    alpha: float = field(default=1.0)
    
    dropout_rate: float = field(default=0.3)
    
    ffn_factor: float = field(default=2.0)
    
    category_dims: List[int] = field(default_factory=lambda: [])
    
    def __post_init__(self):
        super().__post_init__()