from dataclasses import dataclass, field
from ts3l.utils import BaseConfig

from typing import Any, List, Optional

@dataclass
class SwitchTabConfig(BaseConfig):
    """ Configuration class for initializing components of the SwitchTabLightning Module, including hyperparameters of SwitchTab,
    optimizers, learning rate schedulers, and loss functions, along with their respective hyperparameters.

    Inherits Attributes:
        task (str): Specify whether the problem is regression or classification.
        embedding_config (BaseEmbeddingConfig): Configuration for the embedding layer.
        backbone_config (BaseBackboneConfig): Configuration for the backbone network.
        output_dim (int): The dimension of output.
        loss_fn (str): Name of the loss function to be used. Must be an attribute of 'torch.nn'.
        loss_hparams (Dict[str, Any]): Hyperparameters for the loss function. Default is empty dictionary.
        metric (str): Name of the metric to be used. Must be an attribute of 'torchmetrics.functional' or 'sklearn.metrics'. Default is None.
        metric_hparams (Dict[str, Any]): Hyperparameters for the metric. Default is an empty dictionary.
        optim (str): Name of the optimizer to be used. Must be an attribute of 'torch.optim'. Default is 'AdamW'.
        optim_hparams (Dict[str, Any]): Hyperparameters for the optimizer. Default is {'lr': 0.0001, 'weight_decay': 0.00005}.
        scheduler (str): Name of the learning rate scheduler to be used. Must be an attribute of 'torch.optim.lr_scheduler' or None. Default is None.
        scheduler_hparams (Dict[str, Any]): Hyperparameters for the scheduler. Default is None, indicating no scheduler is used.
        initialization (str): The way to initialize neural network parameters. Default is 'kaiming_uniform'.
        random_seed (int): Seed for random number generators to ensure reproducibility. Defaults to 42.
        
    New Attributes:
        u_label (Any): The special token for unlabeled samples.
        corruption_rate (float): The proportion of features to be corrupted. Default is 0.3.
        alpha (float): A hyperparameter that is to control the trade-off between the reconstruction loss and task loss during first phase. Default is 1.0.
    Raises:
        ValueError: Inherited from `BaseConfig` to indicate that a configuration for the task, optimizer, scheduler, loss function, or metric is either invalid or not specified.
    """

    u_label: Any = field(default=-1)
    
    corruption_rate: float = field(default=0.3)
    
    alpha: float = field(default=1.0)

    def __post_init__(self):
        super().__post_init__()