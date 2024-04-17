from dataclasses import dataclass, field
from ts3l.utils import BaseConfig

from typing import Any, List, Optional

@dataclass
class DAEConfig(BaseConfig):
    """ Configuration class for initializing components of the DAELightning Module, including hyperparameters of Denoising AutoEncoder,
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
        noise_type (str): The type of noise to apply. Choices are ["Swap", "Gaussian", "Zero_Out"].
        noise_level (float): Intensity of Gaussian noise to be applied.
        noise_ratio (float): A hyperparameter that is to control the noise ratio during the first phase learning. Default is 0.3.
        mask_loss_weight (float): The special token for unlabeled samples.
        dropout_rate (bool): A hyperparameter that is to control dropout layer. Default is 0.04.

    Raises:
        ValueError: Inherited from `BaseConfig` to indicate that a configuration for the task, optimizer, scheduler, loss function, or metric is either invalid or not specified.
        ValueError: If the specified 'noise_type' is not in ["Swap", "Gaussian", "Zero_Out"].
        ValueError: If the specified 'noise_level' is not a valid value.
    """
    
    hidden_dim: int = field(default=256)
    
    noise_type: str = field(default="Swap")
    
    noise_level: Optional[float] = field(default=None)
    
    noise_ratio: float = field(default=0.3)
    
    mask_loss_weight: float = field(default=1.0)
    
    dropout_rate: float = field(default=0.04)
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.noise_type not in ["Swap", "Gaussian", "Zero_Out"]:
            raise ValueError('The noise type must be one of ["Swap", "Gaussian", "Zero_Out"], but %s.' % self.noise_type)
        
        if (self.noise_type == "Gaussian") and (self.noise_level == None) or (self.noise_level <= 0):
            raise ValueError("The noise level must be a float that is > 0 when the noise type is Gaussian.")