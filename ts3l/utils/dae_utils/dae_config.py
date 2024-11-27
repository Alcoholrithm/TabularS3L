from dataclasses import dataclass, field
from ts3l.utils import BaseConfig

from typing import Any, List, Optional

@dataclass
class DAEConfig(BaseConfig):
    """ Configuration class for initializing components of the DAELightning Module, including hyperparameters of Denoising AutoEncoder,
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
        noise_type (str): The type of noise to apply. Choices are ["Swap", "Gaussian", "Zero_Out"].
        noise_level (float): Intensity of Gaussian noise to be applied.
        noise_ratio (float): A hyperparameter that is to control the noise ratio during the first phase learning. Default is 0.3.
        mask_loss_weight (float): The special token for unlabeled samples.
        dropout_rate (bool): A hyperparameter that is to control dropout layer. Default is 0.04.
        cat_cardinality (List[int]): The cardinality of categorical features.
        num_continuous (int): The number of continuous features.
        
    Raises:
        ValueError: Inherited from `BaseConfig` to indicate that a configuration for the task, optimizer, scheduler, loss function, or metric is either invalid or not specified.
        ValueError: If the specified 'noise_type' is not in ["Swap", "Gaussian", "Zero_Out"].
        ValueError: If the specified 'noise_level' is not a valid value.
        ValueError: Raised if both `num_categoricals` and `num_continuous` are None, indicating that at least one attribute must be specified.
    """
    
    noise_type: str = field(default="Swap")
    
    noise_level: float = field(default=0)
    
    noise_ratio: float = field(default=0.3)
    
    mask_loss_weight: float = field(default=1.0)
    
    dropout_rate: float = field(default=0.04)
    
    cat_cardinality: List[int] = field(default_factory=lambda: [])
    
    num_continuous: Optional[int] = field(default=None)
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.noise_type not in ["Swap", "Gaussian", "Zero_Out"]:
            raise ValueError('The noise type must be one of ["Swap", "Gaussian", "Zero_Out"], but %s.' % self.noise_type)
        
        if (self.noise_type == "Gaussian") and ((self.noise_level == None) or (self.noise_level <= 0)):
            raise ValueError("The noise level must be a float that is > 0 when the noise type is Gaussian.")
        
        if len(self.cat_cardinality) == 0 and self.num_continuous is None:
            raise ValueError("At least one attribute (num_categorical or num_continuous) must be specified.")
        else:
            self.num_continuous = self.num_continuous if self.num_continuous is not None else 0