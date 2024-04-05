from dataclasses import dataclass, field
from ts3l.utils import BaseConfig

from typing import Any, List, Optional

@dataclass
class VIMEConfig(BaseConfig):
    """ Configuration class for initializing components of the VIMELightning Module, including hyperparameters of VIME,
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
        num_categoricals int: The number of categorical features.
        num_continuous int: The number of continuous features.
        u_label (Any): The special token for unlabeled samples.
        alpha1 (float): A hyperparameter that is to control the trade-off between 
                        the mask estimation and categorical feature estimation loss during first phase. 
                        Default is 2.0.
        alpha2 (float): A hyperparameter that is to control the trade-off between 
                        the mask estimation and continuous feature estimation loss during first phase. 
                        Default is 2.0.
        beta (float): A hyperparameter that is to control the trade-off between 
                        the supervised and unsupervised loss during second phase. 
                        Default is 1.0.
        K (int): The number of augmented samples for consistency regularization. Default is 3.
        p_m (float): A hyperparameter that is to control the masking ratio during the first phase learning. Default is 0.3.

    Raises:
        ValueError: Inherited from `BaseConfig` to indicate that a configuration for the task, optimizer, scheduler, loss function, or metric is either invalid or not specified.
        ValueError: Raised if both `num_categoricals` and `num_continuous` are None, indicating that at least one attribute must be specified.
        
    """
    
    hidden_dim: int = field(default=256)
    
    num_categoricals: Optional[int] = field(default=None)
    
    num_continuous: Optional[int] = field(default=None)
    
    u_label: Any = field(default=-1)
    
    alpha1: float = field(default=2.0)
    
    alpha2: float = field(default=2.0)
    
    beta: float = field(default=1.0)
    
    K: int = field(default=3)
    
    p_m: float = field(default=0.3)
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.num_categoricals is None and self.num_continuous is None:
            raise ValueError("At least one attribute (num_categorical or num_continuous) must be specified.")
        else:
            self.num_categoricals = self.num_categoricals if self.num_categoricals is not None else 0
            self.num_continuous = self.num_continuous if self.num_continuous is not None else 0