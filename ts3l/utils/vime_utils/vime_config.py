from dataclasses import dataclass, field
from ts3l.utils import BaseConfig

from typing import Any, List

@dataclass
class VIMEConfig(BaseConfig):
    """ Configuration class for initializing components of the VIMELightning Module, including hyperparameters of VIME,
    optimizers, learning rate schedulers, and loss functions, along with their respective hyperparameters.

    Inherits Attributes:
        task (str): Specify whether the problem is regression or classification.
        optim (str): Name of the optimizer to be used. Must be an attribute of `torch.optim`. Default is 'AdamW'.
        optim_hparams (Dict[str, Any]): Hyperparameters for the optimizer. Default is {'lr': 0.0001, 'weight_decay': 0.00005}.
        scheduler (str): Name of the learning rate scheduler to be used. Must be an attribute of `torch.optim.lr_scheduler` or None. Default is None.
        scheduler_hparams (Dict[str, Any]): Hyperparameters for the scheduler. Default is None, indicating no scheduler is used.
        loss_fn (str): Name of the loss function to be used. Must be an attribute of `torch.nn`.
        loss_hparams (Dict[str, Any]): Hyperparameters for the loss function. Default is empty dictionary.
        metric (str): Name of the metric to be used. Must be an attribute of `torchmetrics.functional` or 'sklearn.metrics'. Default is None.
        metric_hparams (Dict[str, Any]): Hyperparameters for the metric. Default is an empty dictionary.
        random_seed (int): Seed for random number generators to ensure reproducibility. Defaults to 42.
        
    New Attributes:
        input_dim (int): The dimension of the input.
        hidden_dim (int): The hidden dimension of predictor. Default is 256.
        output_dim (int): The output dimension of predictor.
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

    Raises:
        ValueError: Inherited from `BaseConfig` to indicate that a configuration for the task, optimizer, scheduler, loss function, or metric is either invalid or not specified.

        ValueError: Raised if both `num_categoricals` and `num_continuous` are None, indicating that at least one attribute must be specified.
        ValueError: Raised if `input_dim` or `output_dim` are not specified, indicating these dimensions must be defined.                    
    """
    
    input_dim: int = field(default=None)
    
    hidden_dim: int = field(default=256)
    
    output_dim: int = field(default=None)
    
    num_categoricals: int = field(default=None)
    
    num_continuous: int = field(default=None)
    
    u_label: Any = field(default=-1)
    
    alpha1: float = field(default=2.0)
    
    alpha2: float = field(default=2.0)
    
    beta: float = field(default=1.0)
    
    K: int = field(default=3)
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.input_dim is None:
            raise ValueError("The dimension of input must be specified in the 'input_dim' attribute.")
        
        if self.output_dim is None:
            raise ValueError("The dimension of predictor's output must be specified in the 'output_dim' attribute.")
        
        if self.num_categoricals is None and self.num_continuous is None:
            raise ValueError("At least one attribute (num_categorical or num_continuous) must be specified.")
        else:
            self.num_categoricals = self.num_categoricals if self.num_categoricals is not None else 0
            self.num_continuous = self.num_continuous if self.num_continuous is not None else 0