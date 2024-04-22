from dataclasses import dataclass, field
from ts3l.utils import BaseConfig

from typing import Any, List, Optional

@dataclass
class SubTabConfig(BaseConfig):
    """ Configuration class for initializing components of the SubTabLightning Module, including hyperparameters of SubTab,
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
        hidden_dim (int): The dimension of hidden layer. Default is 256.
        tau (float): A hyperparameter that is to scale similarity between projections during the first phase.
        use_cosine_similarity (bool):  A hyperparameter that is to select whether using cosine similarity or dot similarity when calculating similarity
                                        between projections during the first phase. Default is False.
        use_contrastive (bool): A hyperparameter that is to select using contrastive loss or not during the first phase. Default is True.
        use_distance (bool): A hyperparameter that is to select using distance loss or not during the first phase. Default is True.
        n_subsets (int): The number of subsets to generate different views of the data. Default is 4.
        overlap_ratio (float): A hyperparameter that is to control the extent of overlapping between the subsets. Default is 0.75.
        shuffle (bool): Whether to shuffle the subsets. 
        mask_ratio (float): Ratio of features to be masked as noise.
        noise_type (str): The type of noise to apply.
        noise_level (float): Intensity of Gaussian noise to be applied.

    Raises:
        ValueError: Inherited from `BaseConfig` to indicate that a configuration for the task, optimizer, scheduler, loss function, or metric is either invalid or not specified.
        ValueError: If the specified 'noise_type' is not in ["Swap", "Gaussian", "Zero_Out"].
        ValueError: If the specified 'noise_level' is not a valid value.
    """
    
    hidden_dim: int = field(default=256)
    
    tau: float = field(default=0.1)
    
    use_cosine_similarity: bool = field(default=False)
    
    use_contrastive: bool = field(default=True)
    
    use_distance: bool = field(default=True)
    
    n_subsets: int = field(default=4)
    
    overlap_ratio: float = field(default=0.75)
    
    shuffle: bool = field(default=False)
    
    mask_ratio: float = field(default=0.2)
    
    noise_type: str = field(default="Swap")
    
    noise_level: float = field(default=0)
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.noise_type not in ["Swap", "Gaussian", "Zero_Out"]:
            raise ValueError('The noise type must be one of ["Swap", "Gaussian", "Zero_Out"], but %s.' % self.noise_type)
        
        if (self.noise_type == "Gaussian") and (self.noise_level <= 0):
            raise ValueError("The noise level must be a float that is > 0 when the noise type is Gaussian.")