from dataclasses import dataclass, field
from ts3l.utils import BaseConfig

from typing import Any, List, Optional


@dataclass
class TabularBinningConfig(BaseConfig):
    """
    Configuration class for initializing components of the TabularBinningLightning Module, including hyperparameters of TabularBinning,
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
        n_bin (int): The number of bin for the pretext task.
        pretext_task (str): The pretext task for the first phase learning.
        decoder_depth (int): The depth of the decoder.

    Raises:
        ValueError: Inherited from `BaseConfig` to indicate that a configuration for the task, optimizer, scheduler, loss function, or metric is either invalid or not specified.
        ValueError: If the specified 'pretext_task' is not in ["BinRecon", "BinXent"].
    """

    n_bin: int = field(default=10)

    pretext_task: str = field(default="BinRecon")

    decoder_dim: int = field(default=128)

    decoder_depth: int = field(default=3)

    p_m: float = field(default=0.2)

    mask_type: str = field(default="constant")

    dropout_rate: float = field(default=0.2)

    def __post_init__(self):
        super().__post_init__()

        if self.pretext_task not in ["BinRecon", "BinXent"]:
            raise ValueError(
                'The pretext task must be one of ["BinRecon", "BinXent"], but %s.' % self.pretext_task)

        if self.pretext_task == "BinRecon":
            self.n_decoder = 1
            self.first_phase_output_dim = self.embedding_config.input_dim
        else:
            self.n_decoder = self.embedding_config.input_dim
            self.first_phase_output_dim = self.n_bin
