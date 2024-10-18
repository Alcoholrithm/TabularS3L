from dataclasses import dataclass, field

from typing import Dict, Any, Optional, Union, List
from torch import optim, nn
import torchmetrics
import sklearn

import sys
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

    
@dataclass
class EmbeddingConfig:
    """ Configuration class for initializing components of the TabularS3L Lightning Module, including optimizers, 
    learning rate schedulers, and loss functions, along with their respective hyperparameters.

    Attributes:
        

    Raises:
        ValueError: If the specified 'optim' is not a valid optimizer in 'torch.optim'.
        ValueError: If the specified 'scheduler' is not None and is not a valid scheduler in 'torch.optim.lr_scheduler'.

        ValueError: If the specified 'loss_fn' is not None and is not a valid loss function in 'torch.nn'.
        
        ValueError: If the specified 'metric' is not a valid metric in 'torchmetrics' or 'sklearn.metrics'.
        
        ValueError: If the specified 'task' is not a valid task in ['regression', 'classification']'.
        
    """
    
    input_dim: int
    
    args: Dict[str, Any] = field(default_factory=dict)
    
    module: Literal['identity', 'feature_tokenizer'] = "identity"
    
    
    def __check_feature_tokenizer(self):

        keys = ["emb_dim", "cont_nums", "cat_dims", "required_token_dim"]
        key_types = [int, int, list, int]

        for key in keys:
            if not key in self.args.keys():
                if key != "emb_dim" and key != "required_token_dim":
                    raise KeyError(f"{key} is not specified in EmbeddingConfig.args")
                elif key == "emb_dim":
                    self.args["emb_dim"] = 128
                    print("'emb_dim' is set to the default value 128 for feature tokenizer.")
                elif key == "required_token_dim":
                    self.args["required_token_dim"] = 1
                    print("'required_token_dim' is set to the default value 1 for feature tokenizer.")
        
        for idx, key in enumerate(keys):
            if not isinstance(self.args[key], key_types[idx]):
                raise ValueError(f"Invalid type for {key}")
            
        for k, v in self.args.items():
            if not k in keys:
                raise KeyError(f"{k} is an invalid key for EmbeddingConfig.args")
        
    def __post_init__(self):

        if not isinstance(self.module, str):
            raise ValueError(f"{self.module} is not a valid type")
        elif not self.module in ["identity", "feature_tokenizer"]:
            raise ValueError(f"{self.module} is not a valid value. Choices are: ['identity', 'feature_tokenizer']")
        
        if self.module == "feature_tokenizer":
            self.__check_feature_tokenizer()