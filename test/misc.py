from types import SimpleNamespace
from typing import List, Dict, Any
import pandas as pd

from ts3l.utils.embedding_utils import BaseEmbeddingConfig, IdentityEmbeddingConfig, FTEmbeddingConfig
from ts3l.utils.backbone_utils import BaseBackboneConfig, MLPBackboneConfig, TransformerBackboneConfig
from ts3l.utils.misc import get_category_dims

def get_args():
    args = SimpleNamespace()
    
    args.max_epochs = 1
    args.first_phase_patience = 1
    args.second_phase_patience = 1
    args.n_trials = 1

    args.labeled_sample_ratio = 1
    args.valid_size = 0.2
    args.test_size = 0.2
    args.random_seed = 0
    args.batch_size = 128
    
    args.n_jobs = 4
    args.accelerator = "cpu"
    args.devices = "auto"
    
    return args

def get_embedding_config(data: pd.DataFrame, embedding_type: str, input_dim: int, continuous_cols: List[str], category_cols: List[str], required_token_dim: int = 1):
    if embedding_type == "identity":
        embedding_config = IdentityEmbeddingConfig(
            input_dim=input_dim,
        )
    elif embedding_type == "feature_tokenizer":
        embedding_config = FTEmbeddingConfig(
            input_dim = input_dim,
            cont_nums = len(continuous_cols),
            cat_dims = get_category_dims(data, category_cols),
            required_token_dim = required_token_dim
        )
    return embedding_config

def get_backbone_config(backbone_type: str, input_dim: int):
    if backbone_type == "mlp":
        backbone_config = MLPBackboneConfig(
            input_dim = input_dim,
            output_dim = 128
        )
    elif backbone_type == "transformer":
        backbone_config = TransformerBackboneConfig(
            d_model = input_dim
        )
    return backbone_config

def get_base_config_kwargs(
        data: pd.DataFrame, 
        input_dim: int, 
        continuous_cols: List[str], 
        category_cols: List[str],
        embedding_type: str, 
        backbone_type: str,
        output_dim: int,
        metric: str, 
        metric_hparams: Dict[str, Any],
    ):
    
    embedding_config = get_embedding_config(data, embedding_type, input_dim, continuous_cols, category_cols,
                                            required_token_dim = 2 if embedding_type == "feature_tokenizer" and backbone_type == "transformer" else 1)

    backbone_config = get_backbone_config(backbone_type, 
                                            input_dim = embedding_config.emb_dim if backbone_type == "transformer" and embedding_type == "feature_tokenizer" else embedding_config.output_dim)
    
    kwargs = {}
    kwargs["task"] = "classification" if output_dim > 1 else "regression"
    kwargs["embedding_config"] = embedding_config
    kwargs["backbone_config"] = backbone_config
    kwargs["output_dim"] = output_dim
    kwargs["loss_fn"] = "CrossEntropyLoss" if output_dim > 1 else "MSELoss"
    kwargs["metric"] = metric
    kwargs["metric_hparams"] = metric_hparams
    
    return kwargs

def prepare_test(load_data, embedding_type, backbone_type):
    
    data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams = load_data()
    kwargs = get_base_config_kwargs(data, data.shape[1], continuous_cols, category_cols, embedding_type, backbone_type, output_dim, metric, metric_hparams)
    
    return data, label, continuous_cols, category_cols, output_dim, kwargs
    
embedding_backbone_list = [
                            ("identity", "mlp"), 
                            ("feature_tokenizer", "mlp"), 
                            ("feature_tokenizer", "transformer")
                            ]