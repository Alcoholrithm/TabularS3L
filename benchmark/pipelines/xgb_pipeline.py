import argparse
import pandas as pd
from typing import List, Dict, Any, Union, Type

from dataclasses import dataclass, field, asdict
from .pipeline import PipeLine
from xgboost import XGBClassifier, XGBRegressor
from hparams_range.xgb import hparams_range

@dataclass
class XGBConfig:
    
    max_leaves: int
    
    n_estimators: int
    
    learning_rate: float
    
    max_depth: int
    
    scale_pos_weight: int
    
    early_stopping_rounds: int

class XGBModule(object):
    def __init__(self, model_class: Union[XGBClassifier, XGBRegressor]):
        self.model_class = model_class
    
    def __call__(self, config: XGBConfig):
        return self.model_class(**asdict(config))
    
class XGBPipeLine(PipeLine):
    def __init__(self, args: argparse.Namespace, data: pd.DataFrame, label: pd.Series, continuous_cols: List[str], category_cols: List[str], output_dim: int, metric: str, metric_hparams: Dict[str, Any] = {}):
        super().__init__(args, data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams)
    
    def initialize(self):
        self.config_class = XGBConfig
        if self.output_dim == 1:
            self.pl_module_class = XGBRegressor
        else:
            self.pl_module_class = XGBClassifier
        self.pl_module_class = XGBModule(self.pl_module_class)
        
        self.hparams_range = hparams_range
    
    def _get_config(self, hparams: Dict[str, Any]):
        hparams["early_stopping_rounds"] = self.args.second_phase_patience
        
        return self.config_class(**hparams)
    
    def fit_model(self, pl_module: XGBModule, config: XGBConfig):

        pl_module.fit(self.X_train,  self.y_train, eval_set=[(self.X_valid, self.y_valid)], verbose = 0)
        
        return pl_module
        
    def evaluate(self, pl_module: XGBModule, config: XGBConfig, X: pd.DataFrame, y: pd.Series):
        
        if self.output_dim == 1:
            preds = pl_module.predict(X)
        else:
            preds = pl_module.predict_proba(X)
        
        # print(preds.shape, y.shape)
        score = self.metric(preds, y)
        
        return score

