import argparse
import pandas as pd
from typing import List, Type, Dict, Any
from ts3l.utils import BaseConfig, RegressionMetric, ClassificationMetric
from ts3l.pl_modules.base_module import TS3LLightining
from ts3l.utils.misc import get_category_dims

from abc import ABC, abstractmethod
import optuna
from sklearn.model_selection import train_test_split

from copy import deepcopy

class PipeLine(ABC):
    
    def __init__(self, 
                 args: argparse.Namespace, 
                 data: pd.DataFrame, 
                 label: pd.Series,
                 continuous_cols: List[str],
                 category_cols: List[str],
                 output_dim: int,
                 metric: str, 
                 metric_hparams: Dict[str, Any] = {}):
        self.args = args
        self.data = data
        self.label = label
        self.continuous_cols = continuous_cols
        self.category_cols = category_cols
        self.output_dim = output_dim
        self.metric = metric
        self.metric_hparams = metric_hparams

        X_train, X_valid, y_train, y_valid = train_test_split(data, label, test_size = args.valid_size + args.test_size, random_state=args.random_seed)

        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(X_valid, y_valid, test_size = args.test_size / (args.valid_size + args.test_size), random_state=args.random_seed)
        
        if args.labeled_sample_ratio == 1:
            self.X_train, self.y_train = X_train, y_train
            self.X_unlabeled = None
        else:
            self.X_train, self.X_unlabeled, self.y_train, _ = train_test_split(X_train, y_train, train_size = args.labeled_sample_ratio, random_state=args.random_seed)
            
        self.direction = "maximize" if self.output_dim > 1 else "minimize"
        
        self.config_class = None
        self.pl_module_class = None
        self.hparams_range = None

        self.__configure_metric()
        
        self.initialize()
        # self._set_embedding_config()
        # self._set_backbone_config()
        
        self.check_attributes()
        
    def __objective(self, trial: optuna.trial.Trial,
            ) -> float:
        
        hparams = {}
        for k, v in self.hparams_range.items():
            hparams[k] = getattr(trial, v[0])(*v[1])

        config = self._get_config(hparams)

        pl_module = self.pl_module_class(config)

        pl_module = self.fit_model(pl_module, config)

        return self.evaluate(pl_module, config, self.X_valid, self.y_valid)
    
    @abstractmethod
    def fit_model(self, pl_module: TS3LLightining, config: Type[BaseConfig]):
        pass
    
    @abstractmethod
    def evaluate(self, pl_module: TS3LLightining, config: Type[BaseConfig], X: pd.DataFrame, y: pd.Series):
        pass
    
    @abstractmethod
    def initialize(self):
        
        if self.args.embedding == "identity":
            from ts3l.utils.embedding_utils import IdentityEmbeddingConfig
            self._embedding_config_cls = IdentityEmbeddingConfig
        elif self.args.embedding == "feature_tokenizer":
            from ts3l.utils.embedding_utils import FTEmbeddingConfig
            from hparams_range.embeddings.feature_tokenizer import hparams_range as embedding_hparams_range
            
            for k, v in embedding_hparams_range.items():
                v = deepcopy(embedding_hparams_range[k])
                v[1][0] = "embedding_" + v[1][0]
                self.hparams_range["embedding_" + k] = v
                
            self._embedding_config_cls = FTEmbeddingConfig
            
        if self.args.backbone == "mlp":
            from hparams_range.backbones.mlp import hparams_range as backbone_hprams_range
            from ts3l.utils.backbone_utils import MLPBackboneConfig
            
            self._backbone_config_cls = MLPBackboneConfig
            
            for k, v in backbone_hprams_range.items():
                v = deepcopy(backbone_hprams_range[k])
                v[1][0] = "backbone_" + v[1][0]
                self.hparams_range["backbone_" + k] = v
                
        elif self.args.backbone == "transformer":
            from hparams_range.backbones.transformer import hparams_range as backbone_hprams_range
            for k, v in backbone_hprams_range.items():
                v = deepcopy(backbone_hprams_range[k])
                v[1][0] = "backbone_" + v[1][0]
                self.hparams_range["backbone_" + k] = v
                
            from ts3l.utils.backbone_utils import TransformerBackboneConfig
            self._backbone_config_cls = TransformerBackboneConfig
        
        self.category_dims = get_category_dims(self.data, self.category_cols)
        
    def check_attributes(self):
        if self.config_class is None:
            raise NotImplementedError('self.config_class must be defined')
        if self.pl_module_class is None:
            raise NotImplementedError('self.pl_module_class must be defined')
        if self.hparams_range is None:
            raise NotImplementedError('self.hparams_range must be defined')

    def __tune_hyperparameters(self):
        study = optuna.create_study(direction=self.direction,sampler=optuna.samplers.TPESampler(seed=self.args.random_seed))
        study.optimize(self.__objective, n_trials=self.args.n_trials, show_progress_bar=False)

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")

        trial = study.best_trial
        hparams = dict(trial.params.items())
        
        print("  Evaluation Results: {}".format(trial.value))
        print("  Best hyperparameters: ", hparams)
        
        return hparams

    def __configure_metric(self):
        
        if self.output_dim == 1:
            self.metric = RegressionMetric(self.metric, self.metric_hparams)
        else:
            self.metric = ClassificationMetric(self.metric, self.metric_hparams)
        
    @abstractmethod
    def _get_config(self, _hparams: Dict[str, Any]):
        hparams = deepcopy(_hparams)

        hparams = self._set_embedding_config(hparams)
        hparams = self._set_backbone_config(hparams)
        
        hparams["optim_hparams"] = {
            "lr" : hparams["lr"],
            "weight_decay": hparams["weight_decay"]
        }
        del hparams["lr"]
        del hparams["weight_decay"]
        
        hparams["task"] = "regression" if self.output_dim == 1 else "classification"
        hparams["loss_fn"] = "MSELoss" if self.output_dim == 1 else "CrossEntropyLoss"
        hparams["metric"] = self.metric.__name__
        hparams["metric_hparams"] = self.metric_hparams
        hparams["random_seed"] = self.args.random_seed
        
        return hparams
    
    def _set_embedding_config(self, hparams: Dict[str, Any]):
        if self.args.embedding == "identity":
            self._embedding_config = self._embedding_config_cls(input_dim=self.data.shape[1])
        elif self.args.embedding == "feature_tokenizer":
            required_token_dim = 2 if self.args.backbone == "transformer" else 1
            
            self._embedding_config = self._embedding_config_cls(input_dim=self.data.shape[1],
                                                        emb_dim = hparams["embedding_emb_dim"], 
                                                        cont_nums = self.data.shape[1] - len(self.category_cols), 
                                                        cat_dims = self.category_dims, 
                                                        required_token_dim=required_token_dim)

            del hparams["embedding_emb_dim"]

        return hparams
    
    def _set_backbone_config(self, hparams: Dict[str, Any]):
        backbone_hparams = {}
        for k, v in hparams.items():
            if "backbone_" in k:
                backbone_hparams[k.split("backbone_")[1]] = v
                
        if self.args.backbone == "mlp":
            print(backbone_hparams)
            self._backbone_config = self._backbone_config_cls(
                input_dim = self._embedding_config.output_dim,
                **backbone_hparams
            )
        elif self.args.backbone == "transformer":
            self._backbone_config = self._backbone_config_cls(
                d_model = self._embedding_config.emb_dim,
                **backbone_hparams
            )
        
        for k, v in backbone_hparams.items():
            del hparams["backbone_" + k]

        return hparams
    
    def benchmark(self):
        hparams = self.__tune_hyperparameters()

        config = self._get_config(hparams)
        pl_module = self.pl_module_class(config)

        pl_module = self.fit_model(pl_module, config)
        
        print("Evaluation %s: %.4f" % (self.metric.__name__, self.evaluate(pl_module, config, self.X_test, self.y_test)))
        