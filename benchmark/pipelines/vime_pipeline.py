import argparse
import pandas as pd
from typing import List, Type, Dict, Any
from ts3l.utils import BaseConfig
from ts3l.pl_modules.base_module import TS3LLightining

from ts3l.pl_modules import VIMELightning
from ts3l.utils.vime_utils import VIMEConfig, VIMEDataset, VIMESecondPhaseCollateFN
from ts3l.utils import TS3LDataModule

from hparams_range.vime import hparams_range

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from .pipeline import PipeLine

from torch.nn import functional as F
from torch.utils.data import SequentialSampler, DataLoader
import torch

from copy import deepcopy

class VIMEPipeLine(PipeLine):
    
    def __init__(self, args: argparse.Namespace, data: pd.DataFrame, label: pd.Series, continuous_cols: List[str], category_cols: List[str], output_dim: int, metric: str, metric_hparams: Dict[str, Any] = {}):
        super().__init__(args, data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams)
        
    def initialize(self):
        self.config_class = VIMEConfig
        self.pl_module_class = VIMELightning
        self.hparams_range = deepcopy(hparams_range)
        
        super().initialize()
    
    def _get_config(self, hparams: Dict[str, Any]):
        hparams = super()._get_config(hparams)
        
        hparams["num_continuous"] = len(self.continuous_cols)
        hparams["cat_cardinality"] = self.cat_cardinality
        
        return self.config_class(embedding_config=self._embedding_config, backbone_config=self._backbone_config, output_dim = self.output_dim, **hparams)
    
    def fit_model(self, pl_module: TS3LLightining, config: Type[BaseConfig]):
        train_ds = VIMEDataset(X = self.X_train, unlabeled_data = self.X_unlabeled, config=config, continuous_cols = self.continuous_cols, category_cols = self.category_cols)
        test_ds = VIMEDataset(X = self.X_valid, config=config, continuous_cols = self.continuous_cols, category_cols = self.category_cols)
        
        pl_datamodule = TS3LDataModule(train_ds, test_ds, self.args.batch_size, train_sampler='random', n_jobs = self.args.n_jobs)

        pl_module.set_first_phase()

        callbacks = [
            EarlyStopping(
                monitor= 'val_loss', 
                mode = 'min',
                patience = self.args.first_phase_patience,
                verbose = False
            )
        ]
        pretraining_path = f'benchmark_ckpt/pretraining'
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=pretraining_path,
            filename='pretraining-{epoch:02d}-{val_f1:.4f}',
            save_top_k=1,
            mode = 'min'
        )

        callbacks.append(checkpoint_callback)

        trainer = Trainer(
                        accelerator = self.args.accelerator,
                        devices=self.args.devices,
                        max_epochs = self.args.max_epochs,
                        num_sanity_val_steps = 2,
                        callbacks = callbacks,
        )

        trainer.fit(pl_module, pl_datamodule)
        
        pretraining_path = checkpoint_callback.best_model_path

        pl_module = self.pl_module_class.load_from_checkpoint(pretraining_path)

        pl_module.set_second_phase()
        
        train_ds = VIMEDataset(X = self.X_train, Y = self.y_train.values, config = config, unlabeled_data=self.X_unlabeled, continuous_cols=self.continuous_cols, category_cols=self.category_cols, is_second_phase=True, is_regression=True if self.output_dim==1 else False)
        test_ds = VIMEDataset(X = self.X_valid, Y = self.y_valid.values, config = config, continuous_cols=self.continuous_cols, category_cols=self.category_cols, is_second_phase=True, is_regression=True if self.output_dim==1 else False)
        
        pl_datamodule = TS3LDataModule(train_ds, test_ds, batch_size = self.args.batch_size, train_sampler="random" if self.output_dim == 1 else "weighted", train_collate_fn=VIMESecondPhaseCollateFN())
            
        callbacks = [
            EarlyStopping(
                monitor= 'val_' + self.metric.__name__, 
                mode = 'max',
                patience = self.args.second_phase_patience,
                verbose = False
            )
        ]

        checkpoint_path = f'benchmark_ckpt/'
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_' + self.metric.__name__,
            dirpath=checkpoint_path,
            filename='{epoch:02d}-{val_f1:.4f}',
            save_top_k=1,
            mode = "max" if self.direction == "maximize" else "min"
        )

        callbacks.append(checkpoint_callback)

        trainer = Trainer(
                        accelerator = self.args.accelerator,
                        devices=self.args.devices,
                        max_epochs = self.args.max_epochs,
                        num_sanity_val_steps = 2,
                        callbacks = callbacks,
        )

        trainer.fit(pl_module, pl_datamodule)

        pl_module = self.pl_module_class.load_from_checkpoint(checkpoint_callback.best_model_path)
        pl_module.set_second_phase()
        
        return pl_module
        
    def evaluate(self, pl_module: TS3LLightining, config: Type[BaseConfig], X: pd.DataFrame, y: pd.Series):
        
        pl_module.set_second_phase()

        trainer = Trainer(
                    accelerator = self.args.accelerator,
                    devices = self.args.devices,
                    max_epochs = self.args.max_epochs,
                    num_sanity_val_steps = 2,
                    callbacks = None,
        )

        test_ds = VIMEDataset(X = X, category_cols=self.category_cols, continuous_cols=self.continuous_cols, is_second_phase=True, is_regression=True if self.output_dim==1 else False)
        test_dl = DataLoader(test_ds, self.args.batch_size, shuffle=False, sampler = SequentialSampler(test_ds), num_workers=self.args.n_jobs)

        preds = trainer.predict(pl_module, test_dl)
        
        if self.output_dim > 1:
            preds = F.softmax(torch.concat([out.cpu() for out in preds]).squeeze(),dim=1)
        else:
            preds = torch.concat([out.cpu() for out in preds]).squeeze()
            
        score = self.metric(preds, y)
        
        return score