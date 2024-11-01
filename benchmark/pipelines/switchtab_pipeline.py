import argparse
import pandas as pd
from typing import List, Type, Dict, Any
from ts3l.utils import BaseConfig
from ts3l.utils.embedding_utils import FTEmbeddingConfig
from ts3l.utils.backbone_utils import TransformerBackboneConfig
from ts3l.pl_modules.base_module import TS3LLightining

from ts3l.pl_modules import SwitchTabLightning
from ts3l.utils.switchtab_utils import SwitchTabConfig, SwitchTabDataset, SwitchTabFirstPhaseCollateFN
from ts3l.utils import TS3LDataModule
from ts3l.utils.misc import get_category_dims

from hparams_range.switchtab import hparams_range

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from .pipeline import PipeLine

from torch.nn import functional as F
from torch.utils.data import SequentialSampler, DataLoader
import torch

class SwitchTabPipeLine(PipeLine):
    
    def __init__(self, args: argparse.Namespace, data: pd.DataFrame, label: pd.Series, continuous_cols: List[str], category_cols: List[str], output_dim: int, metric: str, metric_hparams: Dict[str, Any] = {}):
        self.category_dims = get_category_dims(data, category_cols)
        super().__init__(args, data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams)
        
        
    def initialize(self):
        self.config_class = SwitchTabConfig
        self.pl_module_class = SwitchTabLightning
        self.hparams_range = hparams_range
        
    
    def _get_config(self, hparams: Dict[str, Any]):
        hparams = super()._get_config(hparams)
        hparams["hidden_dim"] = hparams["encoder_head_dim"] * hparams["n_head"]
        del hparams["encoder_head_dim"]
        
        hparams["category_dims"] = self.category_dims
                
        self._embedding_config = FTEmbeddingConfig(input_dim=self.data.shape[1],
            cont_nums = self.data.shape[1] - len(self.category_cols),
            cat_dims = self.category_dims,
            required_token_dim=2
        )

        self._backbone_config = TransformerBackboneConfig(d_model = self._embedding_config.emb_dim, ffn_factor=hparams["ffn_factor"], hidden_dim=hparams["hidden_dim"], encoder_depth=hparams["encoder_depth"], n_head=hparams["n_head"])
        return self.config_class(embedding_config=self._embedding_config, backbone_config=self._backbone_config, output_dim = self.output_dim, **hparams)
    
    def fit_model(self, pl_module: TS3LLightining, config: Type[BaseConfig]):
        
        train_ds = SwitchTabDataset(X = self.X_train, unlabeled_data = self.X_unlabeled, Y = self.y_train.values, config=config, continuous_cols=self.continuous_cols, category_cols=self.category_cols, is_regression=True if self.output_dim == 1 else False)
        valid_ds = SwitchTabDataset(X = self.X_valid, Y = self.y_valid.values, config=config, continuous_cols=self.continuous_cols, category_cols=self.category_cols, is_regression=True if self.output_dim == 1 else False)

        pl_datamodule = TS3LDataModule(train_ds, valid_ds, self.args.batch_size, train_sampler="random" if self.output_dim == 1 else "weighted", train_collate_fn=SwitchTabFirstPhaseCollateFN(), valid_collate_fn=SwitchTabFirstPhaseCollateFN())

        pl_module.set_first_phase()

        callbacks = [
            EarlyStopping(
                monitor= 'val_loss', 
                mode = 'min',
                patience = self.args.first_phase_patience,
                verbose = False
            )
        ]
        pretraining_path = f'benchmark_ckpt/' + self.args.data + '/pretraining/'
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
        
        train_ds = SwitchTabDataset(X = self.X_train, Y = self.y_train.values, continuous_cols=self.continuous_cols, category_cols=self.category_cols, is_second_phase=True, is_regression=True if self.output_dim == 1 else False)
        valid_ds = SwitchTabDataset(X = self.X_valid, Y = self.y_valid.values, continuous_cols=self.continuous_cols, category_cols=self.category_cols, is_second_phase=True, is_regression=True if self.output_dim == 1 else False)
                
        pl_datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = self.args.batch_size, train_sampler="random" if self.output_dim == 1 else "weighted")

        callbacks = [
            EarlyStopping(
                monitor= 'val_' + self.metric.__name__, 
                mode = 'max',
                patience = self.args.second_phase_patience,
                verbose = False
            )
        ]

        checkpoint_path = None

        checkpoint_path = f'benchmark_ckpt/' + self.args.data + '/'
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

        test_ds = SwitchTabDataset(X, continuous_cols=self.continuous_cols, category_cols=self.category_cols, is_second_phase=True)
        test_dl = DataLoader(test_ds, self.args.batch_size, shuffle=False, sampler = SequentialSampler(test_ds), num_workers=self.args.n_jobs)

        preds = trainer.predict(pl_module, test_dl)
        
        if self.output_dim > 1:
            preds = F.softmax(torch.concat([out.cpu() for out in preds]).squeeze(),dim=1)
        else:
            preds = torch.concat([out.cpu() for out in preds]).squeeze()
            
        score = self.metric(preds, y)
        
        return score