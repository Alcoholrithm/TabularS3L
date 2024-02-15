from ts3l.pl_modules import SCARFLightning
from ts3l.models import SCARF
from ts3l.utils.scarf_utils import SCARFDataset
from ts3l.utils import TabularS3LDataModule

import torch.nn as nn

from .diabetes import load_diabetes

data, label, continuous_cols, category_cols = load_diabetes()
num_categoricals = len(continuous_cols)
num_continuous = len(continuous_cols)
loss_fn = nn.CrossEntropyLoss
metric =  "accuracy_score"
metric_params = {}
random_seed = 0

from ts3l.utils.misc import BaseScorer


class AccuracyScorer(BaseScorer):
    def __init__(self, metric: str) -> None:
        super().__init__(metric)
    
    def __call__(self, y, y_hat) -> float:
        return self.metric(y, y_hat.argmax(1))

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(data, label, train_size = 0.7, random_state=random_seed, stratify=label)

X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=random_seed, stratify=y_train)


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

gpus = []
n_jobs = 4
max_epochs = 10
batch_size = 128

pretraining_patience = 3
early_stopping_patience = 10

batch_size = 512

def fit_model(
            model,
            data_hparams
    ):

    train_ds = SCARFDataset(X_train.append(X_unlabeled), corruption_len=int(data_hparams["corruption_rate"] * X_train.shape[1]))
    test_ds = SCARFDataset(X_valid, corruption_len=int(data_hparams["corruption_rate"] * X_train.shape[1]))
    
    pl_datamodule = TabularS3LDataModule(train_ds, test_ds, batch_size=batch_size, train_sampler="random")

    model.do_pretraining()

    callbacks = [
        EarlyStopping(
            monitor= 'val_loss', 
            mode = 'min',
            patience = pretraining_patience,
            verbose = False
        )
    ]
    pretraining_path = f'temporary_ckpt_data/pretraining'
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=pretraining_path,
        filename='pretraining-{epoch:02d}-{val_f1:.4f}',
        save_top_k=1,
        mode = 'min'
    )

    callbacks.append(checkpoint_callback)

    trainer = Trainer(
                    devices = gpus,
                    accelerator = "cuda" if len(gpus) >= 1 else 'cpu',
                    max_epochs = max_epochs,
                    num_sanity_val_steps = 2,
                    callbacks = callbacks,
    )

    trainer.fit(model, pl_datamodule)
    
    pretraining_path = checkpoint_callback.best_model_path

    model = model.load_from_checkpoint(pretraining_path)

    model.do_finetunning()
    
        
    train_ds = SCARFDataset(X_train, y_train.values)
    test_ds = SCARFDataset(X_valid, y_valid.values)

    pl_datamodule = TabularS3LDataModule(train_ds, test_ds, batch_size = batch_size, train_sampler="weighted")
        
    callbacks = [
        EarlyStopping(
            monitor= 'val_' + metric, 
            mode = 'max',
            patience = early_stopping_patience,
            verbose = False
        )
    ]

    checkpoint_path = None

    checkpoint_path = f'temporary_ckpt_data/'
    checkpoint_callback = ModelCheckpoint(
        monitor='val_' + metric,
        dirpath=checkpoint_path,
        filename='{epoch:02d}-{val_f1:.4f}',
        save_top_k=1,
        mode = 'max'
    )

    callbacks.append(checkpoint_callback)

    trainer = Trainer(
                    devices = gpus,
                    accelerator = "cuda" if len(gpus) >= 1 else 'cpu',
                    max_epochs = max_epochs,
                    num_sanity_val_steps = 2,
                    callbacks = callbacks,
    )

    trainer.fit(model, pl_datamodule)

    model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    return model

hparams_range = {
    
    'emb_dim' : ['suggest_int', ['emb_dim', 16, 512]],
    'encoder_depth' : ['suggest_int', ['encoder_depth', 2, 6]],
    'head_depth' : ['suggest_int', ['head_depth', 1, 3]],
    'corruption_rate' : ['suggest_float', ['corruption_rate', 0.1, 0.7]],
    'dropout_rate' : ['suggest_float', ['dropout_rate', 0.05, 0.3]],

    'lr' : ['suggest_float', ['lr', 0.0001, 0.05]],
    'gamma' : ['suggest_float', ['gamma', 0.1, 0.95]],
    'step_size' : ['suggest_int', ['step_size', 10, 100]],
}

import optuna
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

def objective(      trial: optuna.trial.Trial,
        ) -> float:
        """Objective function for optuna

        Args:
            trial: A object which returns hyperparameters of a model of hyperparameter search trial.
            train_idx: Indices of training data in self.data and self.label.
            test_idx: Indices of test data in self.data and self.label.
            fold_idx: A fold index that denotes which fold under the given k-fold cross validation.
        
        Returns:
            A score of given hyperparameters.
        """
        model_hparams = {
            "input_dim" : data.shape[1],
            "emb_dim" : None,
            "encoder_depth" : None,
            "head_depth" : None,
            'dropout_rate' : None,
            "out_dim" : 2,
            # "sampling_candidate" : X_train.values
            # "features_low" : data.min().values,
            # "features_high" : data.max().values,
        }
        data_hparams = {
            "corruption_rate" : None
        }
        optim_hparams = {
            "lr" : None
        }
        scheduler_hparams = {
            'gamma' : None,
            'step_size' : None
        }

        for k, v in hparams_range.items():
            if k in model_hparams.keys():
                model_hparams[k] = getattr(trial, v[0])(*v[1])
            if k in data_hparams.keys():
                data_hparams[k] = getattr(trial, v[0])(*v[1])
            if k in optim_hparams.keys():
                optim_hparams[k] = getattr(trial, v[0])(*v[1])
            if k in scheduler_hparams.keys():
                scheduler_hparams[k] = getattr(trial, v[0])(*v[1])

        pl_scarf = SCARFLightning(
                model_hparams,
                 "Adam", optim_hparams, "StepLR", scheduler_hparams,
                 loss_fn,
                 {},
                 AccuracyScorer("accuracy_score"),
                 random_seed)
        
        pl_scarf = fit_model(pl_scarf, data_hparams)
        

        trainer = Trainer(
                    devices = gpus,
                    accelerator = "cuda" if len(gpus)>= 1 else 'cpu',
                    max_epochs = max_epochs,
                    num_sanity_val_steps = 2,
                    callbacks = None,
        )

        test_ds = SCARFDataset(X_valid)
        from torch.utils.data import SequentialSampler, DataLoader
        import torch
        test_dl = DataLoader(test_ds, batch_size, shuffle=False, sampler = SequentialSampler(test_ds), num_workers=n_jobs)

        preds = trainer.predict(pl_scarf, test_dl)

        preds = F.softmax(torch.concat([out.cpu() for out in preds]).squeeze(),dim=1)

        accuracy = accuracy_score(y_valid, preds.argmax(1))

        return accuracy
    
study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=random_seed))
study.optimize(objective, n_trials=2, show_progress_bar=False)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")


trial = study.best_trial

print("  Accuracy: {}".format(trial.value))
print("  Best hyperparameters: ", trial)