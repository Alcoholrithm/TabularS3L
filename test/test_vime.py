from ts3l.utils.misc import BaseScorer
class AccuracyScorer(BaseScorer):
    def __init__(self, metric: str) -> None:
        super().__init__(metric)
    
    def __call__(self, y, y_hat) -> float:
        return self.metric(y, y_hat.argmax(1))

def test_vime_classification():
    from ts3l.pl_modules import VIMELightning
    from ts3l.utils.vime_utils import VIMESelfDataset, VIMESemiDataset, VIMECollateFN
    from ts3l.utils import TS3LDataModule

    import torch.nn as nn

    import sys
    sys.path.append("/home/runner/work/TabularS3L/TabularS3L/test")
    from diabetes import load_diabetes

    data, label, continuous_cols, category_cols = load_diabetes()
    num_categoricals = len(continuous_cols)
    num_continuous = len(continuous_cols)
    loss_fn = nn.CrossEntropyLoss
    metric =  "accuracy_score"
    metric_params = {}
    random_seed = 0


    from sklearn.model_selection import train_test_split

    X_train, X_valid, y_train, y_valid = train_test_split(data, label, train_size = 0.7, random_state=random_seed, stratify=label)

    X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=random_seed, stratify=y_train)


    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    import pandas as pd

    accelerator = 'cpu'
    n_jobs = 4
    max_epochs = 3
    batch_size = 128

    pretraining_patience = 3
    early_stopping_patience = 3

    batch_size = 64

    def fit_model(
                model,
                data_hparams
        ):
        
        train_ds = VIMESelfDataset(pd.concat([X_train, X_unlabeled]), data_hparams, continuous_cols, category_cols)
        test_ds = VIMESelfDataset(X_valid, data_hparams, continuous_cols, category_cols)
        
        pl_datamodule = TS3LDataModule(train_ds, test_ds, batch_size, train_sampler='random', n_jobs = n_jobs)

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
                        accelerator = accelerator,
                        max_epochs = max_epochs,
                        num_sanity_val_steps = 2,
                        callbacks = callbacks,
        )

        trainer.fit(model, pl_datamodule)
        
        pretraining_path = checkpoint_callback.best_model_path

        model = VIMELightning.load_from_checkpoint(pretraining_path)

        model.do_finetunning()
        
        train_ds = VIMESemiDataset(X_train, y_train.values, data_hparams, unlabeled_data=X_unlabeled, continous_cols=continuous_cols, category_cols=category_cols)
        test_ds = VIMESemiDataset(X_valid, y_valid.values, data_hparams, continous_cols=continuous_cols, category_cols=category_cols)

        pl_datamodule = TS3LDataModule(train_ds, test_ds, batch_size = batch_size, train_sampler="weighted", train_collate_fn=VIMECollateFN())
            
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
                        accelerator = accelerator,
                        max_epochs = max_epochs,
                        num_sanity_val_steps = 2,
                        callbacks = callbacks,
        )

        trainer.fit(model, pl_datamodule)

        model = VIMELightning.load_from_checkpoint(checkpoint_callback.best_model_path)
        
        return model

    hparams_range = {
        
    'predictor_hidden_dim' : ['suggest_int', ['predictor_hidden_dim', 16, 512]],
    # 'predictor_output_dim' : ['suggest_int', ['emb_dim', 16, 512]],
    
    'p_m' : ["suggest_float", ["p_m", 0.1, 0.9]],
    'alpha1' : ["suggest_float", ["alpha1", 0.1, 5]],
    'alpha2' : ["suggest_float", ["alpha2", 0.1, 5]],
    'beta' : ["suggest_float", ["beta", 0.1, 10]],
    'K' : ["suggest_int", ["K", 2, 20]],


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
                "encoder_dim" : data.shape[1],
                "predictor_hidden_dim" : None,
                "predictor_output_dim" : 2,
                'alpha1' : None,
                'alpha2' : None,
                'beta' : None,
                'K' : None
            }
            
            data_hparams = {
                "K" : None,
                "p_m" : None
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

            pl_subtab = VIMELightning(
                    model_hparams,
                    "Adam", optim_hparams, "StepLR", scheduler_hparams,
                    num_categoricals, num_continuous, -1,
                    loss_fn,
                    {},
                    AccuracyScorer("accuracy_score"),
                    random_seed)

            pl_subtab = fit_model(pl_subtab, data_hparams)
            pl_subtab.do_finetunning()

            trainer = Trainer(
                        accelerator = accelerator,
                        max_epochs = max_epochs,
                        num_sanity_val_steps = 2,
                        callbacks = None,
            )

            test_ds = VIMESemiDataset(X_valid, category_cols=category_cols, continous_cols=continuous_cols, is_test = True)
            from torch.utils.data import SequentialSampler, DataLoader
            import torch
            test_dl = DataLoader(test_ds, batch_size, shuffle=False, sampler = SequentialSampler(test_ds), num_workers=n_jobs)

            preds = trainer.predict(pl_subtab, test_dl)
            
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
    

if __name__ == "__main__":
    test_vime_classification()