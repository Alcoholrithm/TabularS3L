    
def test_scarf_classification():
    from ts3l.pl_modules import SCARFLightning
    from ts3l.models import SCARF
    from ts3l.utils.scarf_utils import SCARFDataset
    from ts3l.utils import TS3LDataModule

    import torch.nn as nn

    import sys
    sys.path.append('.')
    
    from diabetes import load_diabetes

    data, label, continuous_cols, category_cols = load_diabetes()
    num_categoricals = len(category_cols)
    num_continuous = len(continuous_cols)
    loss_fn = "CrossEntropyLoss"
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

        train_ds = SCARFDataset(pd.concat([X_train, X_unlabeled]), corruption_rate=int(data_hparams["corruption_rate"] * X_train.shape[1]))
        test_ds = SCARFDataset(X_valid, corruption_rate=int(data_hparams["corruption_rate"] * X_train.shape[1]))
        
        pl_datamodule = TS3LDataModule(train_ds, test_ds, batch_size=batch_size, train_sampler="random")

        model.set_first_phase()

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

        model = SCARFLightning.load_from_checkpoint(pretraining_path)

        model.set_second_phase()
        
            
        train_ds = SCARFDataset(X_train, y_train.values)
        test_ds = SCARFDataset(X_valid, y_valid.values)

        pl_datamodule = TS3LDataModule(train_ds, test_ds, batch_size = batch_size, train_sampler="weighted")
            
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

        model = SCARFLightning.load_from_checkpoint(checkpoint_callback.best_model_path)
        model.set_second_phase()
        return model

    hparams_range = {
        
        'hidden_dim' : ['suggest_int', ['hidden_dim', 16, 512]],
        'encoder_depth' : ['suggest_int', ['encoder_depth', 2, 6]],
        'head_depth' : ['suggest_int', ['head_depth', 1, 3]],
        'corruption_rate' : ['suggest_float', ['corruption_rate', 0.1, 0.7]],
        'dropout_rate' : ['suggest_float', ['dropout_rate', 0.05, 0.3]],

        'lr' : ['suggest_float', ['lr', 0.0001, 0.05]],
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
                "hidden_dim" : None,
                "encoder_depth" : None,
                "head_depth" : None,
                'dropout_rate' : None,
                "output_dim" : 2,
            }
            data_hparams = {
                "corruption_rate" : None
            }
            optim_hparams = {
                "lr" : None
            }
            scheduler_hparams = {
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

            from ts3l.utils.scarf_utils import SCARFConfig
            config = SCARFConfig(
            task="classification",
            loss_fn=loss_fn, metric=metric, metric_hparams={},
            input_dim=model_hparams["input_dim"], output_dim=model_hparams["output_dim"],
            hidden_dim=model_hparams["hidden_dim"],
            encoder_depth=model_hparams["encoder_depth"], head_depth=model_hparams["head_depth"], 
            dropout_rate=model_hparams["dropout_rate"]
            )
            pl_scarf = SCARFLightning(config)

            pl_scarf = fit_model(pl_scarf, data_hparams)
            

            trainer = Trainer(
                        accelerator = accelerator,
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

def test_scarf_regression():
    from ts3l.pl_modules import SCARFLightning
    from ts3l.models import SCARF
    from ts3l.utils.scarf_utils import SCARFDataset
    from ts3l.utils import TS3LDataModule

    import torch.nn as nn

    import sys
    sys.path.append('.')
    
    from abalone import load_abalone

    data, label, continuous_cols, category_cols = load_abalone()
    num_categoricals = len(category_cols)
    num_continuous = len(continuous_cols)
    loss_fn = "MSELoss"
    metric =  "mean_squared_error"
    metric_params = {}
    random_seed = 0


    from sklearn.model_selection import train_test_split

    X_train, X_valid, y_train, y_valid = train_test_split(data, label, train_size = 0.7, random_state=random_seed)

    X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=random_seed)


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

        train_ds = SCARFDataset(pd.concat([X_train, X_unlabeled]), corruption_rate=int(data_hparams["corruption_rate"] * X_train.shape[1]))
        test_ds = SCARFDataset(X_valid, corruption_rate=int(data_hparams["corruption_rate"] * X_train.shape[1]))
        
        pl_datamodule = TS3LDataModule(train_ds, test_ds, batch_size=batch_size, train_sampler="random")

        model.set_first_phase()

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

        model = SCARFLightning.load_from_checkpoint(pretraining_path)

        model.set_second_phase()
        
            
        train_ds = SCARFDataset(X_train, y_train.values, is_regression=True)
        test_ds = SCARFDataset(X_valid, y_valid.values, is_regression=True)

        pl_datamodule = TS3LDataModule(train_ds, test_ds, batch_size = batch_size, train_sampler="random")
            
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

        model = SCARFLightning.load_from_checkpoint(checkpoint_callback.best_model_path)
        model.set_second_phase()
        return model

    hparams_range = {
        
        'hidden_dim' : ['suggest_int', ['hidden_dim', 16, 512]],
        'encoder_depth' : ['suggest_int', ['encoder_depth', 2, 6]],
        'head_depth' : ['suggest_int', ['head_depth', 1, 3]],
        'corruption_rate' : ['suggest_float', ['corruption_rate', 0.1, 0.7]],
        'dropout_rate' : ['suggest_float', ['dropout_rate', 0.05, 0.3]],

        'lr' : ['suggest_float', ['lr', 0.0001, 0.05]],
    }

    import optuna
    from sklearn.metrics import mean_squared_error

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
                "hidden_dim" : None,
                "encoder_depth" : None,
                "head_depth" : None,
                'dropout_rate' : None,
                "output_dim" : 1,
            }
            data_hparams = {
                "corruption_rate" : None
            }
            optim_hparams = {
                "lr" : None
            }
            scheduler_hparams = {
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

            from ts3l.utils.scarf_utils import SCARFConfig
            config = SCARFConfig(
            task="regression",
            loss_fn=loss_fn, metric=metric, metric_hparams={},
            input_dim=model_hparams["input_dim"], output_dim=model_hparams["output_dim"],
            hidden_dim=model_hparams["hidden_dim"],
            encoder_depth=model_hparams["encoder_depth"], head_depth=model_hparams["head_depth"], 
            dropout_rate=model_hparams["dropout_rate"]
            )
            pl_scarf = SCARFLightning(config)
            pl_scarf = fit_model(pl_scarf, data_hparams)
            

            trainer = Trainer(
                        accelerator = accelerator,
                        max_epochs = max_epochs,
                        num_sanity_val_steps = 2,
                        callbacks = None,
            )

            test_ds = SCARFDataset(X_valid)
            from torch.utils.data import SequentialSampler, DataLoader
            import torch
            test_dl = DataLoader(test_ds, batch_size, shuffle=False, sampler = SequentialSampler(test_ds), num_workers=n_jobs)

            preds = trainer.predict(pl_scarf, test_dl)

            preds = torch.concat([out.cpu() for out in preds]).squeeze()
            
            mse = mean_squared_error(y_valid, preds)

            return mse
        
    study = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=2, show_progress_bar=False)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")


    trial = study.best_trial

    print("  MSE: {}".format(trial.value))
    print("  Best hyperparameters: ", trial)
    
if __name__ == "__main__":
    test_scarf_classification()
    test_scarf_regression()