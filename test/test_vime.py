
def test_vime_classification():
    from ts3l.pl_modules import VIMELightning
    from ts3l.utils.vime_utils import VIMESemiSLCollateFN
    from ts3l.utils.vime_utils import VIMEDataset
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
        
        train_ds = VIMEDataset(X = X_train, unlabeled_data = X_unlabeled, data_hparams=data_hparams, continous_cols = continuous_cols, category_cols = category_cols)
        test_ds = VIMEDataset(X = X_valid, data_hparams=data_hparams, continous_cols = continuous_cols, category_cols = category_cols)
        
        pl_datamodule = TS3LDataModule(train_ds, test_ds, batch_size, train_sampler='random', n_jobs = n_jobs)

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

        model = VIMELightning.load_from_checkpoint(pretraining_path)

        model.set_second_phase()
        
        train_ds = VIMEDataset(X_train, y_train.values, data_hparams, unlabeled_data=X_unlabeled, continous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
        test_ds = VIMEDataset(X_valid, y_valid.values, data_hparams, continous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
        
        pl_datamodule = TS3LDataModule(train_ds, test_ds, batch_size = batch_size, train_sampler="weighted", train_collate_fn=VIMESemiSLCollateFN())
            
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
        model.set_second_phase()
        
        return model

    hparams_range = {
        
    'hidden_dim' : ['suggest_int', ['hidden_dim', 16, 512]],
    
    'p_m' : ["suggest_float", ["p_m", 0.1, 0.9]],
    'alpha1' : ["suggest_float", ["alpha1", 0.1, 5]],
    'alpha2' : ["suggest_float", ["alpha2", 0.1, 5]],
    'beta' : ["suggest_float", ["beta", 0.1, 10]],
    'K' : ["suggest_int", ["K", 2, 20]],


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
                "output_dim" : 2,
                'alpha1' : None,
                'alpha2' : None,
                'beta' : None,
                'K' : None,
                "num_categoricals" : num_categoricals, 
                "num_continuous": num_continuous, 
                "u_label" : -1,
            }
            
            data_hparams = {
                "K" : None,
                "p_m" : None
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

            from ts3l.utils.vime_utils import VIMEConfig
            config = VIMEConfig(
            task="classification",
            loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
            input_dim=model_hparams["input_dim"], hidden_dim=model_hparams["hidden_dim"],
            output_dim=model_hparams["output_dim"],
            alpha1=model_hparams["alpha1"], alpha2=model_hparams["alpha2"], beta=model_hparams["beta"], K=model_hparams["K"],
            num_categoricals=num_categoricals, num_continuous=num_continuous
            )
            pl_vime = VIMELightning(config)

            pl_vime = fit_model(pl_vime, data_hparams)
            pl_vime.set_second_phase()

            trainer = Trainer(
                        accelerator = accelerator,
                        max_epochs = max_epochs,
                        num_sanity_val_steps = 2,
                        callbacks = None,
            )

            test_ds = VIMEDataset(X_valid, category_cols=category_cols, continous_cols=continuous_cols, is_second_phase=True)
            
            from torch.utils.data import SequentialSampler, DataLoader
            import torch
            test_dl = DataLoader(test_ds, batch_size, shuffle=False, sampler = SequentialSampler(test_ds), num_workers=n_jobs)

            preds = trainer.predict(pl_vime, test_dl)
            
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
    

def test_vime_regression():
    from ts3l.pl_modules import VIMELightning
    from ts3l.utils.vime_utils import VIMESemiSLCollateFN
    from ts3l.utils.vime_utils import VIMEDataset
    from ts3l.utils import TS3LDataModule

    import torch.nn as nn

    import sys
    sys.path.append('.')
    
    from abalone import load_abalone

    data, label, continuous_cols, category_cols = load_abalone()
    num_categoricals = len(category_cols)
    num_continuous = len(continuous_cols)
    loss_fn = nn.MSELoss
    metric =  "mean_squared_error"
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
        
        train_ds = VIMEDataset(X = X_train, unlabeled_data = X_unlabeled, data_hparams=data_hparams, continous_cols = continuous_cols, category_cols = category_cols)
        test_ds = VIMEDataset(X = X_valid, data_hparams=data_hparams, continous_cols = continuous_cols, category_cols = category_cols)
        
        pl_datamodule = TS3LDataModule(train_ds, test_ds, batch_size, train_sampler='random', n_jobs = n_jobs)

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

        model = VIMELightning.load_from_checkpoint(pretraining_path)

        model.set_second_phase()
        
        train_ds = VIMEDataset(X_train, y_train.values, data_hparams, unlabeled_data=X_unlabeled, continous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True, is_regression=True)
        test_ds = VIMEDataset(X_valid, y_valid.values, data_hparams, continous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True, is_regression=True)

        pl_datamodule = TS3LDataModule(train_ds, test_ds, batch_size = batch_size, train_sampler="weighted", train_collate_fn=VIMESemiSLCollateFN())
            
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
        model.set_second_phase()
        
        return model

    hparams_range = {
        
    'hidden_dim' : ['suggest_int', ['hidden_dim', 16, 512]],
    
    'p_m' : ["suggest_float", ["p_m", 0.1, 0.9]],
    'alpha1' : ["suggest_float", ["alpha1", 0.1, 5]],
    'alpha2' : ["suggest_float", ["alpha2", 0.1, 5]],
    'beta' : ["suggest_float", ["beta", 0.1, 10]],
    'K' : ["suggest_int", ["K", 2, 20]],


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
                "output_dim" : 1,
                'alpha1' : None,
                'alpha2' : None,
                'beta' : None,
                'K' : None,
                "num_categoricals" : num_categoricals, 
                "num_continuous": num_continuous, 
                "u_label" : -1,
            }
            
            data_hparams = {
                "K" : None,
                "p_m" : None
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
            from ts3l.utils.vime_utils import VIMEConfig
            config = VIMEConfig(
            task="regression",
            loss_fn="MSELoss", metric=metric, metric_hparams={},
            input_dim=model_hparams["input_dim"], hidden_dim=model_hparams["hidden_dim"],
            output_dim=model_hparams["output_dim"],
            alpha1=model_hparams["alpha1"], alpha2=model_hparams["alpha2"], beta=model_hparams["beta"], K=model_hparams["K"],
            num_categoricals=num_categoricals, num_continuous=num_continuous
            )
            pl_vime = VIMELightning(config)

            pl_vime = fit_model(pl_vime, data_hparams)
            pl_vime.set_second_phase()

            trainer = Trainer(
                        accelerator = accelerator,
                        max_epochs = max_epochs,
                        num_sanity_val_steps = 2,
                        callbacks = None,
            )
            
            test_ds = VIMEDataset(X_valid, category_cols=category_cols, continous_cols=continuous_cols, is_second_phase=True, is_regression=True)
            from torch.utils.data import SequentialSampler, DataLoader
            import torch
            test_dl = DataLoader(test_ds, batch_size, shuffle=False, sampler = SequentialSampler(test_ds), num_workers=n_jobs)

            preds = trainer.predict(pl_vime, test_dl)
            
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
    test_vime_classification()
    test_vime_regression()