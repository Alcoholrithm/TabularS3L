# TabularS3L

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Available Models with Quick Start Guides**](#available-models-with-quick-start)
| [**To DO**](#to-do)
| [**Contributing**](#contributing)

[![DOI](https://zenodo.org/badge/756740921.svg)](https://zenodo.org/doi/10.5281/zenodo.10776537)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TabularS3L is a PyTorch-based library designed to facilitate self- and semi-supervised learning with tabular data. While numerous self- and semi-supervised learning tabular models have been proposed, there lacks a comprehensive library catering to the needs of tabular practitioners. This library aims to address this gap by offering a unified PyTorch Lightning-based framework for studying and deploying such models.

## Installation
We provide a Python package ts3l of TabularS3L for users who want to use semi- and self-supervised learning tabular models.

```sh
pip install ts3l
```

## Available Models with Quick Start

TabularS3L employs a two-phase learning approach, where the learning strategies differ between phases. Below is an overview of the models available within TabularS3L, highlighting the learning strategies employed in each phase. The abbreviations 'Self-SL', 'Semi-SL', and 'SL' represent self-supervised learning, semi-supervised learning, and supervised learning, respectively.

| Model | First Phase | Second Phase |
|:---:|:---:|:---:|
| **VIME** ([NeurIPS'20](https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html)) | Self-SL | Semi-SL or SL |
| **SubTab** ([NeurIPS'21](https://proceedings.neurips.cc/paper/2021/hash/9c8661befae6dbcd08304dbf4dcaf0db-Abstract.html)) | Self-SL | SL |
| **SCARF** ([ICLR'22](https://iclr.cc/virtual/2022/spotlight/6297))| Self-SL | SL |

#### VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain
VIME enhances tabular data learning through a dual approach. In its first phase, it utilize a pretext task of estimating mask vectors from corrupted tabular data, alongside a reconstruction pretext task for self-supervised learning. The second phase leverages consistency regularization on unlabeled data.

<details close>
  <summary>Quick Start</summary>
  
  ```python
  # Assume that we have X_train, X_valid, X_test, y_train, y_valid, y_test, category_cols, and continuous_cols

  # Prepare the VIMELightning Module
  from ts3l.pl_modules import VIMELightning
  from ts3l.utils.vime_utils import VIMEDataset
  from ts3l.utils import TS3LDataModule
  from ts3l.utils.vime_utils import VIMEConfig
  from pytorch_lightning import Trainer

  metric = "accuracy_score"
  input_dim = X_train.shape[1]
  hidden_dim = 1024
  output_dim = 2
  alpha1 = 2.0
  alpha2 = 2.0
  beta = 1.0
  K = 3
  p_m = 0.2

  data_hparams = {
              "K" : K,
              "p_m" : p_m
          }

  batch_size = 128

  X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=0, stratify=y_train)

  config = VIMEConfig( task="classification", loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
  input_dim=input_dim, hidden_dim=hidden_dim,
  output_dim=output_dim, alpha1=alpha1, alpha2=alpha2, 
  beta=beta, K=K,
  num_categoricals=len(category_cols), num_continuous=len(continuous_cols)
  )

  pl_vime = VIMELightning(config)

  ### First Phase Learning
  train_ds = VIMEDataset(X = X_train, unlabeled_data = X_unlabeled, data_hparams=data_hparams, continous_cols = continuous_cols, category_cols = category_cols)
  valid_ds = VIMEDataset(X = X_valid, data_hparams=data_hparams, continous_cols = continuous_cols, category_cols = category_cols)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size, train_sampler='random', n_jobs = 4)

  trainer = Trainer(
                      accelerator = 'cpu',
                      max_epochs = 10,
                      num_sanity_val_steps = 2,
      )

  trainer.fit(pl_vime, datamodule)

  ### Second Phase Learning
  from ts3l.utils.vime_utils import VIMESemiSLCollateFN

  pl_vime.set_second_phase()

  train_ds = VIMEDataset(X_train, y_train.values, data_hparams, unlabeled_data=X_unlabeled, continous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
  valid_ds = VIMEDataset(X_valid, y_valid.values, data_hparams, continous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
          
  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = batch_size, train_sampler="weighted", train_collate_fn=VIMESemiSLCollateFN())

  trainer.fit(pl_vime, datamodule)

  # Evaluation
  from sklearn.metrics import accuracy_score
  import torch
  from torch.nn import functional as F
  from torch.utils.data import DataLoader, SequentialSampler

  test_ds = VIMEDataset(X_test, category_cols=category_cols, continous_cols=continuous_cols, is_second_phase=True)
  test_dl = DataLoader(test_ds, batch_size, shuffle=False, sampler = SequentialSampler(test_ds))

  preds = trainer.predict(pl_vime, test_dl)
          
  preds = F.softmax(torch.concat([out.cpu() for out in preds]).squeeze(),dim=1)

  accuracy = accuracy_score(y_test, preds.argmax(1))

  print("Accuracy %.2f" % accuracy)
  ```

</details>


#### SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning
SubTab turns the task of learning from tabular data into as a multi-view representation challenge by dividing input features into multiple subsets during its first phase. During the second phase, collaborative inference is used to derive a joint representation by aggregating latent variables across subsets. This approach improves the model's performance in supervised learning tasks.

<details close>
  <summary>Quick Start</summary>
  
  ```python
  # Assume that we have X_train, X_valid, X_test, y_train, y_valid, y_test, categorical_cols, and continuous_cols

  # Prepare the SubTabLightning Module
  from ts3l.pl_modules import SubTabLightning
  from ts3l.utils.subtab_utils import SubTabDataset, SubTabCollateFN
  from ts3l.utils import TS3LDataModule
  from ts3l.utils.subtab_utils import SubTabConfig
  from pytorch_lightning import Trainer

  metric = "accuracy_score"
  input_dim = X_train.shape[1]
  hidden_dim = 1024
  output_dim = 2
  tau = 1.0
  use_cosine_similarity = True
  use_contrastive = True
  use_distance = True
  n_subsets = 4
  overlap_ratio = 0.75

  mask_ratio = 0.1
  noise_type = "Swap"
  noise_level = 0.1

  data_hparams = {
              "n_subsets" : n_subsets,
              "overlap_ratio" : overlap_ratio,
              "mask_ratio" : mask_ratio,
              "noise_type" : noise_type,
              "noise_level" : noise_level,
              "n_column" : input_dim
          }

  batch_size = 128
  max_epochs = 3

  X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=0, stratify=y_train)

  config = SubTabConfig( task="classification", loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
  input_dim=input_dim, hidden_dim=hidden_dim,
  output_dim=output_dim, tau=tau, use_cosine_similarity= use_cosine_similarity, use_contrastive=use_contrastive, use_distance=use_distance, 
  n_subsets=n_subsets, overlap_ratio=overlap_ratio
  )

  pl_subtab = SubTabLightning(config)

  ### First Phase Learning
  train_ds = SubTabDataset(X_train, unlabeled_data=X_unlabeled)
  valid_ds = SubTabDataset(X_valid)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size, train_sampler='random', train_collate_fn=SubTabCollateFN(data_hparams), valid_collate_fn=SubTabCollateFN(data_hparams), n_jobs = 4)

  trainer = Trainer(
                      accelerator = 'cpu',
                      max_epochs = max_epochs,
                      num_sanity_val_steps = 2,
      )

  trainer.fit(pl_subtab, datamodule)

  ### Second Phase Learning

  pl_subtab.set_second_phase()

  train_ds = SubTabDataset(X_train, y_train.values)
  valid_ds = SubTabDataset(X_valid, y_valid.values)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = batch_size, train_sampler="weighted", train_collate_fn=SubTabCollateFN(data_hparams), valid_collate_fn=SubTabCollateFN(data_hparams))

  trainer.fit(pl_subtab, datamodule)

  # Evaluation
  from sklearn.metrics import accuracy_score
  import torch
  from torch.nn import functional as F
  from torch.utils.data import DataLoader, SequentialSampler

  test_ds = SubTabDataset(X_test)
  test_dl = DataLoader(test_ds, batch_size, shuffle=False, sampler = SequentialSampler(test_ds), num_workers=4, collate_fn=SubTabCollateFN(data_hparams))

  preds = trainer.predict(pl_subtab, test_dl)
          
  preds = F.softmax(torch.concat([out.cpu() for out in preds]).squeeze(),dim=1)

  accuracy = accuracy_score(y_test, preds.argmax(1))

  print("Accuracy %.2f" % accuracy)
  ```

</details>

#### SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption
SCARF introduces a contrastive learning framework specifically tailored for tabular data. By corrupting random subsets of features, SCARF creates diverse views for self-supervised learning in its first phase. The subsequent phase transitions to supervised learning, utilizing a pretrained encoder to enhance model accuracy and robustness.

<details close>
  <summary>Quick Start</summary>
  
  ```python
  # Assume that we have X_train, X_valid, X_test, y_train, y_valid, y_test, categorical_cols, and continuous_cols

  # Prepare the SubTabLightning Module
  from ts3l.pl_modules import SubTabLightning
  from ts3l.utils.subtab_utils import SubTabDataset, SubTabCollateFN
  from ts3l.utils import TS3LDataModule
  from ts3l.utils.subtab_utils import SubTabConfig
  from pytorch_lightning import Trainer

  metric = "accuracy_score"
  input_dim = X_train.shape[1]
  hidden_dim = 1024
  output_dim = 2
  tau = 1.0
  use_cosine_similarity = True
  use_contrastive = True
  use_distance = True
  n_subsets = 4
  overlap_ratio = 0.75

  mask_ratio = 0.1
  noise_type = "Swap"
  noise_level = 0.1

  data_hparams = {
              "n_subsets" : n_subsets,
              "overlap_ratio" : overlap_ratio,
              "mask_ratio" : mask_ratio,
              "noise_type" : noise_type,
              "noise_level" : noise_level,
              "n_column" : input_dim
          }

  batch_size = 128
  max_epochs = 3

  X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=0, stratify=y_train)

  config = SubTabConfig( task="classification", loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
  input_dim=input_dim, hidden_dim=hidden_dim,
  output_dim=output_dim, tau=tau, use_cosine_similarity= use_cosine_similarity, use_contrastive=use_contrastive, use_distance=use_distance, 
  n_subsets=n_subsets, overlap_ratio=overlap_ratio
  )

  pl_subtab = SubTabLightning(config)

  ### First Phase Learning
  train_ds = SubTabDataset(X_train, unlabeled_data=X_unlabeled)
  valid_ds = SubTabDataset(X_valid)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size, train_sampler='random', train_collate_fn=SubTabCollateFN(data_hparams), valid_collate_fn=SubTabCollateFN(data_hparams), n_jobs = 4)

  trainer = Trainer(
                      accelerator = 'cpu',
                      max_epochs = max_epochs,
                      num_sanity_val_steps = 2,
      )

  trainer.fit(pl_subtab, datamodule)

  ### Second Phase Learning

  pl_subtab.set_second_phase()

  train_ds = SubTabDataset(X_train, y_train.values)
  valid_ds = SubTabDataset(X_valid, y_valid.values)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = batch_size, train_sampler="weighted", train_collate_fn=SubTabCollateFN(data_hparams), valid_collate_fn=SubTabCollateFN(data_hparams))

  trainer.fit(pl_subtab, datamodule)

  # Evaluation
  from sklearn.metrics import accuracy_score
  import torch
  from torch.nn import functional as F
  from torch.utils.data import DataLoader, SequentialSampler

  test_ds = SubTabDataset(X_test)
  test_dl = DataLoader(test_ds, batch_size, shuffle=False, sampler = SequentialSampler(test_ds), num_workers=4, collate_fn=SubTabCollateFN(data_hparams))

  preds = trainer.predict(pl_subtab, test_dl)
          
  preds = F.softmax(torch.concat([out.cpu() for out in preds]).squeeze(),dim=1)

  accuracy = accuracy_score(y_test, preds.argmax(1))

  print("Accuracy %.2f" % accuracy)
  ```

</details>

#### To DO

- [x] Release nn.Module and Dataset of VIME, SubTab, and SCARF
  - [x] VIME
  - [x] SubTab
  - [x] SCARF
- [ ] Release LightningModules of VIME, SubTab, and SCARF
  - [x] VIME
  - [x] SubTab
  - [x] SCARF
- [ ] Release Denoising AutoEncoder
  - [ ] nn.Module
  - [ ] LightningModule
- [ ] Release SwitchTab
  - [ ] nn.Module
  - [ ] LightningModule
- [ ] Release PTaRL
  - [ ] Add Backbones
    - [ ] MLP
    - [ ] ResNet
    - [ ] FT-Transformer
  - [ ] LightningModule
- [ ] Add example codes

# Contributing

Contributions to this implementation are highly appreciated. Whether it's suggesting improvements, reporting bugs, or proposing new features, feel free to open an issue or submit a pull request.


## Credit  
```
@software{alcoholrithm_2024_10776538,
  author       = {Minwook Kim},
  title        = {TabularS3L},
  month        = mar,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.20},
  doi          = {10.5281/zenodo.10776538},
  url          = {https://doi.org/10.5281/zenodo.10776538}
}
```

