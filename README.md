# TabularS3L

[**Overview**](#tabulars3l)
| [**Installation**](#installation)
| [**Available Models with Quick Start Guides**](#available-models-with-quick-start)
| [**Benchmark**](#benchmark)
| [**To DO**](#to-do)
| [**Contributing**](#contributing)


[![pypi](https://img.shields.io/pypi/v/ts3l)](https://pypi.org/project/ts3l/0.20/)
[![Downloads](https://static.pepy.tech/badge/ts3l)](https://pepy.tech/project/ts3l)
[![DOI](https://zenodo.org/badge/756740921.svg)](https://zenodo.org/doi/10.5281/zenodo.10776537)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TabularS3L is a PyTorch Lightning-based library designed to facilitate self- and semi-supervised learning with tabular data. While numerous self- and semi-supervised learning tabular models have been proposed, there is a lack of a comprehensive library that addresses the needs of tabular practitioners. This library aims to fill this gap by providing a unified PyTorch Lightning-based framework for exploring and deploying such models.

## Installation
We provide a Python package ts3l of TabularS3L for users who want to use semi- and self-supervised learning tabular models.

```sh
pip install ts3l
```

## Available Models with Quick Start

TabularS3L employs a two-phase learning approach, where the learning strategies differ between phases. Below is an overview of the models available within TabularS3L, highlighting the learning strategies employed in each phase. The abbreviations 'Self-SL', 'Semi-SL', and 'SL' represent self-supervised learning, semi-supervised learning, and supervised learning, respectively.

| Model | First Phase | Second Phase |
|:---:|:---:|:---:|
| **DAE** ([GitHub](https://github.com/ryancheunggit/tabular_dae))| Self-SL | SL |
| **VIME** ([NeurIPS'20](https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html)) | Self-SL | Semi-SL or SL |
| **SubTab** ([NeurIPS'21](https://proceedings.neurips.cc/paper/2021/hash/9c8661befae6dbcd08304dbf4dcaf0db-Abstract.html)) | Self-SL | SL |
| **SCARF** ([ICLR'22](https://iclr.cc/virtual/2022/spotlight/6297))| Self-SL | SL |
| **SwitchTab** ([AAAI'24](https://ojs.aaai.org/index.php/AAAI/article/view/29523)) | Self-SL | SL |

#### Denoising AutoEncoder (DAE)
DAE processes input data that has been partially corrupted, producing clean data and predicting which features are corrupted during the self-supervised learning.
The denoising task enables the model to learn the input distribution and generate latent representations that are robust to corruption. 
These latent representations can be utilized for a variety of downstream tasks.

<details close>
  <summary>Quick Start</summary>
  
  ```python
  # Assume that we have X_train, X_valid, X_test, y_train, y_valid, y_test, categorical_cols, and continuous_cols

  # Prepare the DAELightning Module
  from ts3l.pl_modules import DAELightning
  from ts3l.utils.dae_utils import DAEDataset, DAECollateFN
  from ts3l.utils import TS3LDataModule
  from ts3l.utils.dae_utils import DAEConfig
  from pytorch_lightning import Trainer
  
  metric = "accuracy_score"
  input_dim = X_train.shape[1]
  hidden_dim = 1024
  output_dim = 2
  encoder_depth=4
  head_depth = 2
  noise_type = "Swap"
  noise_ratio = 0.3
  
  max_epochs = 20
  batch_size = 128
  
  X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=0, stratify=y_train)
  
  config = DAEConfig( task="classification", loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
  input_dim=input_dim, hidden_dim=hidden_dim,
  output_dim=output_dim, encoder_depth=encoder_depth,
  head_depth = head_depth,
  noise_type = noise_type,
  noise_ratio = noise_ratio,
  num_categoricals=len(category_cols), num_continuous=len(continuous_cols)
  )
  
  pl_dae = DAELightning(config)
  
  ### First Phase Learning
  train_ds = DAEDataset(X = X_train, unlabeled_data = X_unlabeled, continuous_cols = continuous_cols, category_cols = category_cols)
  valid_ds = DAEDataset(X = X_valid, continuous_cols = continuous_cols, category_cols = category_cols)
  
  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size, train_sampler='random', train_collate_fn=DAECollateFN(config), valid_collate_fn=DAECollateFN(config))
  
  trainer = Trainer(
                      accelerator = 'cpu',
                      max_epochs = max_epochs,
                      num_sanity_val_steps = 2,
      )
  
  trainer.fit(pl_dae, datamodule)
  
  ### Second Phase Learning
  
  pl_dae.set_second_phase()
  
  train_ds = DAEDataset(X = X_train, Y = y_train.values, unlabeled_data=X_unlabeled, continuous_cols=continuous_cols, category_cols=category_cols)
  valid_ds = DAEDataset(X = X_valid, Y = y_valid.values, continuous_cols=continuous_cols, category_cols=category_cols)
          
  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = batch_size, train_sampler="weighted")
  
  trainer = Trainer(
                      accelerator = 'cpu',
                      max_epochs = max_epochs,
                      num_sanity_val_steps = 2,
      )
  
  trainer.fit(pl_dae, datamodule)
  
  # Evaluation
  from sklearn.metrics import accuracy_score
  import torch
  from torch.nn import functional as F
  from torch.utils.data import DataLoader, SequentialSampler
  
  test_ds = DAEDataset(X_test, category_cols=category_cols, continuous_cols=continuous_cols)
  test_dl = DataLoader(test_ds, batch_size, shuffle=False, sampler = SequentialSampler(test_ds))
  
  preds = trainer.predict(pl_dae, test_dl)
          
  preds = F.softmax(torch.concat([out.cpu() for out in preds]).squeeze(),dim=1)
  
  accuracy = accuracy_score(y_test, preds.argmax(1))
  
  print("Accuracy %.2f" % accuracy)
  ```

</details>

#### VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain
VIME enhances tabular data learning through a dual approach. In its first phase, it utilize a pretext task of estimating mask vectors from corrupted tabular data, alongside a reconstruction pretext task for self-supervised learning. The second phase leverages consistency regularization on unlabeled data.

<details close>
  <summary>Quick Start</summary>
  
  ```python
  # Assume that we have X_train, X_valid, X_test, y_train, y_valid, y_test, categorical_cols, and continuous_cols

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

  batch_size = 128

  X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=0, stratify=y_train)

  config = VIMEConfig( task="classification", loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
  input_dim=input_dim, hidden_dim=hidden_dim,
  output_dim=output_dim, alpha1=alpha1, alpha2=alpha2, 
  beta=beta, K=K, p_m = p_m,
  num_categoricals=len(category_cols), num_continuous=len(continuous_cols)
  )

  pl_vime = VIMELightning(config)

  ### First Phase Learning
  train_ds = VIMEDataset(X = X_train, unlabeled_data = X_unlabeled, config=config, continuous_cols = continuous_cols, category_cols = category_cols)
  valid_ds = VIMEDataset(X = X_valid, config=config, continuous_cols = continuous_cols, category_cols = category_cols)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size, train_sampler='random')

  trainer = Trainer(
                      accelerator = 'cpu',
                      max_epochs = 20,
                      num_sanity_val_steps = 2,
      )

  trainer.fit(pl_vime, datamodule)

  ### Second Phase Learning
  from ts3l.utils.vime_utils import VIMESemiSLCollateFN

  pl_vime.set_second_phase()

  train_ds = VIMEDataset(X_train, y_train.values, config, unlabeled_data=X_unlabeled, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
  valid_ds = VIMEDataset(X_valid, y_valid.values, config, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
          
  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = batch_size, train_sampler="weighted", train_collate_fn=VIMESemiSLCollateFN())

  trainer.fit(pl_vime, datamodule)

  # Evaluation
  from sklearn.metrics import accuracy_score
  import torch
  from torch.nn import functional as F
  from torch.utils.data import DataLoader, SequentialSampler

  test_ds = VIMEDataset(X_test, category_cols=category_cols, continuous_cols=continuous_cols, is_second_phase=True)
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

  batch_size = 128
  max_epochs = 3

  X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=0, stratify=y_train)

  config = SubTabConfig( task="classification", loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
  input_dim=input_dim, hidden_dim=hidden_dim,
  output_dim=output_dim, tau=tau, use_cosine_similarity= use_cosine_similarity, use_contrastive=use_contrastive, use_distance=use_distance, 
  n_subsets=n_subsets, overlap_ratio=overlap_ratio, mask_ratio=mask_ratio, noise_type=noise_type, noise_level=noise_level
  )

  pl_subtab = SubTabLightning(config)

  ### First Phase Learning
  train_ds = SubTabDataset(X_train, unlabeled_data=X_unlabeled)
  valid_ds = SubTabDataset(X_valid)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size, train_sampler='random', train_collate_fn=SubTabCollateFN(config), valid_collate_fn=SubTabCollateFN(config), n_jobs = 4)

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

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = batch_size, train_sampler="weighted", train_collate_fn=SubTabCollateFN(config), valid_collate_fn=SubTabCollateFN(config))

  trainer.fit(pl_subtab, datamodule)

  # Evaluation
  from sklearn.metrics import accuracy_score
  import torch
  from torch.nn import functional as F
  from torch.utils.data import DataLoader, SequentialSampler

  test_ds = SubTabDataset(X_test)
  test_dl = DataLoader(test_ds, batch_size, shuffle=False, sampler = SequentialSampler(test_ds), num_workers=4, collate_fn=SubTabCollateFN(config))

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

  # Prepare the SCARFLightning Module
  from ts3l.pl_modules import SCARFLightning
  from ts3l.utils.scarf_utils import SCARFDataset
  from ts3l.utils import TS3LDataModule
  from ts3l.utils.scarf_utils import SCARFConfig
  from pytorch_lightning import Trainer

  metric = "accuracy_score"
  input_dim = X_train.shape[1]
  hidden_dim = 1024
  output_dim = 2
  encoder_depth = 3
  head_depth = 1
  dropout_rate = 0.04

  corruption_rate = 0.6

  batch_size = 128
  max_epochs = 10

  X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=0, stratify=y_train)

  config = SCARFConfig( task="classification", loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
  input_dim=input_dim, hidden_dim=hidden_dim,
  output_dim=output_dim, encoder_depth=encoder_depth, head_depth=head_depth,
  dropout_rate=dropout_rate, corruption_rate = corruption_rate
  )

  pl_scarf = SCARFLightning(config)

  ### First Phase Learning
  train_ds = SCARFDataset(X_train, unlabeled_data=X_unlabeled, config = config)
  valid_ds = SCARFDataset(X_valid, config=config)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size=batch_size, train_sampler="random")

  trainer = Trainer(
                      accelerator = 'cpu',
                      max_epochs = max_epochs,
                      num_sanity_val_steps = 2,
      )

  trainer.fit(pl_scarf, datamodule)

  ### Second Phase Learning

  pl_scarf.set_second_phase()

  train_ds = SCARFDataset(X_train, y_train.values, is_second_phase=True)
  valid_ds = SCARFDataset(X_valid, y_valid.values, is_second_phase=True)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = batch_size, train_sampler="weighted")

  trainer.fit(pl_scarf, datamodule)

  # Evaluation
  from sklearn.metrics import accuracy_score
  import torch
  from torch.nn import functional as F
  from torch.utils.data import DataLoader, SequentialSampler

  test_ds = SCARFDataset(X_test, is_second_phase=True)
  test_dl = DataLoader(test_ds, batch_size, shuffle=False, sampler = SequentialSampler(test_ds), num_workers=4)

  preds = trainer.predict(pl_scarf, test_dl)
          
  preds = F.softmax(torch.concat([out.cpu() for out in preds]).squeeze(),dim=1)

  accuracy = accuracy_score(y_test, preds.argmax(1))

  print("Accuracy %.2f" % accuracy)
  ```

</details>

#### SwitchTab: Switched Autoencoders Are Effective Tabular Learners
SwitchTab introduces a novel self-supervised method specifically designed to decuple mutual and salient features among data pair, resulting in more representative embeddings.
Moreover, the pre-trained salient embeddings can be utilized as plug-and-play features to enhance the performance of various traditional classification methods.

<details close>
  <summary>Quick Start</summary>
  
  ```python
  # Assume that we have X_train, X_valid, X_test, y_train, y_valid, y_test, categorical_cols, and continuous_cols

  # Prepare the SwitchTabLightning Module
  from ts3l.pl_modules import SwitchTabLightning
  from ts3l.utils.switchtab_utils import SwitchTabDataset, SwitchTabFirstPhaseCollateFN
  from ts3l.utils import TS3LDataModule
  from ts3l.utils.switchtab_utils import SwitchTabConfig
  from pytorch_lightning import Trainer

  metric = "accuracy_score"
  input_dim = X_train.shape[1]
  hidden_dim = 1024
  output_dim = 2

  encoder_depth = 3
  n_head = 2
  u_label = -1

  batch_size = 128

  X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=0, stratify=y_train)

  config = SwitchTabConfig( task="classification", loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
  input_dim=input_dim, hidden_dim=hidden_dim,
  output_dim=output_dim, encoder_depth=encoder_depth,
  n_head = n_head,
  u_label = u_label
  )

  pl_switchtab = SwitchTabLightning(config)

  ### First Phase Learning
  train_ds = SwitchTabDataset(X = X_train, unlabeled_data = X_unlabeled, Y = y_train.values, config=config, continuous_cols=continuous_cols, category_cols=category_cols)
  valid_ds = SwitchTabDataset(X = X_valid, config=config, Y = y_valid.values, continuous_cols=continuous_cols, category_cols=category_cols)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size, train_sampler='weighted', train_collate_fn=SwitchTabFirstPhaseCollateFN(), valid_collate_fn=SwitchTabFirstPhaseCollateFN())

  trainer = Trainer(
                  accelerator = 'cpu',
                  max_epochs = 20,
                  num_sanity_val_steps = 2,
  )

  trainer.fit(pl_switchtab, datamodule)

  ### Second Phase Learning

  pl_switchtab.set_second_phase()

  train_ds = SwitchTabDataset(X = X_train, Y = y_train.values, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
  valid_ds = SwitchTabDataset(X = X_valid, Y = y_valid.values, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
      
  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = batch_size, train_sampler="weighted")

  trainer = Trainer(
                  accelerator = 'cpu',
                  max_epochs = 20,
                  num_sanity_val_steps = 2,
  )

  trainer.fit(pl_switchtab, datamodule)

  # Evaluation
  from sklearn.metrics import accuracy_score
  import torch
  from torch.nn import functional as F
  from torch.utils.data import DataLoader, SequentialSampler

  test_ds = SwitchTabDataset(X_test, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
  test_dl = DataLoader(test_ds, batch_size, shuffle=False, sampler = SequentialSampler(test_ds))

  preds = trainer.predict(pl_switchtab, test_dl)
      
  preds = F.softmax(torch.concat([out.cpu() for out in preds]).squeeze(),dim=1)

  accuracy = accuracy_score(y_test, preds.argmax(1))

  print("Accuracy %.2f" % accuracy)
  ```

</details>

## Benchmark

We provide a simple benchmark using TabularS3L against XGBoost. The train-validation-test ratio is 6:2:2 and we tuned each model over 50 trials using Optuna. The results are the average of the random seeds [0,4]. The best results are bold. 'acc', 'b-acc', and 'mse' mean 'Accuracy', 'Balanced Accuracy', and 'Mean Squared Error', respectively.

Use this benchmark for reference only, as only a small number of random seeds were used.

##### 10% labeled samples 

| Model | diabetes (acc) | cmc (b-acc) | abalone (mse) |
|:---:|:---:|:---:|:---:|
| XGBoost | 0.7325 | 0.4763 | **5.5739** |
| DAE | 0.7208 | 0.4885 | 5.6168 | 
| VIME | 0.7182 | **0.5087** | 5.6637 |
| SubTab | 0.7312 | 0.4930 | 7.2418 |
| SCARF | **0.7416** | 0.4710 | 5.8888 |
| SwitchTab |  0.7156 | 0.4886 | 5.9907 |

--------

##### 100% labeled samples

| Model | diabetes (acc) | cmc (b-acc) | abalone (mse) |
|:---:|:---:|:---:|:---:|
| XGBoost | 0.7234 | 0.5291 | 4.8377 |
| DAE | 0.7390 | 0.5500 | 4.5758 |
| VIME | **0.7688** | 0.5477 | 4.5804 |
| SubTab | 0.7390 | 0.5432 | 6.3104 |
| SCARF | 0.7442 | **0.5521** | **4.4443** |
| SwitchTab | 0.7585 | 0.5411 | 4.7489 |

## To DO

- [x] Release nn.Module and Dataset of VIME, SubTab, and SCARF
- [x] Release LightningModules of VIME, SubTab, and SCARF
- [x] Release Denoising AutoEncoder
- [x] Release SwitchTab
- [x] Add example codes
- [ ] Release more tabular models

## Contributing

Contributions to this implementation are highly appreciated. Whether it's suggesting improvements, reporting bugs, or proposing new features, feel free to open an issue or submit a pull request.

