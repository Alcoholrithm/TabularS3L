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

<img src="https://github.com/Alcoholrithm/TabularS3L/assets/29500858/ba05aada-e801-42e6-ba20-10b7bef74b4d"/>

## Installation
We provide a Python package ts3l of TabularS3L for users who want to use semi- and self-supervised learning tabular models.

```sh
pip install ts3l
```

## Available Models with Quick Start

TabularS3L employs a two-phase learning approach, where the learning strategies differ between phases. Below is an overview of the models available within TabularS3L, highlighting the learning strategies employed in each phase. The abbreviations 'Self-SL', 'Semi-SL', and 'SL' represent self-supervised learning, semi-supervised learning, and supervised learning, respectively.

According to the original implementation and the paper, the encoder of DAE, VIME, and SubTab is frozen during the second phase of learning. However, you can choose to freeze the encoder (i.e. backbone network) or not by setting the **freeze_encoder** flag in the **set_second_phase** method.

| Model | First Phase | Second Phase |
|:---:|:---:|:---:|
| **DAE** ([GitHub](https://github.com/ryancheunggit/tabular_dae))| Self-SL | SL |
| **VIME** ([NeurIPS'20](https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html)) | Self-SL | Semi-SL or SL |
| **SubTab** ([NeurIPS'21](https://proceedings.neurips.cc/paper/2021/hash/9c8661befae6dbcd08304dbf4dcaf0db-Abstract.html)) | Self-SL | SL |
| **SCARF** ([ICLR'22](https://iclr.cc/virtual/2022/spotlight/6297))| Self-SL | SL |
| **SwitchTab** ([AAAI'24](https://ojs.aaai.org/index.php/AAAI/article/view/29523)) | Self-SL | SL |

In addition, TabularS3L employs a modular design, allowing you to freely choose the embedding and backbone modules.

The currently supported modules are:

- **Embedding modules**: 
  - `identity`
  - `feature_tokenizer` (from *Revisiting Deep Learning Models for Tabular Data*)
- **Backbone modules**: 
  - `mlp`
  - `transformer`

Note: The `transformer` backbone requires the `feature_tokenizer` as its embedding module.

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
  from ts3l.utils import TS3LDataModule, get_category_cardinality
  from ts3l.utils.dae_utils import DAEConfig
  from ts3l.utils.embedding_utils import IdentityEmbeddingConfig
  from ts3l.utils.backbone_utils import MLPBackboneConfig
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

  embedding_config = IdentityEmbeddingConfig(input_dim = input_dim)
  backbone_config = MLPBackboneConfig(input_dim = embedding_config.output_dim)

  config = DAEConfig( 
                  task="classification", loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
                  embedding_config=embedding_config, backbone_config=backbone_config,
                  output_dim=output_dim,
                  noise_type = noise_type,
                  noise_ratio = noise_ratio,
                  cat_cardinality=get_category_cardinality(data, category_cols), num_continuous=len(continuous_cols)
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
  from ts3l.utils import TS3LDataModule, get_category_cardinality
  from ts3l.utils.vime_utils import VIMEConfig
  from ts3l.utils.embedding_utils import IdentityEmbeddingConfig
  from ts3l.utils.backbone_utils import MLPBackboneConfig
  from pytorch_lightning import Trainer

  metric = "accuracy_score"
  input_dim = X_train.shape[1]
  predictor_dim = 1024
  output_dim = 2
  alpha1 = 2.0
  alpha2 = 2.0
  beta = 1.0
  K = 3
  p_m = 0.2

  batch_size = 128
  max_epochs = 20

  X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=0, stratify=y_train)

  embedding_config = IdentityEmbeddingConfig(input_dim = input_dim)
  backbone_config = MLPBackboneConfig(input_dim = embedding_config.output_dim)

  config = VIMEConfig( 
                      task="classification", loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
                      embedding_config=embedding_config, backbone_config=backbone_config,
                      predictor_dim=predictor_dim,
                      output_dim=output_dim, alpha1=alpha1, alpha2=alpha2, 
                      beta=beta, K=K, p_m = p_m,
                      cat_cardinality=get_category_cardinality(data, category_cols), num_continuous=len(continuous_cols)
  )

  pl_vime = VIMELightning(config)

  ### First Phase Learning
  train_ds = VIMEDataset(X = X_train, unlabeled_data = X_unlabeled, config=config, continuous_cols = continuous_cols, category_cols = category_cols)
  valid_ds = VIMEDataset(X = X_valid, config=config, continuous_cols = continuous_cols, category_cols = category_cols)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size, train_sampler='random')

  trainer = Trainer(
                      accelerator = 'cpu',
                      max_epochs = max_epochs,
                      num_sanity_val_steps = 2,
      )

  trainer.fit(pl_vime, datamodule)

  ### Second Phase Learning
  from ts3l.utils.vime_utils import VIMESecondPhaseCollateFN

  pl_vime.set_second_phase()

  train_ds = VIMEDataset(X_train, y_train.values, config, unlabeled_data=X_unlabeled, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
  valid_ds = VIMEDataset(X_valid, y_valid.values, config, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
          
  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = batch_size, train_sampler="weighted", train_collate_fn=VIMESecondPhaseCollateFN())

  trainer = Trainer(
                      accelerator = 'cpu',
                      max_epochs = max_epochs,
                      num_sanity_val_steps = 2,
      )

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
from ts3l.utils.subtab_utils import SubTabDataset
from ts3l.utils import TS3LDataModule
from ts3l.utils.subtab_utils import SubTabConfig
from ts3l.utils.embedding_utils import IdentityEmbeddingConfig
from ts3l.utils.backbone_utils import MLPBackboneConfig
from pytorch_lightning import Trainer

metric = "accuracy_score"
input_dim = X_train.shape[1]
projection_dim = 1024
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
max_epochs = 20

X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=0, stratify=y_train)

embedding_config = IdentityEmbeddingConfig(input_dim = input_dim)
backbone_config = MLPBackboneConfig(input_dim = embedding_config.output_dim)

config = SubTabConfig( 
                    task="classification", loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
                    embedding_config=embedding_config, backbone_config=backbone_config,
                    projection_dim=projection_dim,
                    output_dim=output_dim, tau=tau, use_cosine_similarity= use_cosine_similarity, use_contrastive=use_contrastive, use_distance=use_distance, 
                    n_subsets=n_subsets, overlap_ratio=overlap_ratio, mask_ratio=mask_ratio, noise_type=noise_type, noise_level=noise_level
)

pl_subtab = SubTabLightning(config)

### First Phase Learning
train_ds = SubTabDataset(X_train, unlabeled_data=X_unlabeled, continuous_cols=continuous_cols, category_cols=category_cols)
valid_ds = SubTabDataset(X_valid, continuous_cols=continuous_cols, category_cols=category_cols)

datamodule = TS3LDataModule(train_ds, valid_ds, batch_size, train_sampler='random', n_jobs = 4)

trainer = Trainer(
                    accelerator = 'cpu',
                    max_epochs = max_epochs,
                    num_sanity_val_steps = 2,
    )

trainer.fit(pl_subtab, datamodule)

### Second Phase Learning

pl_subtab.set_second_phase()

train_ds = SubTabDataset(X_train, y_train.values, continuous_cols=continuous_cols, category_cols=category_cols)
valid_ds = SubTabDataset(X_valid, y_valid.values, continuous_cols=continuous_cols, category_cols=category_cols)

datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = batch_size, train_sampler="weighted")

trainer = Trainer(
                    accelerator = 'cpu',
                    max_epochs = max_epochs,
                    num_sanity_val_steps = 2,
    )

trainer.fit(pl_subtab, datamodule)

# Evaluation
from sklearn.metrics import accuracy_score
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler

test_ds = SubTabDataset(X_test, continuous_cols=continuous_cols, category_cols=category_cols)
test_dl = DataLoader(test_ds, batch_size, shuffle=False, sampler = SequentialSampler(test_ds), num_workers=4)

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
  from ts3l.utils.embedding_utils import IdentityEmbeddingConfig
  from ts3l.utils.backbone_utils import MLPBackboneConfig
  from pytorch_lightning import Trainer

  metric = "accuracy_score"
  input_dim = X_train.shape[1]
  pretraining_head_dim = 1024
  output_dim = 2
  head_depth = 2
  dropout_rate = 0.04

  corruption_rate = 0.6

  batch_size = 128
  max_epochs = 10

  X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=0, stratify=y_train)

  embedding_config = IdentityEmbeddingConfig(input_dim = input_dim)
  backbone_config = MLPBackboneConfig(input_dim = embedding_config.output_dim)

  config = SCARFConfig( 
                      task="classification", loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
                      embedding_config=embedding_config, backbone_config=backbone_config,
                      pretraining_head_dim=pretraining_head_dim,
                      output_dim=output_dim, head_depth=head_depth,
                      dropout_rate=dropout_rate, corruption_rate = corruption_rate
  )

  pl_scarf = SCARFLightning(config)

  ### First Phase Learning
  train_ds = SCARFDataset(X_train, unlabeled_data=X_unlabeled, config = config, continuous_cols=continuous_cols, category_cols=category_cols)
  valid_ds = SCARFDataset(X_valid, config=config, continuous_cols=continuous_cols, category_cols=category_cols)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size=batch_size, train_sampler="random")

  trainer = Trainer(
                      accelerator = 'cpu',
                      max_epochs = max_epochs,
                      num_sanity_val_steps = 2,
      )

  trainer.fit(pl_scarf, datamodule)

  ### Second Phase Learning

  pl_scarf.set_second_phase()

  train_ds = SCARFDataset(X_train, y_train.values, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
  valid_ds = SCARFDataset(X_valid, y_valid.values, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = batch_size, train_sampler="weighted")

  trainer = Trainer(
                      accelerator = 'cpu',
                      max_epochs = max_epochs,
                      num_sanity_val_steps = 2,
      )

  trainer.fit(pl_scarf, datamodule)

  # Evaluation
  from sklearn.metrics import accuracy_score
  import torch
  from torch.nn import functional as F
  from torch.utils.data import DataLoader, SequentialSampler

  test_ds = SCARFDataset(X_test, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
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
  from ts3l.utils.embedding_utils import FTEmbeddingConfig
  from ts3l.utils.backbone_utils import TransformerBackboneConfig
  from ts3l.utils.misc import get_category_cardinality
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

  embedding_config = FTEmbeddingConfig(input_dim = input_dim, emb_dim = 128, cont_nums = len(continuous_cols),
                                          cat_cardinality=get_category_cardinality(data, category_cols), required_token_dim=2)
  backbone_config = TransformerBackboneConfig(d_model = embedding_config.emb_dim, encoder_depth = encoder_depth, n_head = n_head, hidden_dim = hidden_dim)

  config = SwitchTabConfig(
      task="classification", loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
      embedding_config=embedding_config, backbone_config=backbone_config,
      output_dim = output_dim,
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

This section provides a simple benchmark comparing TabularS3L with XGBoost. The train-validation-test ratio is 6:2:2, and each model is tuned over 50 trials using Optuna. The results are averaged over five random seeds (0 to 4). The best results are shown in bold. `acc`, `b-acc`, and `mse` stand for `Accuracy`, `Balanced Accuracy`, and `Mean Squared Error`, respectively.

The embedding and backbone modules used for each model are as follows, which align with their original papers or repositories.

| Model | embedding | backbone |
|:---:|:---:|:---:|
| DAE | <code>identity</code> | <code>mlp</code> |
| VIME | <code>identity</code> | <code>mlp</code> |
| SubTab | <code>identity</code> | <code>mlp</code> |
| SCARF | <code>identity</code> | <code>mlp</code> |
| SwitchTab | <code>feature_tokenizer</code> | <code>transformer</code> |

Use this benchmark for reference only, as only a small number of random seeds were used.

--------

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

## Contributing

Contributions to this implementation are highly appreciated. Whether it's suggesting improvements, reporting bugs, or proposing new features, feel free to open an issue or submit a pull request.

## Citation

#### BibTex

```
@software{Kim_TabularS3L_2024,
author = {Kim, Minwook},
doi = {10.5281/zenodo.10776538},
month = nov,
title = {{TabularS3L}},
url = {https://github.com/Alcoholrithm/TabularS3L},
version = {0.60},
year = {2024}
}
```

#### APA

```
Kim, M. (2024). TabularS3L (Version 0.60) [Computer software]. https://doi.org/10.5281/zenodo.10776538
```

