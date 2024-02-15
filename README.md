# TabularS3L
A PyTorch-based library for self- and semi-supervised learning tabular models.
Currently, [VIME](https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html), [SubTab](https://proceedings.neurips.cc/paper/2021/hash/9c8661befae6dbcd08304dbf4dcaf0db-Abstract.html) and [SCARF](https://iclr.cc/virtual/2022/spotlight/6297) are available.

#### To DO

- [x] Launch nn.Module and Dataset of VIME, SubTab, and SCARF
  - [x] VIME
  - [x] SubTab
  - [x] SCARF
- [ ] Launch pytorch lightning modules of VIME, SubTab, and SCARF
  - [ ] VIME
  - [ ] SubTab
  - [ ] SCARF
- Finish README.md

## Installation
We provide a Python package ts3l of TabularS3L for users who want to use semi- and self-supervised learning tabular models.

```sh
pip install ts3l
```

## How to use?

```python
# Assume we have X_train and y_train. And assume that we also have X_unlabeled for self-supervised learning

from ts3l.models import SCARF
from ts3l.utils.scarf_utils import SCARFDataset
from ts3l.utils.scarf_utils import NTXentLoss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import RandomSampler, WeightedRandomSampler, Dataset, DataLoader

emb_dim = 128
encoder_depth = 4
head_depth = 2
corruption_rate = 6
dropout_rate = 0.15

batch_size = 128

model = SCARF(input_dim = X_train.shape[1],
        emb_dim = emb_dim,
        encoder_depth = encoder_depth,
        head_depth = head_depth,
        dropout_rate = dropout_rate,
        out_dim = 2)
pretraining_loss = NTXentLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_ds = SCARFDataset(X_train.append(X_unlabeled), corruption_len=int(corruption_rate * X_train.shape[1]))

model.do_pretraining() # Now, model.forward conducts self-superivsed learning.

train_dl = DataLoader(train_ds, 
                        batch_size = batch_size, 
                        shuffle=False, 
                        sampler = RandomSampler(train_ds),
                        num_workers=4,
                        drop_last=True)

for epoch in range(2): 
    for i, data in enumerate(train_dl, 0):
        
        optimizer.zero_grad()

        x, x_corrupted = data
        emb_anchor, emb_corrupted = model(x, x_corrupted)

        loss = pretraining_loss(emb_anchor, emb_corrupted)

        loss.backward()
        optimizer.step()


train_ds = SCARFDataset(X_train, y_train.values)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.do_finetunning() # Now, model.forward conducts (semi-)superivsed learning.

train_dl = DataLoader(train_ds, 
                        batch_size = batch_size, 
                        shuffle=False, 
                        sampler = WeightedRandomSampler(train_ds.weights, num_samples = len(train_ds)),
                        num_workers=4,
                        drop_last=True)

for epoch in range(2): 
    for i, data in enumerate(train_dl, 0):
        
        optimizer.zero_grad()

        x, y = data
        y_hat = model(x)

        loss = criterion(y_hat, y)

        loss.backward()
        optimizer.step()

```
