import pandas as pd
from typing import Any, List, Union, Optional, Tuple
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset

import numpy as np
from ts3l.utils.tabularbinning_utils import TabularBinningConfig

class TabularBinningDataset(Dataset):
    def __init__(self):
        return