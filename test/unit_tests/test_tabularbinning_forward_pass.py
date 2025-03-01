from ts3l.utils.tabularbinning_utils import TabularBinningConfig, TabularBinningDataset, TabularBinningFirstPhaseCollateFN
from ts3l.pl_modules import TabularBinningLightning

from torch.utils.data import DataLoader, SequentialSampler

import pytest

import sys
import os
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../..'))
sys.path.append(os.path.join(here, '..'))

import numpy as np

from misc import embedding_backbone_list, prepare_test
from benchmark.datasets import load_diabetes, load_cmc, load_abalone

@pytest.mark.parametrize("load_data", [load_diabetes, load_cmc, load_abalone])
@pytest.mark.parametrize("embedding_type, backbone_type", embedding_backbone_list)
@pytest.mark.parametrize("pretext_task", ["BinRecon", "BinXent"])
@pytest.mark.parametrize("mask_type", ["random", "constant"])
def test_tabularbinning_first_phase_forward(load_data, embedding_type, backbone_type, pretext_task, mask_type):
    
    data, label, continuous_cols, category_cols, output_dim, kwargs = prepare_test(load_data, embedding_type, backbone_type)
    
    config = TabularBinningConfig(n_bin = 10, 
                                  pretext_task = pretext_task, 
                                  mask_type = mask_type,
                                  **kwargs)
    
    pl_module = TabularBinningLightning(config)
    
    if mask_type == "constant":
        constant_x_bar = data.mean()
        constant_x_bar[category_cols] = np.round(constant_x_bar[category_cols])
        constant_x_bar = np.concat((constant_x_bar[category_cols].values, constant_x_bar[continuous_cols].values))
    else:
        constant_x_bar = None

    test_ds = TabularBinningDataset(config, data, category_cols=category_cols, continuous_cols=continuous_cols)
    test_dl = DataLoader(test_ds, 128, shuffle=False, sampler = SequentialSampler(test_ds), collate_fn=TabularBinningFirstPhaseCollateFN(config, constant_x_bar))
    
    batch = next(iter(test_dl))

    print("Test The First Phase Forward")
    pl_module._get_first_phase_loss(batch)
    print("Passed The First Phase Forward")

@pytest.mark.parametrize("load_data", [load_diabetes, load_cmc, load_abalone])
@pytest.mark.parametrize("embedding_type, backbone_type", embedding_backbone_list)
def test_tabularbinning_second_phase_forward(load_data, embedding_type, backbone_type):
    
    data, label, continuous_cols, category_cols, output_dim, kwargs = prepare_test(load_data, embedding_type, backbone_type)
    
    config = TabularBinningConfig(n_bin = 10, 
                                  **kwargs)

    pl_module = TabularBinningLightning(config)

    pl_module.set_second_phase()
    test_ds = TabularBinningDataset(config, data, label, category_cols=category_cols, continuous_cols=continuous_cols, is_regression= True if output_dim == 1 else False, is_second_phase=True)
    test_dl = DataLoader(test_ds, 128, shuffle=False, sampler = SequentialSampler(test_ds))
    
    batch = next(iter(test_dl))
    
    print("Test The Second Phase Forward")
    pl_module._get_second_phase_loss(batch)
    print("Passed The Second Phase Forward")
    
if __name__ == "__main__":
    test_tabularbinning_first_phase_forward(load_cmc, "feature_tokenizer", "transformer", "BinRecon", "constant")
    test_tabularbinning_second_phase_forward(load_abalone, "feature_tokenizer", "transformer")