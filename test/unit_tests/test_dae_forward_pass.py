from ts3l.utils.dae_utils import DAEConfig, DAEDataset, DAECollateFN
from ts3l.pl_modules import DAELightning

from torch.utils.data import DataLoader, SequentialSampler

import pytest

import sys
import os
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../..'))
sys.path.append(os.path.join(here, '..'))

from misc import embedding_backbone_list, prepare_test
from benchmark.datasets import load_diabetes, load_cmc, load_abalone

@pytest.mark.parametrize("load_data", [load_diabetes, load_cmc, load_abalone])
@pytest.mark.parametrize("embedding_type, backbone_type", embedding_backbone_list)
def test_dae_first_phase_forward(load_data, embedding_type, backbone_type):
    
    data, label, continuous_cols, category_cols, output_dim, kwargs = prepare_test(load_data, embedding_type, backbone_type)
    
    config = DAEConfig(
        num_categoricals = len(category_cols),
        num_continuous = len(continuous_cols),
        **kwargs
    )
    
    pl_module = DAELightning(config)
    
    test_ds = DAEDataset(data, category_cols=category_cols, continuous_cols=continuous_cols)
    test_dl = DataLoader(test_ds, 128, shuffle=False, sampler = SequentialSampler(test_ds), collate_fn=DAECollateFN(config))
    
    batch = next(iter(test_dl))

    print("Test First Phase Forward")
    pl_module._get_first_phase_loss(batch)

@pytest.mark.parametrize("load_data", [load_diabetes, load_cmc, load_abalone])
@pytest.mark.parametrize("embedding_type, backbone_type", embedding_backbone_list)
def test_dae_second_phase_forward(load_data, embedding_type, backbone_type):
    
    data, label, continuous_cols, category_cols, output_dim, kwargs = prepare_test(load_data, embedding_type, backbone_type)
    
    config = DAEConfig(
        num_categoricals = len(category_cols),
        num_continuous = len(continuous_cols),
        **kwargs
    )
    
    pl_module = DAELightning(config)
    
    pl_module.set_second_phase()
    test_ds = DAEDataset(data, label, category_cols=category_cols, continuous_cols=continuous_cols, is_regression= True if output_dim == 1 else False)
    test_dl = DataLoader(test_ds, 128, shuffle=False, sampler = SequentialSampler(test_ds))
    
    batch = next(iter(test_dl))
    
    print("Test Second Phase Forward")
    pl_module._get_second_phase_loss(batch)
    
if __name__ == "__main__":
    test_dae_first_phase_forward(load_diabetes, "feature_tokenizer", "transformer")