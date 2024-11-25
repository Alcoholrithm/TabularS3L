from ts3l.utils.switchtab_utils import SwitchTabConfig, SwitchTabDataset, SwitchTabFirstPhaseCollateFN
from ts3l.pl_modules import SwitchTabLightning

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
def test_switchtab_first_phase_forward(load_data, embedding_type, backbone_type):
    
    data, label, continuous_cols, category_cols, output_dim, kwargs = prepare_test(load_data, embedding_type, backbone_type)

    config = SwitchTabConfig(
        **kwargs
    )
    
    pl_module = SwitchTabLightning(config)
    
    test_ds = SwitchTabDataset(data, config=config, continuous_cols=continuous_cols, category_cols=category_cols)
    test_dl = DataLoader(test_ds, 128, shuffle=False, sampler = SequentialSampler(test_ds), collate_fn=SwitchTabFirstPhaseCollateFN())

    batch = next(iter(test_dl))
    
    print("Test The First Phase Forward")
    pl_module._get_first_phase_loss(batch)
    print("Passed The First Phase Forward")

@pytest.mark.parametrize("load_data", [load_diabetes, load_cmc, load_abalone])
@pytest.mark.parametrize("embedding_type, backbone_type", embedding_backbone_list)
def test_switchtab_second_phase_forward(load_data, embedding_type, backbone_type):
    
    data, label, continuous_cols, category_cols, output_dim, kwargs = prepare_test(load_data, embedding_type, backbone_type)
    
    config = SwitchTabConfig(
        corruption_rate = 0.3,
        **kwargs
    )
    
    pl_module = SwitchTabLightning(config)
    
    pl_module.set_second_phase()
    test_ds = SwitchTabDataset(data, label, config=config, continuous_cols=continuous_cols, category_cols=category_cols, is_regression= True if output_dim == 1 else False, is_second_phase=True)
    test_dl = DataLoader(test_ds, 128, shuffle=False, sampler = SequentialSampler(test_ds))
    
    batch = next(iter(test_dl))
    
    print("Test The Second Phase Forward")
    pl_module._get_second_phase_loss(batch)
    print("Passed The Second Phase Forward")
    
if __name__ == "__main__":
    # test_switchtab_first_phase_forward(load_diabetes, "feature_tokenizer", "transformer")
    # test_switchtab_first_phase_forward(load_diabetes, "identity", "mlp")
    test_switchtab_first_phase_forward(load_cmc, "feature_tokenizer", "mlp")