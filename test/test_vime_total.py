from misc import get_args
import pytest

import sys
import os
here = os.path.dirname(__file__)

sys.path.append(os.path.join(here, '..'))

@pytest.mark.parametrize("embedding, backbone", [("identity", "mlp")])
def test_vime_classification(embedding, backbone):

    from benchmark.datasets import load_diabetes
    data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams = load_diabetes()
    
    sys.path.append(os.path.join(os.path.join(here, '..'), "benchmark"))
    
    from benchmark.pipelines import VIMEPipeLine
    
    args = get_args()
    args.embedding = embedding
    args.backbone = backbone
    
    pipeline = VIMEPipeLine(args, data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams)
    
    pipeline.benchmark()

@pytest.mark.parametrize("embedding, backbone", [("identity", "mlp")])
def test_vime_regression(embedding, backbone):

    from benchmark.datasets import load_abalone
    data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams = load_abalone()
    
    sys.path.append(os.path.join(os.path.join(here, '..'), "benchmark"))
    
    from benchmark.pipelines import VIMEPipeLine
    
    args = get_args()
    args.embedding = embedding
    args.backbone = backbone
    
    pipeline = VIMEPipeLine(args, data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams)
    
    pipeline.benchmark()
    
if __name__ == "__main__":
    test_vime_classification("identity", "mlp")
    test_vime_regression("identity", "mlp")