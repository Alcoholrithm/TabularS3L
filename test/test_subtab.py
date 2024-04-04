from misc import get_args

import sys
import os
here = os.path.dirname(__file__)

sys.path.append(os.path.join(here, '..'))

def test_subtab_classification():
    
    from benchmark.datasets import load_diabetes
    data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams = load_diabetes()
    
    sys.path.append(os.path.join(os.path.join(here, '..'), "benchmark"))
    
    from benchmark.pipelines import SubTabPipeLine
    
    args = get_args()
    
    pipeline = SubTabPipeLine(args, data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams)
    
    pipeline.benchmark()

def test_subtab_regression():
    
    from benchmark.datasets import load_abalone
    data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams = load_abalone()
    
    sys.path.append(os.path.join(os.path.join(here, '..'), "benchmark"))
    
    from benchmark.pipelines import SubTabPipeLine
    
    args = get_args()
    
    pipeline = SubTabPipeLine(args, data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams)
    
    pipeline.benchmark()
    
if __name__ == "__main__":
    test_subtab_classification()
    test_subtab_regression()