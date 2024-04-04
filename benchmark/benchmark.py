import argparse
from datasets import load_diabetes, load_abalone
from pipelines import VIMEPipeLine, SubTabPipeLine, SCARFPipeLine, XGBPipeLine

def main():
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('--model', type=str, choices=["xgb", "vime", "subtab", "scarf"])
    parser.add_argument('--data', type=str, choices=["diabetes", "abalone"])
    
    parser.add_argument('--labeled_sample_ratio', type=float, default=0.1)
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=0)
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--first_phase_patience', type=int, default=8)
    parser.add_argument('--second_phase_patience', type=int, default=16)
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--n_jobs', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=200)
    
    parser.add_argument('--accelerator', type=str, choices=["cuda", "cpu"])
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    
    parser.add_argument('--fast_dev_run', action="store_true")
    
    args = parser.parse_args()
    
    if args.accelerator == 'cpu':
        args.device = 'auto'
    
    if args.data == "diabetes":
        load_data = load_diabetes
    elif args.data == "abalone":
        load_data = load_abalone
        
    data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams = load_data()
    
    if args.model == "xgb":
        pipeline_class = XGBPipeLine
    elif args.model == "vime":
        pipeline_class = VIMEPipeLine
    elif args.model == "subtab":
        pipeline_class = SubTabPipeLine
    elif args.model == "scarf":
        pipeline_class = SCARFPipeLine
    
    if args.fast_dev_run:
        args.max_epochs = 1
        args.first_phase_patience = 1
        args.second_phase_patience = 1
        args.n_trials = 1
        
    pipeline = pipeline_class(args, data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams)
    
    pipeline.benchmark()

    
if __name__ == "__main__":
    main()