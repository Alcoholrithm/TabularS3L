from types import SimpleNamespace

def get_args():
    args = SimpleNamespace()
    
    args.embedding = "feature_tokenizer"
    args.embedding = "identity"
    
    args.backbone = "transformer"
    args.backbone = "mlp"
    
    args.max_epochs = 1
    args.first_phase_patience = 1
    args.second_phase_patience = 1
    args.n_trials = 1

    args.labeled_sample_ratio = 1
    args.valid_size = 0.2
    args.test_size = 0.2
    args.random_seed = 0
    args.batch_size = 128
    
    args.n_jobs = 4
    args.accelerator = "cpu"
    args.devices = "auto"
    
    return args

embeddings = ["identity", "feature_tokenizer"]