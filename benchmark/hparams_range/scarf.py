hparams_range = {
        'pretraining_head_dim' : ['suggest_int', ['pretraining_head_dim', 16, 512]],
        'head_depth' : ['suggest_int', ['head_depth', 1, 3]],
        'corruption_rate' : ['suggest_float', ['corruption_rate', 0.1, 0.7]],
        'dropout_rate' : ['suggest_float', ['dropout_rate', 0.05, 0.3]],

        'lr' : ['suggest_float', ['lr', 0.0001, 0.05]],
        'weight_decay' : ['suggest_float', ['weight_decay', 0.00001, 0.0005]],
    }