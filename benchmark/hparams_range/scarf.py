hparams_range = {
        
        'hidden_dim' : ['suggest_int', ['hidden_dim', 16, 512]],
        'encoder_depth' : ['suggest_int', ['encoder_depth', 2, 6]],
        'head_depth' : ['suggest_int', ['head_depth', 1, 3]],
        'corruption_rate' : ['suggest_float', ['corruption_rate', 0.1, 0.7]],
        'dropout_rate' : ['suggest_float', ['dropout_rate', 0.05, 0.3]],

        'lr' : ['suggest_float', ['lr', 0.0001, 0.05]],
        'weight_decay' : ['suggest_float', ['weight_decay', 0.00001, 0.0005]],
    }