hparams_range = {
        'encoder_head_dim' : ['suggest_int', ['encoder_head_dim', 16, 256]],
        'encoder_depth' : ['suggest_int', ['encoder_depth', 1, 4]],
        'n_head' : ['suggest_int', ['n_head', 1, 4]],
        'ffn_factor' : ['suggest_int', ['ffn_factor', 1, 3]],
        'dropout_rate' : ['suggest_float', ['dropout_rate', 0.05, 0.3]],
        'lr' : ['suggest_float', ['lr', 0.0001, 0.05]],
        'weight_decay' : ['suggest_float', ['weight_decay', 0.00001, 0.0005]],
    }