hparams_range = {
        'hidden_dim' : ['suggest_int', ['hidden_dim', 16, 512]],
        'encoder_depth' : ['suggest_int', ['encoder_depth', 1, 4]],
        'n_head' : ['suggest_int', ['n_head', 1, 4]],
        'ffn_factor' : ['suggest_float', ['ffn_factor', 1, 3]],
        'dropout_rate' : ['suggest_float', ['dropout_rate', 0.05, 0.4]],
    }