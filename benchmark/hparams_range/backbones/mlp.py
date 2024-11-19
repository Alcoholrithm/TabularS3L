hparams_range = {
        'hidden_dims' : ['suggest_int', ['hidden_dims', 16, 512]],
        'n_hiddens' : ['suggest_int', ['n_hiddens', 1, 4]],
        'use_batch_norm': ["suggest_categorical", ["use_batch_norm", [True, False]]],
        'dropout_rate' : ['suggest_float', ['dropout_rate', 0.05, 0.4]],
    }