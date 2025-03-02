hparams_range = {
        "n_bin" : ["suggest_categorical", ["n_bin", [2, 5, 10, 20, 50, 100]]],
        "pretext_task" : ["suggest_categorical", ["pretext_task", ["BinRecon", "BinXent"]]],
        "decoder_dim" : ["suggest_categorical", ["decoder_dim", [128, 256, 512, 1024]]],
        "decoder_depth" : ["suggest_int", ["decoder_depth", 1, 5]],
        'dropout_rate' : ['suggest_float', ['dropout_rate', 0.05, 0.3]],

        'lr' : ['suggest_float', ['lr', 0.0001, 0.05]],
        'weight_decay' : ['suggest_float', ['weight_decay', 0.00001, 0.0005]],
    }