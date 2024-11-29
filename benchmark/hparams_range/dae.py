hparams_range = {
        "noise_ratio" : ["suggest_float", ["noise_ratio", 0.1, 0.3]],
        "noise_level" : ["suggest_float", ["noise_level", 0.5, 2]],
        "noise_type" : ["suggest_categorical", ["noise_type", ["Swap", "Gaussian", "Zero_Out"]]],
        "mask_loss_weight" : ["suggest_float", ["mask_loss_weight", 0.1, 5]],
        'dropout_rate' : ['suggest_float', ['dropout_rate', 0.05, 0.3]],

        'lr' : ['suggest_float', ['lr', 0.0001, 0.05]],
        'weight_decay' : ['suggest_float', ['weight_decay', 0.00001, 0.0005]],
    }