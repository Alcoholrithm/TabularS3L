hparams_range = {
        
        'hidden_dim' : ['suggest_int', ['hidden_dim', 16, 512]],
        'encoder_depth' : ['suggest_int', ['encoder_depth', 2, 6]],
        'head_depth' : ['suggest_int', ['head_depth', 1, 3]],
        
        "noise_ratio" : ["suggest_float", ["noise_ratio", 0.1, 0.3]],
        "noise_level" : ["suggest_float", ["noise_level", 0.5, 2]],
        "noise_type" : ["suggest_categorical", ["noise_type", ["Swap", "Gaussian", "Zero_Out"]]],
        "mask_loss_weight" : ["suggest_float", ["mask_loss_weight", 0.1, 5]],
        'dropout_rate' : ['suggest_float', ['dropout_rate', 0.05, 0.3]],

        'lr' : ['suggest_float', ['lr', 0.0001, 0.05]],
        'weight_decay' : ['suggest_float', ['weight_decay', 0.00001, 0.0005]],
    }