hparams_range = {
        
    'hidden_dim' : ['suggest_int', ['hidden_dim', 16, 512]],
    
    'p_m' : ["suggest_float", ["p_m", 0.1, 0.9]],
    'alpha1' : ["suggest_float", ["alpha1", 0.1, 5]],
    'alpha2' : ["suggest_float", ["alpha2", 0.1, 5]],
    'beta' : ["suggest_float", ["beta", 0.1, 10]],
    'K' : ["suggest_int", ["K", 2, 20]],


    'lr' : ['suggest_float', ['lr', 0.0001, 0.05]],
    'weight_decay' : ['suggest_float', ['weight_decay', 0.00001, 0.0005]],
    }