hparams_range = {
        
    'projection_dim' : ['suggest_int', ['projection_dim', 4, 1024]],
    
    'tau' : ["suggest_float", ["tau", 0.05, 0.15]],
    "use_cosine_similarity" : ["suggest_categorical", ["use_cosine_similarity", [True, False]]],
    "use_contrastive" : ["suggest_categorical", ["use_contrastive", [True, False]]],
    "use_distance" : ["suggest_categorical", ["use_distance", [True, False]]],
    
    "n_subsets" : ["suggest_int", ["n_subsets", 2, 7]],
    "overlap_ratio" : ["suggest_float", ["overlap_ratio", 0., 1]],
    
    "mask_ratio" : ["suggest_float", ["mask_ratio", 0.1, 0.3]],
    "noise_level" : ["suggest_float", ["noise_level", 0.5, 2]],
    "noise_type" : ["suggest_categorical", ["noise_type", ["Swap", "Gaussian", "Zero_Out"]]],

    'lr' : ['suggest_float', ['lr', 0.0001, 0.05]],
    'weight_decay' : ['suggest_float', ['weight_decay', 0.00001, 0.0005]],
    }