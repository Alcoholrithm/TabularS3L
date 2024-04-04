hparams_range = {
    'max_leaves' : ['suggest_int', ['max_leaves', 300, 4000]],
    'n_estimators' : ['suggest_int', ['n_estimators', 10, 3000]],
    'learning_rate' : ['suggest_float', ['learning_rate',0, 1]],
    'max_depth' : ['suggest_int', ['max_depth', 3, 20]],
    'scale_pos_weight' : ['suggest_int', ['scale_pos_weight', 1, 100]],
}