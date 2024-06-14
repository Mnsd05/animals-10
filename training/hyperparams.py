hyperparams = {
    "num_epochs": 2,
    "num_trials": 2,
    'optimizers': 
    {
        'adam':
        {
            'lr': (1e-5, 1e-1),
            'beta1': (0.8, 0.999),
            'beta2': (0.8, 0.999),
        },
        'nadam':
        {
            'lr': (1e-5, 1e-1),
            'beta1': (0.8, 0.999),
            'beta2': (0.8, 0.999),
        },
    },
}
