# Config file

sweep_config = {'method': 'grid'}

# metrics
metric = {
    'name': 'val F1',
    'goal': 'maximize'
}

# constant parameters
parameters_dict = {
    'epochs': {
        'value': 10
    },
    'lr': {
        'value': 1e-05
    },
    'batch_size': {
        'value': 32
    },
    'data': {
        'value': 'MeldCSV/'
    },
    'seed': {
        'values': [42, 11, 100]
    }
}

sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict
