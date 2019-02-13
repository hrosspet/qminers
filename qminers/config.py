CONFIG = {
    'TEST_RATIO': 0.2,
    'TIME_FORMAT': '%Y-%m-%d %H:%M:%S',
    'DATA_PATH': 'data/',
    'TRAIN_VAL_NAME': '/data_train_val.csv',
    'TEST_NAME': '/data_test.csv',
    # 'DEBUG': True,
    'DEBUG': False,
    'N_SPLITS': 10,
    'SEQUENCE_LENGTH': 10,
    'EPOCHS': 1000,
    'MODEL_PARAMS': {
        'batch_size': 32,
        'lr': 1e-4,
        'loss': 'mae',
        'output_dim': 1,
    }
}