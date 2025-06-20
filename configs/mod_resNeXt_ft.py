import numpy as np
import os

GENERAL_PATH = os.getcwd()

current_filename = os.path.splitext(os.path.basename(__file__))[0]

model_params = {
    'configs_file_name' : current_filename,
    'model_name' : 'resnext_real_vs_ai',
    'variant': "resnext101_32x8d",  # "resnext50_32x4d" or "resnext101_32x8d", ...
    'train_blocks': ["layer4"],
    'epochs': 50,
    'batch_size': 32,
    'lr_backbone': 1e-6,
    'lr_fc': 1e-6,
    'weight_decay': 1e-4,
    'early_stopping_patience': 15,
    'dropout': 0.6,
    'verbose': True,
}

dataset_params = {
    'data_train_path' : os.path.join(GENERAL_PATH, f'data/splits/train.csv'),
    'data_val_path' : os.path.join(GENERAL_PATH, f'data/splits/val.csv'),
    'data_test_path' : os.path.join(GENERAL_PATH, f'data/splits/test.csv'),
} 
