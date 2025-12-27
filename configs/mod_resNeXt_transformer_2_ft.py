import numpy as np
import os
import torch

GENERAL_PATH = os.getcwd()

current_filename = os.path.splitext(os.path.basename(__file__))[0]

model_params = {
    'configs_file_name' : current_filename,
    'model_name' : 'resnext_real_vs_ai_transformer',  # Name of the model
    'backbone_var': "resnext101_32x8d",  # "resnext50_32x4d" or "resnext101_32x8d", ...
    'unfreeze_layers': [],
    "freeze_layers": [], # Layers to freeze explicitly, if none specified all will be freezed except the unfreeze _layers
    'num_epochs': 5000,
    'batch_size_tr': 256, # Batch size for training
    'batch_size_va': 128, # Batch size for validation
    'lr_backbone': 1e-6,
    'lr_head': 1e-6,
    'weight_decay': 1e-4,
    'early_stopping_patience': 50,
    'lr_on_plateau_patience': 10,  # Number of epochs with no improvement after which learning rate will be reduced
    'loss_weight': 3,  # Weight for the positive class (AI images)
    'dropout': 0.6,
    'verbose': True,
    "num_classes": 2,  # Real vs AI
    "pretrained": True,  # Use pretrained weights,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,  # Number of workers for data loading
    "encoder_blocks": 6,  # Number of transformer encoder blocks
    "attention_heads": 8,  # Number of attention heads in the transformer
    "mlp_expansion_factor": 4,  # MLP expansion factor in the transformer
    "ckpt_path": "runs/model_resnext_real_vs_ai_mod_resNeXt_ft_0.9582.pt",  # Path to save the model checkpoint
    "load_ckpt_seed": 'runs\model_resnext_real_vs_ai_mod_resNeXt_transformer_2_ft_0.9423.pt'
}

dataset_params = {
    'data_train_path' : os.path.join(GENERAL_PATH, f'data/splits/train.csv'),
    'data_val_path' : os.path.join(GENERAL_PATH, f'data/splits/val.csv'),
    'data_test_path' : os.path.join(GENERAL_PATH, f'data/splits/test.csv'),
} 
