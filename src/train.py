import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as init
import numpy as np
import random

import sys
sys.path.append('D:\Documents\Machine Learning\Machine_Learning_PF')
from src.earlystopping import EarlyStopping

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For GPU
    torch.cuda.manual_seed_all(seed)  # For all GPUs if using multiple
    np.random.seed(seed)  # For NumPy
    random.seed(seed)  # For Python's random module

# Function to initialize model weights with He Initialization (Kaiming)
def initialize_weights(model):
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # He initialization for ReLU activations
            init.zeros_(layer.bias)  # Initialize bias to zero

# Function to create the model
def create_model(input_size, hidden_layers, output_size, activation, dropout_rate):
    layers = []
    in_features = input_size
    for hidden_layer in hidden_layers:
        layers.append(nn.Linear(in_features, hidden_layer))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU())
        elif activation == 'elu':
            layers.append(nn.ELU())
        elif activation == 'swish':
            layers.append(nn.SiLU())  # Swish activation
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        in_features = hidden_layer
    layers.append(nn.Linear(in_features, output_size))
    return nn.Sequential(*layers)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None, early_stopping_patience=None, print_progress=True):
    """
    Train a PyTorch model with optional learning rate scheduler, early stopping, and progress printing.

    Args:
        model: The PyTorch model to be trained.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer for training.
        num_epochs: Number of training epochs.
        device: Device to train on ('cpu' or 'cuda').
        scheduler: Optional learning rate scheduler (default: None).
        early_stopping_patience: Number of epochs with no improvement to stop training early (default: None).
        print_progress: Whether to print progress messages during training (default: True).

    Returns:
        model: Trained PyTorch model.
        history: Dictionary containing training and validation losses.
    """
    # Initialize seed and model weights
    set_seed(17)  # Use default seed value 42
    model = model.to(device)
    initialize_weights(model)  # Apply He initialization

    history = {'train_loss': [], 'val_loss': []}

    # Initialize early stopping if patience is provided
    early_stopping = None
    if early_stopping_patience is not None:
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            monitor='val_loss',
            mode='min',
            verbose=True,
        )

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            # Batch progress
            if print_progress and batch_idx % 10 == 0:  # Update progress every 10 batches
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        # Learning rate scheduler step
        if scheduler:
            scheduler.step(val_loss)
            if print_progress:
                current_lr = scheduler.optimizer.param_groups[0]['lr']  # Log current LR
                print(f"Learning Rate: {current_lr:.6f}")

        # Early stopping check
        if early_stopping is not None:
            early_stopping(val_loss, model)  # Call early stopping with val_loss
            if early_stopping.early_stop:  # Check if early stopping should trigger
                if print_progress:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                break

        if print_progress:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # If early stopping was used, load the best model weights
    if early_stopping and early_stopping.best_model_wts:
        print("Loading best model from early stopping.")
        early_stopping.load_best_model(model)

    return model, history


