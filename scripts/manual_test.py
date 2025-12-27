import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from tqdm import tqdm
import glob
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def test():
    #Load the model
    model = torch.load("models/checkpoints/resnext_transformer2.pt", map_location=torch.device('cpu'))

    image_dirs = ["testset/Midjourneyv5-5K"]
    dir_labels = [1]

    for image_dir, label in zip(image_dirs, dir_labels):
        test_dataset = load_test_dataset(image_dir, label, limit = 600)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

        overall_acc, class_acc = predict_dataset(model, test_loader)
        print(f"Directory: {image_dir}, Overall Accuracy: {overall_acc:.2%}")

def simple_test():
    # Load the model
    model = torch.load("models/checkpoints/resnext.pt", map_location=torch.device('cpu'))
    images = ["testset/Midjourneyv5-5K/0.png", "testset/Midjourneyv5-5K/51.png", "testset/Midjourneyv5-5K/13.png", "testset/Midjourneyv5-5K/5416.png", "testset/Midjourneyv5-5K/5397.png"]

    for image_path in images:
        if not os.path.exists(image_path):
             print(f"Image not found: {image_path}")
             continue
        
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0)
        image = image.to(torch.device('cpu'))
        # Handle models that might be wrapped or raw
        if hasattr(model, 'model'):
             model.model.eval()
             net = model.model
        else:
             model.eval()
             net = model
             
        with torch.no_grad(), autocast():
            output = net(image)
            pred = output.argmax(dim=1).item()
            print(f"Image: {image_path}, Predicted Class: {pred}")

if __name__ == "__main__":
    # test()
    simple_test()




