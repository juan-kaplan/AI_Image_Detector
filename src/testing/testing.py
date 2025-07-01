import torch
import torch.nn as nn
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
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

def predict_dataset(self, test_loader):
        """
        Evaluates the model on the given test_loader.
        Returns:
            overall_accuracy (float),
            class_accuracies (dict: class_idx -> accuracy)
        """
        self.model.eval()
        correct = 0
        total = 0
        class_correct = {}
        class_total = {}

        with torch.no_grad(), autocast():
            for imgs, lbls in tqdm(test_loader, desc="Testing", leave=False):
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                outputs = self.model(imgs)
                preds = outputs.argmax(dim=1)
                for label, pred in zip(lbls, preds):
                    total += 1
                    class_total[label.item()] = class_total.get(label.item(), 0) + 1
                    if label == pred:
                        correct += 1
                        class_correct[label.item()] = class_correct.get(label.item(), 0) + 1

        overall_accuracy = correct / total if total > 0 else 0.0
        class_accuracies = {
            cls: class_correct.get(cls, 0) / class_total.get(cls, 1)
            for cls in class_total
        }
        return overall_accuracy, class_accuracies

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, min_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform 
        self.min_size = min_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 1. Carga y conversión a RGB
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        # 2. Calculamos cuánto padding necesitamos
        w, h = image.size
        pad_w = max(0, self.min_size - w)
        pad_h = max(0, self.min_size - h)
        
        if pad_w > 0 or pad_h > 0:
            # Distribuimos el padding de forma equilibrada en ambos lados
            left   = pad_w // 2
            right  = pad_w - left
            top    = pad_h // 2
            bottom = pad_h - top
            
            # 3. Aplicamos padding por reflexión
            image = TF.pad(image, padding=(left, top, right, bottom), padding_mode='reflect')
        
        # 4. Transformaciones adicionales (resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

def load_test_dataset(image_dir, label, transform=None, limit=None):
    image_paths = sorted(
        [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
         if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    )
    if limit is not None:
        image_paths = image_paths[:limit]
    if transform is None:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    labels = [label] * len(image_paths)
    return ImageDataset(image_paths, labels, transform)

def predict_dataset(self, test_loader):
        """
        Evaluates the model on the given test_loader.
        Returns:
            overall_accuracy (float),
            class_accuracies (dict: class_idx -> accuracy)
        """
        self.model.eval()
        correct = 0
        total = 0
        class_correct = {}
        class_total = {}

        with torch.no_grad(), autocast():
            for imgs, lbls in tqdm(test_loader, desc="Testing", leave=False):
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                outputs = self.model(imgs)
                preds = outputs.argmax(dim=1)
                for label, pred in zip(lbls, preds):
                    total += 1
                    class_total[label.item()] = class_total.get(label.item(), 0) + 1
                    if label == pred:
                        correct += 1
                        class_correct[label.item()] = class_correct.get(label.item(), 0) + 1

        overall_accuracy = correct / total if total > 0 else 0.0
        class_accuracies = {
            cls: class_correct.get(cls, 0) / class_total.get(cls, 1)
            for cls in class_total
        }
        return overall_accuracy, class_accuracies

def test():
    #Load the model
    model = torch.load("resnext_transformer2.pt", map_location=torch.device('cpu'))

    image_dirs = ["testset/Midjourneyv5-5K"]
    dir_labels = [1]

    for image_dir, label in zip(image_dirs, dir_labels):
        test_dataset = load_test_dataset(image_dir, label, limit = 600)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

        overall_acc, class_acc = predict_dataset(model, test_loader)
        print(f"Directory: {image_dir}, Overall Accuracy: {overall_acc:.2%}")

def simple_test():
    # Load the model
    model = torch.load("resnext.pt", map_location=torch.device('cpu'))
    images = ["testset/Midjourneyv5-5K/0.png", "testset/Midjourneyv5-5K/51.png", "testset/Midjourneyv5-5K/13.png", "testset/Midjourneyv5-5K/5416.png", "testset/Midjourneyv5-5K/5397.png"]

    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0)
        image = image.to(torch.device('cpu'))
        model.model.eval()
        with torch.no_grad(), autocast():
            output = model.model(image)
            pred = output.argmax(dim=1).item()
            print(f"Image: {image_path}, Predicted Class: {pred}")

if __name__ == "__main__":
    # test()
    simple_test()




