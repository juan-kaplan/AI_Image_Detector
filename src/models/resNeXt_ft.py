from typing import Optional, Tuple, List
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class ImageCSVDataset(Dataset):
    """Dataset que lee un .csv con las columnas `image_path` y `label`."""
    label_map = None

    def __init__(self, classes, csv_path, img_dir=None, transform=None):
        self.df = pd.read_csv(csv_path)
        if "image_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("El CSV debe tener columnas 'image_path' y 'label'.")
        self.img_dir = img_dir or ""
        self.transform = transform

        if ImageCSVDataset.label_map is None:
            ImageCSVDataset.label_map = {cls: idx for idx, cls in enumerate(classes)}

        self.label_map = ImageCSVDataset.label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = (
            os.path.join(self.img_dir, row["image_path"])
            if self.img_dir and not os.path.isabs(row["image_path"])
            else row["image_path"]
        )
        image = Image.open(path).convert("RGB")
        label = self.label_map[row["label"]]

        image = self.transform(image)
        return image, label


class ResNeXt_FT(nn.Module):
    def __init__(self,
                 num_classes: int = 2,  # Set default to 2 for binary classification
                 variant: str = "resnext50_32x4d",
                 device: str = None):
        """
        Args:
            num_classes: cantidad de clases finales (AI/Real).
            variant: "resnext50_32x4d" o "resnext101_32x8d".
            device: si no se pasa, elige "cuda:0" si GPU disponible.
        """
        super().__init__()
        self.num_classes = num_classes
        self.classes = None
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        # 1) Cargar ResNeXt preentrenado
        if variant == "resnext50_32x4d":
            weights = models.ResNeXt50_32X4D_Weights.DEFAULT
        elif variant == "resnext101_32x8d":
            weights = models.ResNeXt101_32X8D_Weights.DEFAULT
        else:
            raise ValueError(f"Variant desconocida: {variant}")

        # Cargamos el backbone completo con sus pesos ImageNet
        self.model = getattr(models, variant)(weights=weights)

        # 2) Reemplazar la capa final (fc) por Dropout + Linear
        #    Conservamos in_features original y lo mapeamos a num_classes.
        in_feats = self.model.fc.in_features
        # self.model.fc = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_feats, self.num_classes)
        # )

        red_dim = in_feats//2
        self.model.fc = nn.Sequential(
            nn.Linear(in_feats, red_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(red_dim, self.num_classes)
        )
        
        # Inicializamos pesos de la última capa lineal igual que en train
        nn.init.xavier_uniform_(self.model.fc[0].weight)
        self.model.fc[0].bias.data.zero_()

        # 3) Definir transformaciones de train / test
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # 4) Mover todo el modelo a device
        self.model.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def create_dataloader(self, name_classes, csv_path, img_dir=None, batch_size=32, num_workers=4, test=False, shuffle=True):
        """Crea DataLoader a partir de CSV ."""
        if test:
            ds = ImageCSVDataset(name_classes, csv_path, img_dir, self.test_transform)
        else:
            ds = ImageCSVDataset(name_classes, csv_path, img_dir, self.train_transform)

        if self.classes is None:
            inv_map = {v: k for k, v in ds.label_map.items()}
            self.classes = tuple(inv_map[i] for i in range(len(inv_map)))

        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,  num_workers=num_workers, pin_memory=True)
        return loader

    
    def train_model(self, train_path, val_path, name_classes, model_params=None, verbose=True, save_name=None):
        # --- Extracción de parámetros de configuración ---
        if model_params:
            train_blocks = model_params.get("train_blocks", ['layer4'])
            blocks       = model_params.get("blocks", {'layer4':[0,1,2]})   # índices de los sub-bloques
            epochs       = model_params.get("epochs", 50)
            batch_size   = model_params.get("batch_size", 32)
            lr_backbone  = model_params.get("lr_backbone", 1e-4)
            lr_fc        = model_params.get("lr_fc", 1e-3)
            weight_decay = model_params.get("weight_decay", 1e-5)
            early_stopping_patience = model_params.get("early_stopping_patience", 10)
            dropout      = model_params.get("dropout", 0.2)

        self.classes = name_classes
        self.class2idx = {c: i for i, c in enumerate(self.classes)}

        # --- 3. Congelar todos los parámetros inicialmente ---
        for p in self.model.parameters():
            p.requires_grad = False

        # --- 4. Descongelar solo los sub-bloques deseados ---
        unfrozen_params = []
        for layer_name in train_blocks:
            layer_module = getattr(self.model, layer_name)
            children = list(layer_module.children())
            for idx in blocks[layer_name]:
                real_idx = idx if idx >= 0 else len(children) + idx
                block = children[real_idx]
                for p in block.parameters():
                    p.requires_grad = True
                unfrozen_params.append(block.parameters())

        # --- 5. Asegurarnos de que la capa fc esté entrenable ---
        for p in self.model.fc.parameters():
            p.requires_grad = True
        unfrozen_params.append(self.model.fc.parameters())

        # --- 6. Preparar DataLoaders ---
        train_loader = self.create_dataloader(name_classes, train_path,
                                              batch_size=batch_size, num_workers=4, test=False)
        val_loader   = self.create_dataloader(name_classes, val_path,
                                              batch_size=batch_size, num_workers=4, test=True)

        # --- 7. Optimizador con distintos LR para backbone vs fc ---
        opt_groups = []
        for params in unfrozen_params[:-1]:
            opt_groups.append({"params": params, "lr": lr_backbone})
        opt_groups.append({"params": self.model.fc.parameters(), "lr": lr_fc})

        optimizer = torch.optim.AdamW(opt_groups, weight_decay=weight_decay)
        # Use BCEWithLogitsLoss for binary classification
        criterion = nn.BCEWithLogitsLoss()
        # Use ReduceLROnPlateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6, verbose=verbose)

        scaler = torch.cuda.amp.GradScaler()

        best_val_acc = -float("inf")
        patience_cnt = 0

        # --- 8. Ciclo de entrenamiento ---
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in tqdm(train_loader, disable=not verbose, desc=f"Epoch {epoch}/{epochs} [Train]"):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logits = self.model(images)
                    if self.num_classes == 2:
                        labels = labels.float().unsqueeze(1)
                        loss = criterion(logits, labels)
                        preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
                    else:
                        loss = criterion(logits, labels)
                        preds = logits.argmax(dim=1)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                # Liberar tensores temporales
                del images, labels, logits, loss

            train_loss = running_loss / len(train_loader.dataset)
            train_acc  = 100 * correct / total
            val_acc    = self._evaluate(val_loader)

            scheduler.step(val_acc)

            if verbose:
                print(f"\nEpoch {epoch:02d}/{epochs} | "
                      f"train loss: {train_loss:.4f} | "
                      f"train acc: {train_acc:.2f}% | "
                      f"val acc: {val_acc:.2f}%")

            # --- Early stopping y checkpoint ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_cnt = 0
                if save_name:
                    os.makedirs("runs", exist_ok=True)
                    torch.save(self.state_dict(), os.path.join("runs", f"model_resNeXt_ft_{save_name}.pt"))
            else:
                patience_cnt += 1
                if early_stopping_patience and patience_cnt >= early_stopping_patience:
                    if verbose:
                        print("Early stopping: no improvement.")
                    break

        # Limpiar caché de GPU
        torch.cuda.empty_cache()

    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images).argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100.0 * correct / total

    # ---------------------------------------------------------------------
    def  eval_model(self, test_path, results_folder=None):
        test_loader = self.create_dataloader(self.classes, test_path, test=True)
        
        self.model.eval()
        #correct = total = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images).argmax(1)
                all_preds.extend(preds.tolist())
                all_targets.extend(labels.tolist())
                # total += labels.size(0)
                # correct += (preds == labels).sum().item()



        report = classification_report(all_targets, all_preds, target_names=self.classes)
        print("\nClassification Report:\n", report)
    
        if results_folder is not None:
            # Plot confusion matrix
            plt.figure(figsize=(15, 12))
            cm = confusion_matrix(all_targets, all_preds)
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.classes,
                yticklabels=self.classes
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{results_folder}/confusion_matrix.png')

            report = classification_report(all_targets, all_preds, target_names=self.classes)
            report_file_path = f'{results_folder}/classification_report.txt'
            with open(report_file_path, 'w') as f:
                f.write(report)


    def load_params_model(self, weights_path: str, class_names: list):
        """
        Carga pesos guardados y define las clases asociadas.

        Args:
            weights_path (str): Ruta al archivo .pt o .pth del modelo.
            class_names (list): Lista de clases (str) en orden correcto.
        """
        self.classes = class_names
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        #self.dropout = nn.Dropout(p=0.2)  # Reasegura que exista para .forward()
        self.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.eval()
        print(f"Modelo cargado desde {weights_path}.")

    def test_model(self, test_csv: str, batch_size: int = 32, num_workers: int = 4):
        """
        Evalúa el modelo sobre un CSV de test (debe incluir columnas: image_path, label).

        Args:
            test_csv (str): Ruta al CSV.
            batch_size (int): Batch size.
            num_workers (int): Cantidad de workers del DataLoader.
        """
        df = pd.read_csv(test_csv)
        if not hasattr(self, 'class2idx'):
            self.classes = sorted(df['label'].unique())
            self.class2idx = {c: i for i, c in enumerate(self.classes)}
        test_loader = self.create_dataloader(self.classes, test_csv, batch_size=batch_size, shuffle=False, num_workers=num_workers, test=False)

        self.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                preds = self(images).argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        report = classification_report(all_labels, all_preds, target_names=self.classes)
        print("\n--- TEST REPORT ---\n")
        print(report)

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()