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


class ImageCSVDataset(Dataset):
    def __init__(self, classes, csv_path, img_dir=None, transform=None):
        self.df = pd.read_csv(csv_path)
        if "image_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("El CSV debe tener columnas 'image_path' y 'label'.")
        self.img_dir = img_dir or ""
        self.transform = transform
        self.label_map = {cls: idx for idx, cls in enumerate(classes)}

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


class ResNeXtRealVsAITrainer:
    """
    Fine-tunes ResNeXt-50 32×4d for 2-class 'real vs AI' detection,
    letting the caller select which backbone layers stay frozen.
    """

    def __init__(self, model_params: dict | None = None):
        # ------------------ defaults ------------------------------------
        self.num_classes    = 2
        self.backbone_var   = "resnext50_32x4d"
        self.pretrained     = True
        self.dropout        = 0.5

        # training / optimisation
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr_backbone    = 1e-4
        self.lr_head        = 1e-3
        self.weight_decay   = 1e-4
        self.num_epochs     = 10
        self.patience       = 2

        self.batch_size_tr  = 32
        self.batch_size_va  = 64
        self.num_workers    = 4

        # layer control
        self.unfreeze_layers = ["layer4"]                 # default behaviour
        self.freeze_layers   = []                         # none explicitly

        # ------------------ overrides from dict -------------------------
        if model_params:
            self.num_classes      = model_params.get("num_classes",      self.num_classes)
            self.backbone_var     = model_params.get("backbone_var",     self.backbone_var)
            self.pretrained       = model_params.get("pretrained",       self.pretrained)
            self.dropout          = model_params.get("dropout",          self.dropout)

            self.device           = model_params.get("device",           self.device)
            self.lr_backbone      = model_params.get("lr_backbone",      self.lr_backbone)
            self.lr_head          = model_params.get("lr_head",          self.lr_head)
            self.weight_decay     = model_params.get("weight_decay",     self.weight_decay)
            self.num_epochs       = model_params.get("num_epochs",       self.num_epochs)
            self.patience         = model_params.get("patience",         self.patience)

            self.batch_size_tr    = model_params.get("batch_size_tr",    self.batch_size_tr)
            self.batch_size_va    = model_params.get("batch_size_va",    self.batch_size_va)
            self.num_workers      = model_params.get("num_workers",      self.num_workers)

            # NEW: layer control
            self.unfreeze_layers  = model_params.get("unfreeze_layers",  self.unfreeze_layers)
            self.freeze_layers    = model_params.get("freeze_layers",    self.freeze_layers)

        self.device = torch.device(self.device)
        self._build_model()
        self._build_optim()

    # ------------------------------------------------------------------ #
    #                         MODEL ARCHITECTURE                         #
    # ------------------------------------------------------------------ #
    def _build_model(self):
        """Create backbone, freeze / unfreeze as requested, replace head."""
        self.model = getattr(models, self.backbone_var)(
            weights="IMAGENET1K_V1" if self.pretrained else None
        )

        # --- 1. freeze everything ---------------------------------------
        for p in self.model.parameters():
            p.requires_grad_(False)

        # --- 2. unfreeze requested layers -------------------------------
        for layer_name in self.unfreeze_layers:
            module = getattr(self.model, layer_name, None)
            if module is None:
                raise ValueError(f"Layer '{layer_name}' not found in model")
            for p in module.parameters():
                p.requires_grad_(True)

        # --- 3. re-freeze any explicit 'freeze_layers' -------------------
        for layer_name in self.freeze_layers:
            module = getattr(self.model, layer_name, None)
            if module is None:
                raise ValueError(f"Layer '{layer_name}' not found in model")
            for p in module.parameters():
                p.requires_grad_(False)

        # --- 4. replace fc ----------------------------------------------
        in_feats = self.model.fc.in_features  # 2048 for ResNeXt-50
        self.model.fc = nn.Sequential(
            nn.Linear(in_feats, in_feats // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(in_feats // 2, self.num_classes),
            
        )
        # ensure head is trainable
        for p in self.model.fc.parameters():
            p.requires_grad_(True)

        self.model.to(self.device)

    # ------------------------------------------------------------------ #
    #                        OPTIMISER + SCHEDULER                       #
    # ------------------------------------------------------------------ #
    def _build_optim(self):
        # parameters that actually require grad
        backbone_trainable = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not n.startswith("fc")
        ]
        head_params = self.model.fc.parameters()

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_trainable, "lr": self.lr_backbone},
                {"params": head_params,        "lr": self.lr_head},
            ],
            weight_decay=self.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.3, patience=self.patience
        )
        
        self.scaler = GradScaler()
        
        self.criterion = (
            nn.BCEWithLogitsLoss() if self.num_classes == 1 else nn.CrossEntropyLoss()
        )

    # ------------------------------------------------------------------ #
    #                              TRAINING                              #
    # ------------------------------------------------------------------ #
    def fit(self, train_loader, val_loader=None, save_name=None):
        best_val_acc = -float("inf")
        patience_cnt = 0
        for epoch in range(self.num_epochs):
            train_loss = self._train_one_epoch(train_loader)
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
                self.scheduler.step(val_loss)
            print(
                f"Epoch {epoch+1:02d}/{self.num_epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_cnt = 0
                if save_name:
                    os.makedirs("runs", exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join("runs", f"model_resnext_real_vs_ai_{save_name}.pt"))
            else:
                patience_cnt += 1
                if self.patience and patience_cnt >= self.patience:
                    print("Early stopping: no improvement.")
                    break
        torch.cuda.empty_cache()

    # -- helpers --------------------------------------------------------- #
    def _train_one_epoch(self, loader):
        self.model.train()
        running = 0.0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(self.device), lbls.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = self.model(imgs)
                loss = self.criterion(out, lbls)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += loss.item() * imgs.size(0)

        return running / len(loader.dataset)

    def _validate(self, loader):
        self.model.eval()
        loss_sum, correct = 0.0, 0
        with torch.no_grad(), autocast():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                out = self.model(imgs)
                loss_sum += self.criterion(out, lbls).item() * imgs.size(0)
                preds = out.argmax(1)
                correct += (preds == lbls).sum().item()

        val_loss = loss_sum / len(loader.dataset)
        val_acc = correct / len(loader.dataset)
        return val_loss, val_acc

    def create_dataloader(self, classes, csv_path, img_dir=None, batch_size=32, num_workers=4, test=False, shuffle=True):
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if test:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        ds = ImageCSVDataset(classes, csv_path, img_dir, transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        return loader