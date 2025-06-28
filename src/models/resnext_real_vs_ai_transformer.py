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


class ImageCSVDataset(Dataset):
    def __init__(self, classes, csv_path, img_dir=None, transform=None):
        self.df = pd.read_csv(csv_path)
        if "filepath" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("El CSV debe tener columnas 'filepath' y 'label'.")
        self.img_dir    = img_dir or ""
        self.transform  = transform
        self.label_map  = {cls: idx for idx, cls in enumerate(classes)}
        # keep the original class names for confusion matrix plotting
        self.classes    = list(classes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]

        path  = row["filepath"]

        image = Image.open(path).convert("RGB")
        label = self.label_map[row["label"]]
        generator = row['generator']

        if self.transform:
                image = self.transform(image)
        return image, label, path, generator

class HFDatasetWrapper(torch.utils.data.Dataset):
    """
    Thin wrapper that:
      • accepts a Hugging Face split (`datasets.Dataset`)
      • applies torchvision transforms on the fly
      • returns exactly (image_tensor, label_int)
    """
    def __init__(self, hf_split, transform):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]                     # {'image': PIL, 'label': int, ...}
        img = self.transform(sample["image"])
        return img, sample["label"]

class TransformerHead(nn.Module):
    """
    Converts the 7×7 feature map (B, C, H, W) to a sequence of tokens,
    runs a few self-attn layers, then classifies with a linear layer.
    """
    def __init__(self, d_model, n_cls, n_layers=2, n_heads=8,
                mlp_mult=4, dropout=0.1, pool='cls'):
        super().__init__()
        self.pool = pool
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_mult,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_cls)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):          
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        cls = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls, x), dim=1)    
        x = self.transformer(x)
        x = x[:, 0] if self.pool == 'cls' else x[:, 1:].mean(1)
        x = self.norm(x)
        return self.fc(x)

class ResNeXtRealVsAITransformer:
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
            self.patience         = model_params.get("early_stopping_patience",         self.patience)

            self.batch_size_tr    = model_params.get("batch_size_tr",    self.batch_size_tr)
            self.batch_size_va    = model_params.get("batch_size_va",    self.batch_size_va)
            self.num_workers      = model_params.get("num_workers",      self.num_workers)

            self.unfreeze_layers  = model_params.get("unfreeze_layers",  self.unfreeze_layers)
            self.freeze_layers    = model_params.get("freeze_layers",    self.freeze_layers)

            # Transformer params
            self.encoder_blocks    = model_params.get("encoder_blocks",    self.encoder_blocks)
            self.attention_heads      = model_params.get("attention_heads",      self.attention_heads)
            self.mlp_expansion_factor        = model_params.get("mlp_expansion_factor",        self.mlp_expansion_factor) 
            self.ckpt_path    = model_params.get("ckpt_path", self.ckpt_path)

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
        state = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)

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
        in_feats = self.model.fc.in_features        # 2048 for ResNeXt-50
        self.model.avgpool = nn.Identity()          # keep spatial grid
        
        # replaced fully connected layer with a transformer encoder layer, a normalization layer and a linear layer.
        self.model.fc = TransformerHead(
            d_model=in_feats,
            n_cls=self.num_classes,
            dropout=self.dropout,
            n_layers=self.encoder_blocks,  
            n_heads=self.attention_heads,   
            mlp_mult=self.mlp_expansion_factor,  
        )

        # make sure the head is trainable
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
        prev_best_path = None
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
                    new_path = os.path.join("runs", f"model_resnext_real_vs_ai_{save_name}_{val_acc:.4f}.pt")
                    # Delete previous best model if it exists
                    if prev_best_path and os.path.exists(prev_best_path):
                        os.remove(prev_best_path)
                    torch.save(self.model.state_dict(), new_path)
                    prev_best_path = new_path
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

    def create_dataloader(
            self,
            dataset_or_classes,
            split_or_path,        # accepts HF split or CSV path
            *,
            batch_size=32,
            num_workers=4,
            shuffle=True
    ):
        mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if isinstance(split_or_path, str):           # OLD behaviour — CSV
            ds_obj = ImageCSVDataset(dataset_or_classes, split_or_path,
                                    transform=transform)
        else:                                        # NEW behaviour — HF dataset
            ds_obj = HFDatasetWrapper(split_or_path, transform)

        return DataLoader(ds_obj,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=True)

    def load_weights(self, ckpt_path: str, strict: bool = True):
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state, strict=strict)
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, loader, save_path=None):
        """
        Evaluate the model, save predictions, a classification report, a confusion matrix,
        and a generator×prediction matrix (both CSV and heatmap).

        Args:
            loader (DataLoader): yields (imgs, labels, filepaths, generator)
            save_path (str, optional): directory to write outputs. Defaults to "results".

        Returns:
            report_str (str): the classification report
            df_preds (pd.DataFrame): dataframe of all predictions
            df_gen_matrix (pd.DataFrame): generator × pred_label matrix
        """
        # 1) Prepare output directory
        if save_path is None:
            save_path = "results"
        os.makedirs(save_path, exist_ok=True)

        # 2) Run inference
        self.model.eval()
        all_preds, all_labels, all_paths, all_gens = [], [], [], []
        with torch.no_grad(), autocast():
            for imgs, labels, paths, gens in loader:
                imgs = imgs.to(self.device)
                logits = self.model(imgs)
                preds = logits.argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_paths.extend(paths)
                all_gens.extend(gens)

        # 3) Save predictions CSV
        df_preds = pd.DataFrame({
            "filepath":   all_paths,
            "true_label": all_labels,
            "pred_label": all_preds,
            "generator":  all_gens
        })
        df_preds.to_csv(os.path.join(save_path, "predictions.csv"), index=False)

        # 4) Classification report
        report_str = classification_report(all_labels, all_preds, digits=4)
        with open(os.path.join(save_path, "classification_report.txt"), "w") as f:
            f.write(report_str)

        # 5) Overall confusion matrix (true vs. pred)
        cm = confusion_matrix(all_labels, all_preds)
        dataset    = getattr(loader, "dataset", None)
        class_names = getattr(dataset, "classes", None)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        if class_names is not None:
            ax.set(
                xticks=np.arange(len(class_names)),
                yticks=np.arange(len(class_names)),
                xticklabels=class_names,
                yticklabels=class_names,
                xlabel="Predicted label",
                ylabel="True label",
                title="Confusion Matrix"
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            ax.set(xlabel="Predicted label", ylabel="True label", title="Confusion Matrix")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, "confusion_matrix.png"))
        plt.close(fig)

        # 6) Generator × Predicted-label matrix
        #    Build a crosstab: rows = generators, columns = predicted labels
        mapped_preds = ["real" if p == 0 else "fake" for p in all_preds]
        df_gen = pd.DataFrame({
            "generator":  all_gens,
            "pred_label": mapped_preds
        })

        df_gen_matrix = pd.crosstab(df_gen["generator"], df_gen["pred_label"])

        # Save CSV
        df_gen_matrix.to_csv(os.path.join(save_path, "generator_prediction_matrix.csv"))

        # Plot heatmap
        fig, ax = plt.subplots()
        im = ax.imshow(df_gen_matrix.values, interpolation="nearest", cmap=plt.cm.Oranges)
        plt.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(df_gen_matrix.shape[1]),
            yticks=np.arange(df_gen_matrix.shape[0]),
            xticklabels=df_gen_matrix.columns,
            yticklabels=df_gen_matrix.index,
            xlabel="Predicted label",
            ylabel="Generator",
            title="Generator vs. Predicted-Label"
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Annotate counts
        thresh2 = df_gen_matrix.values.max() / 2.0
        for i in range(df_gen_matrix.shape[0]):
            for j in range(df_gen_matrix.shape[1]):
                count = df_gen_matrix.iat[i, j]
                ax.text(j, i, count, ha="center", va="center",
                        color="white" if count > thresh2 else "black")
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, "generator_prediction_matrix.png"))
        plt.close(fig)

        return report_str, df_preds, df_gen_matrix