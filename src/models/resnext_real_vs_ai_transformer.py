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
import torch.nn.functional as F



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

class HFDatasetWrapper(Dataset):
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
        sample = self.ds[idx]                 # {'image': PIL, 'label': int, ...}
        img = sample["image"]
        # force 3-channel RGB (in-place no-op if already RGB)
        img = img.convert("RGB")
        img = self.transform(img)
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
        if x.dim() == 2:
            B, N = x.shape
            d_model = self.cls_token.shape[-1]      # 2048 en su caso
            side = int((N // d_model) ** 0.5)       # H = W = 7
            x = x.view(B, d_model, side, side)         
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
        self.lr_on_plateau_patience = 5  # epochs with no improvement before reducing LR
        self.loss_weight    = 1.0          # weight for the positive class (AI images)
        self.batch_size_tr  = 32
        self.batch_size_va  = 64
        self.num_workers    = 4

        # layer control
        self.unfreeze_layers = ["layer4"]                 # default behaviour
        self.freeze_layers   = []                         # none explicitly

        self.encoder_blocks    = 6
        self.attention_heads   = 8
        self.mlp_expansion_factor = 4
        self.ckpt_path = "runs/model_resnext_real_vs_ai_transformer.pt"
        self.load_ckpt_seed = None
        
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
            self.lr_on_plateau_patience = model_params.get("lr_on_plateau_patience", self.lr_on_plateau_patience)
            self.loss_weight      = model_params.get("loss_weight",      self.loss_weight)


            self.batch_size_tr    = model_params.get("batch_size_tr",    self.batch_size_tr)
            self.batch_size_va    = model_params.get("batch_size_va",    self.batch_size_va)
            self.num_workers      = model_params.get("num_workers",      self.num_workers)

            self.unfreeze_layers  = model_params.get("unfreeze_layers",  self.unfreeze_layers)
            self.freeze_layers    = model_params.get("freeze_layers",    self.freeze_layers)

            # Transformer params
            self.encoder_blocks    = model_params.get("encoder_blocks",    self.encoder_blocks)
            self.attention_heads      = model_params.get("attention_heads",      self.attention_heads)
            self.mlp_expansion_factor        = model_params.get("mlp_expansion_factor",        self.mlp_expansion_factor) 
            self.ckpt_path    = model_params.get("ckpt_path", self.ckpt_path) #path del modelo resnext finetuneado
            self.load_ckpt_seed = model_params.get('load_ckpt_seed', None) #path de instancia de este objeto ya entrenado

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

        # filter out any head weights:
        backbone_state = {k: v for k, v in state.items() if not k.startswith("fc.")}

        # load only those, ignore missing/unexpected:
        missing, unexpected = self.model.load_state_dict(backbone_state, strict=False)
        print(f"[INFO] Backbone loaded; missing keys: {missing}\nunexpected keys: {unexpected}")
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

        if self.load_ckpt_seed:
            self.load_checkpoint(self.load_ckpt_seed, strict=True)

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
            self.optimizer, mode="min", factor=0.3, patience=self.lr_on_plateau_patience,
        )
        
        self.scaler = GradScaler()
        
        self.criterion = (
            self.loss_function
        )

    # ------------------------------------------------------------------ #
    #                              TRAINING                              #
    # ------------------------------------------------------------------ #
    def fit(self, train_loader, val_loader=None, save_name=None):
        best_val_acc = -float("inf")
        patience_cnt = 0
        prev_best_path = None
        for epoch in trange(self.num_epochs, desc="Epochs"):
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
                    torch.save({
                        "model":      self.model.state_dict(),
                        "optimizer":  self.optimizer.state_dict(),
                        "scheduler":  self.scheduler.state_dict(),
                        "scaler":     self.scaler.state_dict(),
                    }, new_path)
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
        for imgs, lbls in tqdm(loader, desc="Training", leave=False):
            imgs, lbls = imgs.to(self.device), lbls.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = self.model(imgs)
                loss = self.criterion(out, lbls, pos_weight=self.loss_weight)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += loss.item() * imgs.size(0)

        return running / len(loader.dataset)

    def _validate(self, loader):
        self.model.eval()
        loss_sum, correct = 0.0, 0
        with torch.no_grad(), autocast():
            for imgs, lbls in tqdm(loader, desc="Validating", leave=False):
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                out = self.model(imgs)
                loss_sum += self.criterion(out, lbls, pos_weight=self.loss_weight ).item() * imgs.size(0)
                preds = out.argmax(dim=1)
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
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        ds_obj = HFDatasetWrapper(split_or_path, transform)

        return DataLoader(ds_obj,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=True)

    def load_model_weights(self, ckpt_path: str, strict: bool = True):
        state = torch.load(ckpt_path, map_location=self.device)
        if 'model' in state:
            self.model.load_state_dict(state['model'], strict=strict)
        else:
            self.model.load_state_dict(state, strict=strict)
        self.model.to(self.device)
        
    def load_checkpoint(self, ckpt_path: str, strict: bool = True):
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # 2. optimiser / scheduler / scaler – only if they exist
        if "optimizer" in ckpt:
            self.model.load_state_dict(ckpt["model"], strict=strict)
            self._build_optim()                     # create them first
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.scaler.load_state_dict(ckpt["scaler"])
            for st in self.optimizer.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(self.device)
                    
        else:
            self.model.load_state_dict(ckpt, strict=strict)
        self.model.to(self.device)  
        
    def loss_function(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            pos_weight: float = 1.0,           # >1 ⇒ penalise fake-as-real more
            reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Weighted cross-entropy for 2-class (real=0, fake=1).

        `pos_weight` multiplies the contribution of the *fake* class (index 1).
        """
        # class-wise weights:   [real_weight, fake_weight]
        weight = torch.tensor([1.0, pos_weight],
                            device=logits.device,
                            dtype=logits.dtype)

        # make sure labels are long ints and 1-D
        targets = targets.long().view(-1)

        return F.cross_entropy(logits, targets,
                            weight=weight,
                            reduction=reduction) 
        
    def load_model_for_inference(self, ckpt_path, strict=True):
        state = torch.load(ckpt_path, map_location=self.device)
        
        if 'model' in state:
            self.model.load_state_dict(state["model"], strict=strict)
        else:
            self.model.load_state_dict(state, strict=strict)
        self.model.to(self.device)
        self.model.eval()
    
    def predict_image(self, image):
        """
        Predict the class of a single image (PIL.Image or file path).
        Returns the predicted class index (int).
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad(), autocast():
            output = self.model(img_tensor)
            pred = output.argmax(dim=1).item()
        return pred