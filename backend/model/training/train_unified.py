"""
Unified Training Script for Decepta Deepfake Detection.

Supports:
- ResNet-50 Fine-Tuned (Stage B) — current champion
- Full Visual Model (Spatial + FFT + Gated Fusion + Temporal Transformer)
- Full Multimodal Model (Visual + Audio + Sync + Adaptive Fusion)

Includes all training best practices:
- Full epoch training (no batch caps)
- Cosine / Step / Plateau LR scheduling
- Data augmentation (flips, color jitter, erasing, blur)
- Mixed precision training (AMP)
- Gradient accumulation compatible
- WeightedRandomSampler for class balancing
- Early stopping with patience
- Comprehensive metrics logging to CSV
- Periodic + best-model checkpointing

Usage:
    python training/train_unified.py --config training/train_config.yaml
    python training/train_unified.py --config training/train_config.yaml --dry-run
"""

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as T
import yaml

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG
from training.dataset import VideoSampleItem
from training.dataset_adapter import DatasetAdapter
from training.multimodal_dataset import MultimodalVideoDataset, collate_multimodal_batch
from training.losses import DeepfakeDetectionLoss, MultimodalCompoundLoss
from evaluation.metrics import calculate_deepfake_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class ResNet50StageBModel(nn.Module):
    """
    ImageNet Pretrained ResNet-50 with configurable layer unfreezing.
    Supports: layer4_only, layer3_layer4, all
    """
    def __init__(
        self,
        feature_dim: int = 256,
        dropout: float = 0.1,
        unfreeze_strategy: str = "layer3_layer4"
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Apply freezing strategy
        self._apply_freeze_strategy(unfreeze_strategy)

        self.projection = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.classifier = nn.Linear(feature_dim, 1)

    def _apply_freeze_strategy(self, strategy: str):
        """Freeze layers based on strategy."""
        # First freeze everything
        for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in module.parameters():
                p.requires_grad = False

        # Then selectively unfreeze
        if strategy == "layer4_only":
            for p in self.layer4.parameters():
                p.requires_grad = True
        elif strategy == "layer3_layer4":
            for p in self.layer3.parameters():
                p.requires_grad = True
            for p in self.layer4.parameters():
                p.requires_grad = True
        elif strategy == "all":
            for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
                for p in module.parameters():
                    p.requires_grad = True
        else:
            raise ValueError(f"Unknown unfreeze strategy: {strategy}")

        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"ResNet-50 [{strategy}]: {trainable:,}/{total:,} trainable params "
                    f"({trainable/total*100:.1f}%)")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(
        self,
        face_frames: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C, H, W = face_frames.shape
        flat_frames = face_frames.view(B * N, C, H, W)

        chunk_size = 32
        all_feats = []
        for i in range(0, B * N, chunk_size):
            chunk = flat_frames[i:i + chunk_size]
            cnn_out = self.extract_features(chunk)
            proj = self.projection(cnn_out)
            all_feats.append(proj)

        flat_feats = torch.cat(all_feats, dim=0)
        batch_feats = flat_feats.view(B, N, -1)

        if padding_mask is not None:
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            video_features = (batch_feats * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        else:
            video_features = torch.mean(batch_feats, dim=1)

        logits = self.classifier(video_features)
        return logits, video_features


# =============================================================================
# AUGMENTATION BUILDER
# =============================================================================

def build_augmentation(aug_config: Dict[str, Any]) -> Optional[T.Compose]:
    """Build a torchvision transform pipeline from augmentation config."""
    if not aug_config.get("enabled", False):
        return None

    transforms = []

    # Horizontal flip
    flip_p = aug_config.get("horizontal_flip", 0.0)
    if flip_p > 0:
        transforms.append(T.RandomHorizontalFlip(p=flip_p))

    # Random rotation
    rotation = aug_config.get("random_rotation", 0)
    if rotation > 0:
        transforms.append(T.RandomRotation(degrees=rotation))

    # Color jitter
    cj = aug_config.get("color_jitter", {})
    if cj.get("enabled", False):
        transforms.append(T.ColorJitter(
            brightness=cj.get("brightness", 0.0),
            contrast=cj.get("contrast", 0.0),
            saturation=cj.get("saturation", 0.0),
            hue=cj.get("hue", 0.0)
        ))

    # Gaussian blur
    blur_p = aug_config.get("gaussian_blur", 0.0)
    if blur_p > 0:
        transforms.append(T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=blur_p))

    # Random erasing (operates on tensors, must be last)
    erase_p = aug_config.get("random_erasing", 0.0)
    if erase_p > 0:
        transforms.append(T.RandomErasing(p=erase_p, scale=(0.02, 0.15)))

    if not transforms:
        return None

    logger.info(f"Data augmentation: {len(transforms)} transforms enabled")
    return T.Compose(transforms)


# =============================================================================
# SCHEDULER BUILDER
# =============================================================================

def build_scheduler(optimizer: torch.optim.Optimizer, sched_config: Dict, epochs: int):
    """Build a learning rate scheduler from config."""
    stype = sched_config.get("type", "none").lower()

    if stype == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=sched_config.get("eta_min", 1e-7)
        )
        logger.info(f"Scheduler: CosineAnnealing (T_max={epochs}, eta_min={sched_config.get('eta_min', 1e-7)})")
    elif stype == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config.get("step_size", 10),
            gamma=sched_config.get("gamma", 0.1)
        )
        logger.info(f"Scheduler: StepLR (step={sched_config.get('step_size', 10)}, gamma={sched_config.get('gamma', 0.1)})")
    elif stype == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=sched_config.get("patience", 5),
            factor=sched_config.get("factor", 0.5),
            min_lr=sched_config.get("eta_min", 1e-7)
        )
        logger.info(f"Scheduler: ReduceLROnPlateau (patience={sched_config.get('patience', 5)})")
    else:
        scheduler = None
        logger.info("Scheduler: None")

    return scheduler


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Early stopping tracker based on validation AUC."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# =============================================================================
# METRICS LOGGER
# =============================================================================

class MetricsLogger:
    """Logs training metrics to CSV for analysis."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False

    def log(self, epoch: int, metrics: Dict[str, Any]):
        metrics["epoch"] = epoch
        mode = "a" if self._header_written else "w"
        with open(self.log_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(metrics)


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML training configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def load_datasets(cfg: Dict) -> Tuple[List[VideoSampleItem], List[VideoSampleItem], Optional[List[VideoSampleItem]]]:
    """Load train/val/test datasets from config."""
    ds_cfg = cfg["dataset"]
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / ds_cfg["data_root"]

    # Column mappings
    col_map = ds_cfg.get("columns", {})

    adapter = DatasetAdapter(data_root=data_root, column_map=col_map)

    auto_split = ds_cfg.get("auto_split", {})
    if auto_split.get("enabled", False):
        # Auto-split a single manifest
        manifest = project_root / ds_cfg["train_manifest"]
        train_items, val_items, test_items = adapter.auto_split_manifest(
            manifest,
            train_ratio=auto_split.get("train_ratio", 0.70),
            val_ratio=auto_split.get("val_ratio", 0.15),
            test_ratio=auto_split.get("test_ratio", 0.15),
            seed=auto_split.get("seed", 42)
        )
    else:
        # Separate manifest files
        train_manifest = project_root / ds_cfg["train_manifest"]
        val_manifest = project_root / ds_cfg["val_manifest"]

        train_items = adapter.load_manifest(train_manifest, split="train")
        val_items = adapter.load_manifest(val_manifest, split="val")

        test_items = None
        test_manifest_path = ds_cfg.get("test_manifest")
        if test_manifest_path:
            test_path = project_root / test_manifest_path
            if test_path.exists():
                test_items = adapter.load_manifest(test_path, split="test")

    return train_items, val_items, test_items


def build_model(cfg: Dict, device: torch.device) -> nn.Module:
    """Build the model based on config architecture selection."""
    model_cfg = cfg["model"]
    arch = model_cfg["architecture"]

    if arch == "resnet50_stage_b":
        resnet_cfg = model_cfg.get("resnet50", {})
        model = ResNet50StageBModel(
            feature_dim=model_cfg.get("feature_dim", 256),
            dropout=model_cfg.get("dropout", 0.1),
            unfreeze_strategy=resnet_cfg.get("unfreeze_strategy", "layer3_layer4")
        )

    elif arch == "visual_full":
        from models.visual_model import VisualDeepfakeDetector
        vis_cfg = model_cfg.get("visual", {})
        model = VisualDeepfakeDetector(
            spatial_dim=vis_cfg.get("spatial_dim", 256),
            frequency_dim=vis_cfg.get("frequency_dim", 256),
            fusion_hidden_dim=vis_cfg.get("fusion_hidden_dim", 128),
            fused_dim=vis_cfg.get("fused_dim", 256),
            transformer_dim=vis_cfg.get("transformer_dim", 768),
            transformer_heads=vis_cfg.get("transformer_heads", 8),
            transformer_layers=vis_cfg.get("transformer_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
            mode="full",
            frame_chunk_size=cfg.get("sampling", {}).get("frame_chunk_size", 16)
        )

    elif arch == "multimodal_full":
        from models.multimodal_detector import MultimodalDeepfakeDetector
        mm_cfg = model_cfg.get("multimodal", {})
        model = MultimodalDeepfakeDetector(
            visual_dim=mm_cfg.get("visual_dim", 768),
            audio_dim=mm_cfg.get("audio_dim", 768),
            sync_dim=mm_cfg.get("sync_dim", 256),
            fusion_dim=mm_cfg.get("fusion_dim", 768),
            mode="full",
            dropout=model_cfg.get("dropout", 0.1),
            frame_chunk_size=cfg.get("sampling", {}).get("frame_chunk_size", 16)
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}. "
                         f"Options: resnet50_stage_b, visual_full, multimodal_full")

    # Resume from checkpoint
    ckpt_path = model_cfg.get("resume_checkpoint")
    if ckpt_path:
        ckpt_path = Path(ckpt_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Resumed from checkpoint: {ckpt_path}")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path}, training from scratch")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {arch} | Total: {total_params:,} | Trainable: {trainable_params:,}")

    return model


def build_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Build optimizer with differential learning rates for pretrained models."""
    train_cfg = cfg["training"]
    arch = cfg["model"]["architecture"]

    if arch == "resnet50_stage_b":
        # Differential LR: backbone layers get lower LR
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(layer_name in name for layer_name in ["layer3", "layer4"]):
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": train_cfg.get("backbone_lr", 1e-5)},
            {"params": head_params, "lr": train_cfg.get("learning_rate", 1e-4)}
        ], weight_decay=train_cfg.get("weight_decay", 1e-4))

        logger.info(f"Optimizer: AdamW (backbone_lr={train_cfg.get('backbone_lr', 1e-5)}, "
                    f"head_lr={train_cfg.get('learning_rate', 1e-4)})")
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg.get("learning_rate", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-4)
        )
        logger.info(f"Optimizer: AdamW (lr={train_cfg.get('learning_rate', 1e-4)})")

    return optimizer


def build_dataloaders(
    train_items: List[VideoSampleItem],
    val_items: List[VideoSampleItem],
    cfg: Dict,
    augmentation: Optional[T.Compose] = None
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders."""
    train_cfg = cfg["training"]
    samp_cfg = cfg.get("sampling", {})
    arch = cfg["model"]["architecture"]

    dataset_kwargs = dict(
        coverage_ratio=samp_cfg.get("coverage_ratio", 0.70),
        min_frames=samp_cfg.get("min_frames", 16),
        max_frames=samp_cfg.get("max_frames", 32),
        face_size=samp_cfg.get("face_size", 224),
        device="cpu"
    )

    # Use MultimodalVideoDataset for all architectures (it provides face_frames,
    # mouth_crops, mel_windows, and padding masks)
    train_dataset = MultimodalVideoDataset(samples=train_items, **dataset_kwargs)
    val_dataset = MultimodalVideoDataset(samples=val_items, **dataset_kwargs)

    # Store augmentation on the dataset for application during __getitem__
    # (We'll apply it in the training loop since MultimodalVideoDataset
    #  doesn't have a built-in transform slot for face_frames)
    train_dataset._augmentation = augmentation

    # Build sampler
    sampler = None
    shuffle = True
    if train_cfg.get("use_weighted_sampler", True):
        train_labels = [s.label for s in train_items]
        counts = np.bincount(train_labels)
        if len(counts) >= 2 and counts.min() > 0:
            weights = 1.0 / counts.astype(float)
            sample_weights = [weights[l] for l in train_labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
            logger.info(f"WeightedRandomSampler: Real={counts[0]}, Fake={counts[1]}, "
                        f"Weights={weights.tolist()}")

    batch_size = train_cfg.get("batch_size", 4)
    num_workers = train_cfg.get("num_workers", 0)
    pin_memory = train_cfg.get("pin_memory", False) and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_multimodal_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_multimodal_batch,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info(f"DataLoaders: batch_size={batch_size}, workers={num_workers}, "
                f"train_batches={len(train_loader)}, val_batches={len(val_loader)}")

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    arch: str,
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_clip: float = 1.0,
    augmentation: Optional[T.Compose] = None,
    epoch: int = 0,
    total_epochs: int = 0
) -> Dict[str, float]:
    """Train for one full epoch."""
    model.train()
    running_loss = 0.0
    n_samples = 0
    all_targets = []
    all_probs = []

    total_batches = len(train_loader)
    log_interval = max(1, total_batches // 10)

    for step, batch in enumerate(train_loader):
        faces = batch["face_frames"].to(device)
        pad_v = batch["padding_mask_v"].to(device)
        labels = batch["labels"].to(device).float()

        # Apply augmentation to face frames if available
        if augmentation is not None:
            B, N, C, H, W = faces.shape
            flat = faces.view(B * N, C, H, W)
            flat = augmentation(flat)
            faces = flat.view(B, N, C, H, W)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                if arch == "resnet50_stage_b":
                    logits, _ = model(faces, padding_mask=pad_v)
                elif arch == "multimodal_full":
                    mouth_crops = batch["mouth_crops"].to(device)
                    mel_windows = batch["mel_windows"].to(device)
                    out = model(
                        face_frames=faces,
                        mouth_crops=mouth_crops,
                        mel_windows=mel_windows,
                        padding_mask_v=pad_v
                    )
                    logits = out.logits
                else:
                    # visual_full
                    out = model(faces, padding_mask=pad_v)
                    logits = out.logits
                loss = criterion(logits.view(-1), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if arch == "resnet50_stage_b":
                logits, _ = model(faces, padding_mask=pad_v)
            elif arch == "multimodal_full":
                mouth_crops = batch["mouth_crops"].to(device)
                mel_windows = batch["mel_windows"].to(device)
                out = model(
                    face_frames=faces,
                    mouth_crops=mouth_crops,
                    mel_windows=mel_windows,
                    padding_mask_v=pad_v
                )
                logits = out.logits
            else:
                out = model(faces, padding_mask=pad_v)
                logits = out.logits
            loss = criterion(logits.view(-1), labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        bs = faces.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

        probs = torch.sigmoid(logits).view(-1).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(labels.cpu().numpy().tolist())

        if (step + 1) % log_interval == 0 or step == total_batches - 1:
            avg_loss = running_loss / n_samples
            logger.info(
                f"  Epoch [{epoch}/{total_epochs}] Step [{step + 1}/{total_batches}] "
                f"Loss: {avg_loss:.4f}"
            )

    avg_loss = running_loss / n_samples if n_samples > 0 else 0.0

    # Compute training metrics
    train_metrics = {"train_loss": round(avg_loss, 4), "train_samples": n_samples}
    try:
        if len(np.unique(all_targets)) >= 2:
            from sklearn.metrics import roc_auc_score
            train_metrics["train_auc"] = round(float(roc_auc_score(all_targets, all_probs)), 4)
    except Exception:
        pass

    return train_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    arch: str
) -> Dict[str, float]:
    """Run full validation."""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for batch in val_loader:
        faces = batch["face_frames"].to(device)
        pad_v = batch["padding_mask_v"].to(device)
        labels = batch["labels"].to(device).float()

        if arch == "resnet50_stage_b":
            logits, _ = model(faces, padding_mask=pad_v)
        elif arch == "multimodal_full":
            mouth_crops = batch["mouth_crops"].to(device)
            mel_windows = batch["mel_windows"].to(device)
            out = model(
                face_frames=faces,
                mouth_crops=mouth_crops,
                mel_windows=mel_windows,
                padding_mask_v=pad_v
            )
            logits = out.logits
        else:
            out = model(faces, padding_mask=pad_v)
            logits = out.logits

        loss = criterion(logits.view(-1), labels)
        bs = faces.size(0)
        running_loss += loss.item() * bs

        probs = torch.sigmoid(logits).view(-1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(labels.cpu().numpy().tolist())

    n_samples = len(all_targets)
    avg_loss = running_loss / n_samples if n_samples > 0 else 0.0

    metrics = {"val_loss": round(avg_loss, 4), "val_samples": n_samples}

    try:
        if len(np.unique(all_targets)) >= 2:
            detailed = calculate_deepfake_metrics(np.array(all_targets), np.array(all_probs))
            metrics.update({
                "val_auc": detailed["roc_auc"],
                "val_accuracy": detailed["accuracy"],
                "val_balanced_acc": detailed["balanced_accuracy"],
                "val_precision": detailed["precision"],
                "val_recall": detailed["recall"],
                "val_specificity": detailed["specificity"],
                "val_f1": detailed["f1_score"],
                "val_pr_auc": detailed["pr_auc"],
            })
        else:
            metrics["val_auc"] = 0.5
            logger.warning("Validation set contains only one class — AUC set to 0.5")
    except Exception as e:
        metrics["val_auc"] = 0.5
        logger.warning(f"Metrics computation failed: {e}")

    return metrics


def run_training(cfg: Dict):
    """Main training orchestrator."""
    train_cfg = cfg["training"]
    output_cfg = cfg["output"]
    project_root = Path(__file__).resolve().parents[2]

    # Reproducibility
    set_seed(train_cfg.get("seed", 42))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} "
                    f"({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load datasets
    logger.info("=" * 80)
    logger.info("LOADING DATASETS")
    logger.info("=" * 80)
    train_items, val_items, test_items = load_datasets(cfg)

    # Build augmentation
    augmentation = build_augmentation(cfg.get("augmentation", {}))

    # Build model
    logger.info("=" * 80)
    logger.info("BUILDING MODEL")
    logger.info("=" * 80)
    model = build_model(cfg, device)
    arch = cfg["model"]["architecture"]

    # Build optimizer
    optimizer = build_optimizer(model, cfg)

    # Build scheduler
    epochs = train_cfg.get("epochs", 30)
    scheduler = build_scheduler(optimizer, train_cfg.get("scheduler", {}), epochs)

    # Build loss
    loss_cfg = train_cfg.get("loss", {})
    criterion = DeepfakeDetectionLoss(
        label_smoothing=loss_cfg.get("label_smoothing", 0.05),
        pos_weight=loss_cfg.get("pos_weight"),
        focal_gamma=loss_cfg.get("focal_gamma", 0.0)
    ).to(device)

    # Build dataloaders
    logger.info("=" * 80)
    logger.info("BUILDING DATALOADERS")
    logger.info("=" * 80)
    train_loader, val_loader = build_dataloaders(train_items, val_items, cfg, augmentation)

    # AMP setup
    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp) if use_amp else None

    # Early stopping
    es_cfg = train_cfg.get("early_stopping", {})
    early_stopper = EarlyStopping(
        patience=es_cfg.get("patience", 10),
        min_delta=es_cfg.get("min_delta", 0.001)
    ) if es_cfg.get("enabled", True) else None

    # Metrics logger
    exp_name = output_cfg.get("experiment_name", "train_run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{exp_name}_{arch}_{timestamp}"
    ckpt_dir = project_root / output_cfg.get("checkpoint_dir", "model/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_logger = None
    if output_cfg.get("save_metrics_log", True):
        log_path = ckpt_dir / f"{run_name}_metrics.csv"
        metrics_logger = MetricsLogger(log_path)
        logger.info(f"Metrics log: {log_path}")

    best_ckpt_path = ckpt_dir / f"{run_name}_best.pt"
    save_every = output_cfg.get("save_every_n_epochs", 5)

    # Training banner
    logger.info("=" * 80)
    logger.info(f"TRAINING: {arch.upper()}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {train_cfg.get('batch_size', 4)}")
    logger.info(f"  Train videos: {len(train_items)}")
    logger.info(f"  Val videos: {len(val_items)}")
    logger.info(f"  Total train batches/epoch: {len(train_loader)}")
    logger.info(f"  Augmentation: {'Enabled' if augmentation else 'Disabled'}")
    logger.info(f"  AMP: {'Enabled' if use_amp else 'Disabled'}")
    logger.info(f"  Checkpoint dir: {ckpt_dir}")
    logger.info("=" * 80)

    # Header for compact epoch summary table
    header = (
        f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | "
        f"{'Val AUC':<9} | {'Bal Acc':<9} | {'Recall':<9} | "
        f"{'Spec':<9} | {'F1':<9} | {'LR':<12} | {'Time':<8}"
    )
    logger.info(header)
    logger.info("=" * 110)

    best_val_auc = 0.0
    training_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            arch=arch,
            use_amp=use_amp,
            scaler=scaler,
            grad_clip=train_cfg.get("gradient_clip", 1.0),
            augmentation=augmentation,
            epoch=epoch,
            total_epochs=epochs
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, arch)

        # Scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            stype = train_cfg.get("scheduler", {}).get("type", "none").lower()
            if stype == "plateau":
                scheduler.step(val_metrics.get("val_auc", 0.5))
            else:
                scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log compact summary row
        val_auc = val_metrics.get("val_auc", 0.5)
        row = (
            f"{epoch:<6d} | "
            f"{train_metrics.get('train_loss', 0):<10.4f} | "
            f"{val_metrics.get('val_loss', 0):<10.4f} | "
            f"{val_auc * 100:<9.2f}% | "
            f"{val_metrics.get('val_balanced_acc', 0) * 100:<9.2f}% | "
            f"{val_metrics.get('val_recall', 0) * 100:<9.2f}% | "
            f"{val_metrics.get('val_specificity', 0) * 100:<9.2f}% | "
            f"{val_metrics.get('val_f1', 0) * 100:<9.2f}% | "
            f"{current_lr:<12.2e} | "
            f"{epoch_time:<8.1f}s"
        )
        logger.info(row)

        # Log metrics to CSV
        if metrics_logger:
            all_metrics = {**train_metrics, **val_metrics, "lr": current_lr, "epoch_time_s": round(epoch_time, 1)}
            metrics_logger.log(epoch, all_metrics)

        # Best model checkpoint
        if val_auc >= best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_val_auc,
                "val_metrics": val_metrics,
                "architecture": arch,
                "config": cfg
            }, best_ckpt_path)
            logger.info(f"  ★ New best model saved: AUC={best_val_auc * 100:.2f}% → {best_ckpt_path.name}")

        # Periodic checkpoint
        if save_every > 0 and epoch % save_every == 0:
            periodic_path = ckpt_dir / f"{run_name}_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "architecture": arch
            }, periodic_path)

        # Early stopping check
        if early_stopper is not None:
            if early_stopper.step(val_auc):
                logger.info(f"Early stopping triggered at epoch {epoch} "
                            f"(no improvement for {early_stopper.patience} epochs)")
                break

    # Training complete
    total_time = time.time() - training_start
    logger.info("=" * 110)
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"  Best Val ROC-AUC: {best_val_auc * 100:.2f}%")
    logger.info(f"  Total training time: {total_time / 60:.1f} minutes")
    logger.info(f"  Best checkpoint: {best_ckpt_path}")
    if metrics_logger:
        logger.info(f"  Metrics log: {metrics_logger.log_path}")
    logger.info("=" * 110)


def dry_run(cfg: Dict):
    """Quick validation of config without training."""
    logger.info("=" * 80)
    logger.info("DRY RUN — Validating configuration")
    logger.info("=" * 80)

    # Check dataset
    try:
        train_items, val_items, test_items = load_datasets(cfg)
        logger.info(f"✅ Datasets loaded: train={len(train_items)}, val={len(val_items)}, "
                    f"test={len(test_items) if test_items else 'N/A'}")
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        return

    # Check model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = build_model(cfg, device)
        logger.info(f"✅ Model built: {cfg['model']['architecture']}")
    except Exception as e:
        logger.error(f"❌ Model build failed: {e}")
        return

    # Check augmentation
    aug = build_augmentation(cfg.get("augmentation", {}))
    logger.info(f"✅ Augmentation: {'Enabled' if aug else 'Disabled'}")

    # Quick forward pass
    try:
        samp_cfg = cfg.get("sampling", {})
        n_frames = samp_cfg.get("min_frames", 16)
        face_size = samp_cfg.get("face_size", 224)
        dummy_input = torch.randn(1, n_frames, 3, face_size, face_size, device=device)
        dummy_mask = torch.zeros(1, n_frames, dtype=torch.bool, device=device)

        model.eval()
        with torch.no_grad():
            arch = cfg["model"]["architecture"]
            if arch == "resnet50_stage_b":
                logits, feats = model(dummy_input, padding_mask=dummy_mask)
                logger.info(f"✅ Forward pass: logits={logits.shape}, features={feats.shape}")
            elif arch == "visual_full":
                out = model(dummy_input, padding_mask=dummy_mask)
                logger.info(f"✅ Forward pass: logits={out.logits.shape}, "
                            f"video_feature={out.video_feature.shape}")
            else:
                logger.info(f"✅ Model instantiated (full forward pass skipped for multimodal)")
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        return

    logger.info("=" * 80)
    logger.info("✅ DRY RUN PASSED — Configuration is valid. Ready to train!")
    logger.info("=" * 80)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Decepta Unified Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training/train_unified.py --config training/train_config.yaml
  python training/train_unified.py --config training/train_config.yaml --dry-run
  python training/train_unified.py --config training/train_config.yaml --epochs 50
  python training/train_unified.py --config training/train_config.yaml --arch multimodal_full
"""
    )
    parser.add_argument("--config", type=str, default="training/train_config.yaml",
                        help="Path to YAML training config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config and do a test forward pass without training")
    # CLI overrides for common settings
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--arch", type=str, default=None,
                        choices=["resnet50_stage_b", "visual_full", "multimodal_full"],
                        help="Override model architecture")
    parser.add_argument("--train-manifest", type=str, default=None,
                        help="Override training manifest CSV path")
    parser.add_argument("--val-manifest", type=str, default=None,
                        help="Override validation manifest CSV path")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override dataset root directory")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Override experiment name for checkpoints")

    args = parser.parse_args()

    # Load base config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / args.config

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    cfg = load_config(str(config_path))

    # Apply CLI overrides
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.arch is not None:
        cfg["model"]["architecture"] = args.arch
    if args.train_manifest is not None:
        cfg["dataset"]["train_manifest"] = args.train_manifest
    if args.val_manifest is not None:
        cfg["dataset"]["val_manifest"] = args.val_manifest
    if args.data_root is not None:
        cfg["dataset"]["data_root"] = args.data_root
    if args.experiment_name is not None:
        cfg["output"]["experiment_name"] = args.experiment_name

    if args.dry_run:
        dry_run(cfg)
    else:
        run_training(cfg)


if __name__ == "__main__":
    main()
