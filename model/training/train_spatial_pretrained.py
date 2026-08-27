"""
Experiment 3 — ImageNet Pretrained Spatial ResNet-50 Baseline.

Stage A — Frozen Backbone:
- ResNet-50 with torchvision ImageNet weights (ResNet50_Weights.DEFAULT) -> FROZEN
- FC Classifier (2048 -> 256 -> 1) -> TRAINABLE
- WeightedRandomSampler (50:50 REAL/FAKE)
- BCEWithLogitsLoss
- Classifier LR = 1e-3
- 10 Epochs on FF++ train_ffpp.csv / val_ffpp.csv
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.models as models

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG
from training.multimodal_dataset import MultimodalVideoDataset, collate_multimodal_batch
from training.dataset import VideoSampleItem
from evaluation.metrics import calculate_deepfake_metrics


class PretrainedSpatialDetector(nn.Module):
    """
    ImageNet Pretrained ResNet-50 Spatial Detector:
    Backbone: torchvision resnet50(weights=DEFAULT)
    Classification Head: Conv / Pooling -> Linear(2048 -> 256) -> LayerNorm -> ReLU -> Linear(256 -> 1)
    """
    def __init__(self, feature_dim: int = 256, dropout: float = 0.1, freeze_backbone: bool = True):
        super().__init__()
        # Load torchvision ImageNet weights
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)

        # Remove existing fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # outputs (B, 2048, 1, 1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, face_frames: torch.Tensor, padding_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C, H, W = face_frames.shape
        flat_frames = face_frames.view(B * N, C, H, W)

        chunk_size = 32
        all_feats = []
        for i in range(0, B * N, chunk_size):
            chunk = flat_frames[i:i + chunk_size]
            cnn_out = self.backbone(chunk).squeeze(-1).squeeze(-1) # (C, 2048)
            proj = self.projection(cnn_out)                       # (C, 256)
            all_feats.append(proj)

        flat_feats = torch.cat(all_feats, dim=0) # (B*N, 256)
        batch_feats = flat_feats.view(B, N, -1)  # (B, N, 256)

        if padding_mask is not None:
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            video_features = (batch_feats * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        else:
            video_features = torch.mean(batch_feats, dim=1)

        logits = self.classifier(video_features)
        return logits, video_features


def train_pretrained_stage_a(
    epochs: int = 10,
    batch_size: int = 4,
    classifier_lr: float = 1e-3,
    train_manifest: str = "Datasets/metadata/train_ffpp.csv",
    val_manifest: str = "Datasets/metadata/val_ffpp.csv",
    data_root: str = "Datasets/raw/faceforensicspp",
    output_checkpoint: str = "model/checkpoints/spatial_pretrained_stage_a_best.pt"
):
    project_root = Path(__file__).resolve().parents[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- STARTING EXPERIMENT 3 STAGE A: PRETRAINED RESNET-50 (FROZEN BACKBONE) ON {device} ---")

    train_df = pd.read_csv(project_root / train_manifest)
    val_df = pd.read_csv(project_root / val_manifest)
    d_root = project_root / data_root

    def create_items(df, split):
        items = []
        for _, row in df.iterrows():
            rel_path = str(row["video_path"])
            full_path = d_root / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
            items.append(
                VideoSampleItem(
                    video_id=str(row.get("sample_id", Path(rel_path).stem)),
                    video_path=str(full_path),
                    label=int(row["label"]),
                    split=split
                )
            )
        return items

    train_items = create_items(train_df, "train")
    val_items = create_items(val_df, "validation")

    train_dataset = MultimodalVideoDataset(
        samples=train_items,
        coverage_ratio=DEFAULT_CONFIG.FRAME_COVERAGE_RATIO,
        min_frames=16,
        max_frames=16,
        face_size=DEFAULT_CONFIG.FACE_SIZE,
        device=DEFAULT_CONFIG.DEVICE
    )
    val_dataset = MultimodalVideoDataset(
        samples=val_items,
        coverage_ratio=DEFAULT_CONFIG.FRAME_COVERAGE_RATIO,
        min_frames=16,
        max_frames=16,
        face_size=DEFAULT_CONFIG.FACE_SIZE,
        device=DEFAULT_CONFIG.DEVICE
    )

    train_labels = [s.label for s in train_items]
    counts = np.bincount(train_labels)
    weights = 1.0 / np.maximum(counts, 1)
    sample_weights = [weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_multimodal_batch, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal_batch, num_workers=0)

    model = PretrainedSpatialDetector(feature_dim=256, dropout=0.1, freeze_backbone=True).to(device)

    # Train only trainable parameters (projection + classifier)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=classifier_lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    out_path = project_root / output_checkpoint
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val AUC':<8} | {'Bal Acc':<8} | {'Spec (TN)':<10} | {'Recall (TP)':<11}")
    print("=" * 90)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        n_train = 0

        for step, batch in enumerate(train_loader):
            if step >= 20: # 20 batches per epoch
                break
            faces = batch["face_frames"].to(device)
            pad_v = batch["padding_mask_v"].to(device)
            labels = batch["labels"].to(device).float()

            optimizer.zero_grad()
            logits, _ = model(faces, padding_mask=pad_v)
            loss = criterion(logits.view(-1), labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * faces.size(0)
            n_train += faces.size(0)

        avg_train_loss = running_train_loss / n_train if n_train > 0 else 0.0

        model.eval()
        running_val_loss = 0.0
        val_targets = []
        val_probs = []

        with torch.no_grad():
            for step_v, batch in enumerate(val_loader):
                if step_v >= 10: # 10 val batches per epoch
                    break
                faces = batch["face_frames"].to(device)
                pad_v = batch["padding_mask_v"].to(device)
                labels = batch["labels"].to(device).float()

                logits, _ = model(faces, padding_mask=pad_v)
                loss = criterion(logits.view(-1), labels)
                running_val_loss += loss.item() * faces.size(0)

                probs = torch.sigmoid(logits).view(-1).cpu().numpy()
                val_probs.extend(probs.tolist())
                val_targets.extend(labels.cpu().numpy().tolist())

        avg_val_loss = running_val_loss / len(val_targets) if val_targets else 0.0
        metrics = calculate_deepfake_metrics(np.array(val_targets), np.array(val_probs))

        print(
            f"{epoch:<6d} | "
            f"{avg_train_loss:<10.4f} | "
            f"{avg_val_loss:<10.4f} | "
            f"{metrics['roc_auc']*100:<8.2f}% | "
            f"{metrics['balanced_accuracy']*100:<8.2f}% | "
            f"{metrics['specificity']*100:<10.2f}% | "
            f"{metrics['recall']*100:<11.2f}%"
        )
        sys.stdout.flush()

        if metrics["roc_auc"] >= best_val_auc:
            best_val_auc = metrics["roc_auc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": best_val_auc,
                "metrics": metrics
            }, out_path)

    print("=" * 90)
    print(f"Pretrained ResNet-50 Stage A Complete! Best Validation ROC-AUC: {best_val_auc * 100:.2f}%")
    print(f"Saved best checkpoint to: {out_path}\n")


if __name__ == "__main__":
    train_pretrained_stage_a(epochs=10, batch_size=4, classifier_lr=1e-3)
