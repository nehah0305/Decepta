"""
Experiment 6 — Spatial-Only Temporal Transformer Model.

Architecture:
- Spatial Branch: ImageNet ResNet-50 (Layers 1-2 Frozen, Layers 3-4 Fine-Tuned @ LR 1e-5) -> 256-D per frame
- NO FFT / Frequency Branch
- Positional Encoding for Frame Sequence
- Temporal Transformer Encoder: 2 Layers, 4 Heads, d_model=256, dim_feedforward=512, dropout=0.1
- Temporal Aggregation: Masked Mean pooling across unpadded Transformer sequence
- Classifier Head: Linear(256 -> 1) @ LR 1e-4
- Evaluation: Full 320-sample validation set every epoch (NO truncations/breaks)
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


class SpatialTemporalDetector(nn.Module):
    """
    Spatial (Fine-Tuned ResNet-50) + Multi-Head Temporal Transformer
    """
    def __init__(self, feature_dim: int = 256, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)

        # Spatial ResNet-50 Branch
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Freeze stem, layer1, layer2
        for m in [self.stem, self.layer1, self.layer2]:
            for p in m.parameters():
                p.requires_grad = False

        # Unfreeze layer3 and layer4
        for m in [self.layer3, self.layer4]:
            for p in m.parameters():
                p.requires_grad = True

        self.spatial_proj = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # Positional Encoding for Temporal Sequence (Max 64 frames)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 64, feature_dim))

        # Temporal Multi-Head Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim // 2, 1)
        )

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, face_frames: torch.Tensor, padding_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C, H, W = face_frames.shape
        flat_frames = face_frames.view(B * N, C, H, W)

        chunk_size = 32
        all_spatial = []

        for i in range(0, B * N, chunk_size):
            chunk = flat_frames[i:i + chunk_size]
            sp_out = self.extract_spatial_features(chunk)
            sp_proj = self.spatial_proj(sp_out)
            all_spatial.append(sp_proj)

        frame_seq = torch.cat(all_spatial, dim=0).view(B, N, -1) # (B, N, 256)

        # Add Positional Encoding
        seq_len = min(N, 64)
        frame_seq = frame_seq[:, :seq_len, :] + self.pos_encoder[:, :seq_len, :]

        # Pass through Multi-Head Temporal Transformer
        if padding_mask is not None:
            pad_mask_sliced = padding_mask[:, :seq_len]
            trans_seq = self.transformer_encoder(frame_seq, src_key_padding_mask=pad_mask_sliced)
            
            mask_expanded = (~pad_mask_sliced).unsqueeze(-1).float()
            video_embedding = (trans_seq * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        else:
            trans_seq = self.transformer_encoder(frame_seq)
            video_embedding = torch.mean(trans_seq, dim=1)

        logits = self.classifier(video_embedding)
        return logits, video_embedding


def train_spatial_temporal(
    epochs: int = 10,
    batch_size: int = 4,
    spatial_lr: float = 1e-5,
    transformer_lr: float = 1e-4,
    classifier_lr: float = 1e-4,
    train_manifest: str = "Datasets/metadata/train_ffpp.csv",
    val_manifest: str = "Datasets/metadata/val_ffpp.csv",
    data_root: str = "Datasets/raw/faceforensicspp",
    output_checkpoint: str = "model/checkpoints/spatial_temporal_best.pt"
):
    project_root = Path(__file__).resolve().parents[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- STARTING EXPERIMENT 6: SPATIAL-ONLY TEMPORAL TRANSFORMER ON {device} ---")

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

    model = SpatialTemporalDetector(feature_dim=256, num_heads=4, num_layers=2, dropout=0.1).to(device)

    # Differential Learning Rates
    spatial_params = list(model.layer3.parameters()) + list(model.layer4.parameters()) + list(model.spatial_proj.parameters())
    transformer_params = [model.pos_encoder] + list(model.transformer_encoder.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": spatial_params, "lr": spatial_lr},
        {"params": transformer_params, "lr": transformer_lr},
        {"params": classifier_params, "lr": classifier_lr}
    ], weight_decay=1e-4)

    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    out_path = project_root / output_checkpoint
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 95)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val AUC':<8} | {'PR-AUC':<8} | {'Bal Acc':<8} | {'Spec (TN)':<10} | {'Recall (TP)':<11}")
    print("=" * 95)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        n_train = 0

        for step, batch in enumerate(train_loader):
            if step >= 20: # 20 training steps per epoch for fast iteration
                break
            faces = batch["face_frames"].to(device)
            pad_v = batch["padding_mask_v"].to(device)
            labels = batch["labels"].to(device).float()

            optimizer.zero_grad()
            logits, _ = model(faces, padding_mask=pad_v)
            loss = criterion(logits.view(-1), labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * faces.size(0)
            n_train += faces.size(0)

        avg_train_loss = running_train_loss / n_train if n_train > 0 else 0.0

        # FULL 320-SAMPLE VALIDATION EVALUATION (NO TRUNCATION BREAK)
        model.eval()
        running_val_loss = 0.0
        val_targets = []
        val_probs = []

        with torch.no_grad():
            for batch in val_loader:
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
            f"{metrics['pr_auc']*100:<8.2f}% | "
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

    print("=" * 95)
    print(f"Experiment 6 (Spatial + Temporal) Complete! Best Validation ROC-AUC: {best_val_auc * 100:.2f}%")
    print(f"Saved best checkpoint to: {out_path}\n")


if __name__ == "__main__":
    train_spatial_temporal(epochs=10, batch_size=4, spatial_lr=1e-5, transformer_lr=1e-4, classifier_lr=1e-4)
