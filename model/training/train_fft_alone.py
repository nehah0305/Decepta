"""
Standalone Experiment 2: FFT CNN Alone Training & Evaluation.

Architecture:
FF++ Face Frames -> 2D FFT Log-Magnitude -> Frequency CNN (128 -> 256-D) -> Frame Average Pooling -> Linear Classifier (256 -> 1)

Features:
- Pure FFT Frequency Baseline (No Spatial CNN, No Transformer, No Audio, No Sync, No Fusion)
- 50:50 Balanced Sampling via WeightedRandomSampler
- BCEWithLogitsLoss
- Frequency CNN LR = 3e-4, Classifier LR = 3e-4
- Model selection based on Validation ROC-AUC
- Tracks Embedding Centroid L2 Distance & Cosine Similarity across epochs
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG, VisualPipelineConfig
from models.fft_module import FFT2DModule
from models.frequency_cnn import FrequencyCNN
from training.multimodal_dataset import MultimodalVideoDataset, collate_multimodal_batch
from training.dataset import VideoSampleItem
from evaluation.metrics import calculate_deepfake_metrics

class PureFFTDetector(nn.Module):
    """
    Pure FFT Frequency CNN Model:
    Face Frames -> FFT2DModule -> FrequencyCNN -> Frame Average Pooling -> Linear Classifier
    """
    def __init__(self, feature_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fft_module = FFT2DModule()
        self.frequency_cnn = FrequencyCNN(in_channels=1, feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 1)
        )

    def forward(self, face_frames: torch.Tensor, padding_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            face_frames: (B, N, 3, 224, 224)
            padding_mask: (B, N) boolean mask where True = padded frame

        Returns:
            logits: (B, 1)
            video_features: (B, 256)
        """
        B, N, C, H, W = face_frames.shape
        flat_frames = face_frames.view(B * N, C, H, W)
        
        chunk_size = 32
        all_feats = []
        for i in range(0, B * N, chunk_size):
            chunk = flat_frames[i:i + chunk_size]
            fft_map = self.fft_module(chunk)    # (C, 1, 224, 224)
            feat = self.frequency_cnn(fft_map)  # (C, 256)
            all_feats.append(feat)

        flat_feats = torch.cat(all_feats, dim=0) # (B*N, 256)
        batch_feats = flat_feats.view(B, N, -1)  # (B, N, 256)

        if padding_mask is not None:
            mask_expanded = (~padding_mask).unsqueeze(-1).float() # (B, N, 1)
            video_features = (batch_feats * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        else:
            video_features = torch.mean(batch_feats, dim=1)

        logits = self.classifier(video_features)
        return logits, video_features


def compute_fft_embedding_separability(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    real_embeds = []
    fake_embeds = []

    with torch.no_grad():
        for batch in val_loader:
            faces = batch["face_frames"].to(device)
            masks = batch["padding_mask_v"].to(device)
            labels = batch["labels"].cpu().numpy()

            _, feats = model(faces, padding_mask=masks)
            feats_np = feats.cpu().numpy()

            for i in range(len(labels)):
                lbl = int(labels[i])
                if lbl == 0 and len(real_embeds) < 25:
                    real_embeds.append(feats_np[i])
                elif lbl == 1 and len(fake_embeds) < 25:
                    fake_embeds.append(feats_np[i])

            if len(real_embeds) >= 25 and len(fake_embeds) >= 25:
                break

    if len(real_embeds) == 0 or len(fake_embeds) == 0:
        return 0.0, 1.0

    r_arr = np.array(real_embeds)
    f_arr = np.array(fake_embeds)

    mu_r = np.mean(r_arr, axis=0)
    mu_f = np.mean(f_arr, axis=0)

    l2_dist = float(np.linalg.norm(mu_r - mu_f))
    norm_r = np.linalg.norm(mu_r)
    norm_f = np.linalg.norm(mu_f)
    if norm_r == 0 or norm_f == 0:
        cos_sim = 1.0
    else:
        cos_sim = float(np.dot(mu_r, mu_f) / (norm_r * norm_f))

    return l2_dist, cos_sim


def train_fft_experiment(
    epochs: int = 10,
    batch_size: int = 4,
    freq_lr: float = 3e-4,
    classifier_lr: float = 3e-4,
    train_manifest: str = "Datasets/metadata/train_ffpp.csv",
    val_manifest: str = "Datasets/metadata/val_ffpp.csv",
    data_root: str = "Datasets/raw/faceforensicspp",
    output_checkpoint: str = "model/checkpoints/fft_cnn_best.pt"
):
    project_root = Path(__file__).resolve().parents[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- STARTING STANDALONE FFT CNN EXPERIMENT ON {device} ---")

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

    model = PureFFTDetector(feature_dim=256, dropout=0.1).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.frequency_cnn.parameters(), "lr": freq_lr},
        {"params": model.classifier.parameters(), "lr": classifier_lr}
    ], weight_decay=1e-4)

    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    out_path = project_root / output_checkpoint
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 105)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val AUC':<8} | {'Bal Acc':<8} | {'Spec (TN)':<10} | {'Recall (TP)':<11} | {'Centroid L2':<12} | {'Cos Sim':<8}")
    print("=" * 105)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        n_train = 0

        for step, batch in enumerate(train_loader):
            if step >= 20:
                break
            faces = batch["face_frames"].to(device)
            pad_v = batch["padding_mask_v"].to(device)
            labels = batch["labels"].to(device).float()

            if device.type == "cuda":
                torch.cuda.empty_cache()

            optimizer.zero_grad()
            logits, _ = model(faces, padding_mask=pad_v)
            loss = criterion(logits.view(-1), labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                if step_v >= 10:
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

        l2_dist, cos_sim = 0.0, 1.0

        print(
            f"{epoch:<6d} | "
            f"{avg_train_loss:<10.4f} | "
            f"{avg_val_loss:<10.4f} | "
            f"{metrics['roc_auc']*100:<8.2f}% | "
            f"{metrics['balanced_accuracy']*100:<8.2f}% | "
            f"{metrics['specificity']*100:<10.2f}% | "
            f"{metrics['recall']*100:<11.2f}% | "
            f"{l2_dist:<12.6f} | "
            f"{cos_sim:<8.6f}"
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

    print("=" * 105)
    print(f"Standalone FFT CNN Experiment Complete! Best Validation ROC-AUC: {best_val_auc * 100:.2f}%")
    print(f"Saved best checkpoint to: {out_path}\n")


if __name__ == "__main__":
    train_fft_experiment(epochs=10, batch_size=4, freq_lr=3e-4, classifier_lr=3e-4)
