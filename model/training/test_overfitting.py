"""
Master Pipeline Debugging & 20-Sample Tiny Overfitting Test Suite.

Checks 3-8 & Overfitting Tests:
1. Classifier logits check (shape, mean, std, sample values)
2. Classifier initial weights check (mean, std, bias)
3. Parameter update verification (weight diffs before and after optimizer step)
4. Input tensor normalization check (min, max, mean, std)
5. Input tensor pairwise difference check ((x[0] - x[1]).abs().mean())
6. Test A: Spatial CNN Overfitting Test (20 samples: 10 REAL + 10 FAKE) for 50 Epochs
7. Test B: FFT CNN Overfitting Test (20 samples: 10 REAL + 10 FAKE) for 50 Epochs
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG
from models.spatial_cnn import SpatialCNN
from models.fft_module import FFT2DModule
from models.frequency_cnn import FrequencyCNN
from training.multimodal_dataset import MultimodalVideoDataset, collate_multimodal_batch
from training.dataset import VideoSampleItem


class PureSpatialDetector(nn.Module):
    def __init__(self, feature_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.spatial_cnn = SpatialCNN(in_channels=3, feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 1)
        )

    def forward(self, face_frames: torch.Tensor, padding_mask: torch.Tensor = None):
        B, N, C, H, W = face_frames.shape
        flat_frames = face_frames.view(B * N, C, H, W)
        
        all_feats = []
        for i in range(0, B * N, 32):
            chunk = flat_frames[i:i + 32]
            feat = self.spatial_cnn(chunk)
            all_feats.append(feat)

        flat_feats = torch.cat(all_feats, dim=0)
        batch_feats = flat_feats.view(B, N, -1)

        if padding_mask is not None:
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            video_features = (batch_feats * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        else:
            video_features = torch.mean(batch_feats, dim=1)

        logits = self.classifier(video_features)
        return logits, video_features


class PureFFTDetector(nn.Module):
    def __init__(self, feature_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fft_module = FFT2DModule()
        self.frequency_cnn = FrequencyCNN(in_channels=1, feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 1)
        )

    def forward(self, face_frames: torch.Tensor, padding_mask: torch.Tensor = None):
        B, N, C, H, W = face_frames.shape
        flat_frames = face_frames.view(B * N, C, H, W)
        
        all_feats = []
        for i in range(0, B * N, 32):
            chunk = flat_frames[i:i + 32]
            fft_map = self.fft_module(chunk)
            feat = self.frequency_cnn(fft_map)
            all_feats.append(feat)

        flat_feats = torch.cat(all_feats, dim=0)
        batch_feats = flat_feats.view(B, N, -1)

        if padding_mask is not None:
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            video_features = (batch_feats * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        else:
            video_features = torch.mean(batch_feats, dim=1)

        logits = self.classifier(video_features)
        return logits, video_features


def run_overfitting_tests():
    project_root = Path(__file__).resolve().parents[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print("   DECEPTA MASTER DIAGNOSTIC & 20-SAMPLE OVERFITTING TEST SUITE")
    print("=" * 80)

    # 1. Load 10 Real + 10 Fake samples
    d_root = project_root / "Datasets/raw/faceforensicspp"
    val_df = pd.read_csv(project_root / "Datasets/metadata/val_ffpp.csv")
    
    real_10 = val_df[val_df["label"] == 0].head(10)
    fake_10 = val_df[val_df["label"] == 1].head(10)
    overfit_df = pd.concat([real_10, fake_10], ignore_index=True)

    items = [
        VideoSampleItem(
            video_id=Path(row["video_path"]).stem,
            video_path=str(d_root / row["video_path"]),
            label=int(row["label"]),
            split="overfit"
        )
        for _, row in overfit_df.iterrows()
    ]

    dataset = MultimodalVideoDataset(
        samples=items,
        min_frames=16,
        max_frames=16,
        face_size=224,
        device=DEFAULT_CONFIG.DEVICE
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_multimodal_batch)

    # -----------------------------------------------------------------
    # PIPELINE DIAGNOSTICS (CHECKS 3, 4, 5, 6, 8)
    # -----------------------------------------------------------------
    print("\n[CHECK 6 & 8] INPUT TENSOR NORMALIZATION & PAIRWISE DIFFERENCE:")
    sample_batch = next(iter(loader))
    x_faces = sample_batch["face_frames"].to(device)
    print(f"  Input Tensor Shape: {list(x_faces.shape)}")
    print(f"  Input Tensor Range: min={x_faces.min().item():.4f}, max={x_faces.max().item():.4f}")
    print(f"  Input Tensor Stats: mean={x_faces.mean().item():.4f}, std={x_faces.std().item():.4f}")
    
    if x_faces.size(0) >= 2:
        diff = (x_faces[0] - x_faces[1]).abs().mean().item()
        print(f"  Pairwise Batch Sample Difference (|x[0] - x[1]|): {diff:.4f}")
        assert diff > 0.001, "Input batch samples are identical!"
        print("  ✅ Input tensor non-identical check PASSED.")

    print("\n[CHECK 3, 4, 5] CLASSIFIER LOGITS & WEIGHT UPDATE VERIFICATION:")
    diag_model = PureSpatialDetector().to(device)
    
    # Pre-train weights check
    w_head = diag_model.classifier[1].weight
    print(f"  Initial Classifier Weight -> Mean: {w_head.mean().item():.6f}, Std: {w_head.std().item():.6f}")
    print(f"  Initial Classifier Bias   -> {diag_model.classifier[1].bias.detach().cpu().numpy()}")

    logits_pre, _ = diag_model(x_faces)
    print(f"  Forward Pass Logits (Batch of 4): {logits_pre.squeeze().detach().cpu().numpy()}")
    print(f"  Logits Stats -> Mean: {logits_pre.mean().item():.6f}, Std: {logits_pre.std().item():.6f}")

    # Parameter update check
    w_class_old = diag_model.classifier[1].weight.clone()
    w_cnn_first_old = diag_model.spatial_cnn.block1[0].weight.clone()
    w_cnn_last_old = diag_model.spatial_cnn.block4[0].weight.clone()

    optimizer = torch.optim.AdamW(diag_model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    loss = criterion(logits_pre.view(-1), sample_batch["labels"].to(device).float())
    loss.backward()
    optimizer.step()

    w_class_new = diag_model.classifier[1].weight.clone()
    w_cnn_first_new = diag_model.spatial_cnn.block1[0].weight.clone()
    w_cnn_last_new = diag_model.spatial_cnn.block4[0].weight.clone()

    diff_class = (w_class_new - w_class_old).abs().mean().item()
    diff_cnn_first = (w_cnn_first_new - w_cnn_first_old).abs().mean().item()
    diff_cnn_last = (w_cnn_last_new - w_cnn_last_old).abs().mean().item()

    print(f"  Weight Delta (Classifier Head): {diff_class:.6e}")
    print(f"  Weight Delta (First CNN Layer): {diff_cnn_first:.6e}")
    print(f"  Weight Delta (Final CNN Layer): {diff_cnn_last:.6e}")

    if diff_class > 0 and diff_cnn_first > 0 and diff_cnn_last > 0:
        print("  ✅ Parameter updates strictly verified across all layers!")
    else:
        print("  ❌ WARNING: Parameter updates failed!")

    # -----------------------------------------------------------------
    # TEST A: 20-SAMPLE OVERFITTING TEST FOR SPATIAL CNN
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("   TEST A: SPATIAL CNN OVERFITTING TEST (20 SAMPLES, 50 EPOCHS)")
    print("=" * 80)

    spatial_model = PureSpatialDetector().to(device)
    optimizer_sp = torch.optim.AdamW(spatial_model.parameters(), lr=3e-4)
    criterion_sp = nn.BCEWithLogitsLoss()

    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Accuracy':<15}")
    print("-" * 45)

    for epoch in range(1, 51):
        spatial_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            faces = batch["face_frames"].to(device)
            masks = batch["padding_mask_v"].to(device)
            labels = batch["labels"].to(device).float()

            optimizer_sp.zero_grad()
            logits, _ = spatial_model(faces, padding_mask=masks)
            loss = criterion_sp(logits.view(-1), labels)
            loss.backward()
            optimizer_sp.step()

            running_loss += loss.item() * faces.size(0)
            preds = (torch.sigmoid(logits.view(-1)) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += faces.size(0)

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0

        if epoch in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            print(f"{epoch:<8d} | {epoch_loss:<12.4f} | {epoch_acc:<14.2f}%")

    # -----------------------------------------------------------------
    # TEST B: 20-SAMPLE OVERFITTING TEST FOR FFT CNN
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("   TEST B: FFT CNN OVERFITTING TEST (20 SAMPLES, 50 EPOCHS)")
    print("=" * 80)

    fft_model = PureFFTDetector().to(device)
    optimizer_fft = torch.optim.AdamW(fft_model.parameters(), lr=3e-4)
    criterion_fft = nn.BCEWithLogitsLoss()

    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Accuracy':<15}")
    print("-" * 45)

    for epoch in range(1, 51):
        fft_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            faces = batch["face_frames"].to(device)
            masks = batch["padding_mask_v"].to(device)
            labels = batch["labels"].to(device).float()

            optimizer_fft.zero_grad()
            logits, _ = fft_model(faces, padding_mask=masks)
            loss = criterion_fft(logits.view(-1), labels)
            loss.backward()
            optimizer_fft.step()

            running_loss += loss.item() * faces.size(0)
            preds = (torch.sigmoid(logits.view(-1)) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += faces.size(0)

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0

        if epoch in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            print(f"{epoch:<8d} | {epoch_loss:<12.4f} | {epoch_acc:<14.2f}%")

    print("\n" + "=" * 80)
    print("   20-SAMPLE OVERFITTING DIAGNOSTIC COMPLETED")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_overfitting_tests()
