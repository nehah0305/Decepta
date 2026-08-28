"""
Experiment 6B — Controlled Spatial-Only Temporal Transformer Model with Full-Dataset Optimization.

Configuration:
- Spatial Branch: ImageNet ResNet-50 (Layers 1-2 Frozen, Layers 3-4 Fine-Tuned @ initial LR 1e-5) -> 256-D per frame
- Positional Encoding: Sequence length up to 64 frames
- Temporal Transformer Encoder: 2 Layers, 4 Heads, d_model=256, dim_feedforward=512, dropout=0.1 (initial LR 1e-4)
- Temporal Aggregation: Masked Mean pooling across unpadded Transformer sequence
- Classifier Head: Linear(256 -> 128) -> ReLU -> Dropout -> Linear(128 -> 1) (initial LR 1e-4)
- Optimization: Full dataset per epoch (no step limit), 25 Epochs, CosineAnnealingLR (T_max=25, eta_min=1e-6)
- Evaluation: Full 320-sample validation set every epoch
- Loss: BCEWithLogitsLoss, WeightedRandomSampler
"""

import io
import os
import sys
import time
from pathlib import Path
from typing import Tuple

# Force UTF-8 stdout on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

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

        frame_seq = torch.cat(all_spatial, dim=0).view(B, N, -1)  # (B, N, 256)

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


def train_spatial_temporal_6b(
    epochs: int = 25,
    batch_size: int = 4,
    spatial_lr: float = 1e-5,
    transformer_lr: float = 1e-4,
    classifier_lr: float = 1e-4,
    min_lr: float = 1e-6,
    train_manifest: str = "Datasets/metadata/train_ffpp.csv",
    val_manifest: str = "Datasets/metadata/val_ffpp.csv",
    data_root: str = "Datasets/raw/faceforensicspp/c23",
    output_checkpoint: str = "model/checkpoints/spatial_temporal_best.pt"
):
    project_root = Path(__file__).resolve().parents[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- STARTING EXPERIMENT 6B: FULL-DATASET SPATIAL-TEMPORAL OPTIMIZATION ON {device} ---")

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

    total_train_steps = len(train_loader)
    total_val_steps = len(val_loader)

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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=min_lr
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    out_path = project_root / output_checkpoint
    out_path.parent.mkdir(parents=True, exist_ok=True)

    LOG_PATH = project_root / "model" / "spatial_temporal_training.log"
    log_file = open(LOG_PATH, "w", buffering=1, encoding="utf-8")

    def log(msg: str):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    def fmt_time(seconds: float) -> str:
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        elif m > 0:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    header = (
        f"\n{'='*110}\n"
        f"EXPERIMENT 6B: FULL-DATASET SPATIAL-TEMPORAL RESNET-50 + TRANSFORMER\n"
        f"Epochs: {epochs} | Batch Size: {batch_size} | Sampler: WeightedRandomSampler\n"
        f"Train Batches: {total_train_steps} | Val Batches: {total_val_steps}\n"
        f"Initial LRs -> Spatial: {spatial_lr:.1e} | Transformer: {transformer_lr:.1e} | Classifier: {classifier_lr:.1e}\n"
        f"Scheduler: CosineAnnealingLR (T_max={epochs}, eta_min={min_lr:.1e})\n"
        f"{'='*110}"
    )
    log(header)

    run_start = time.time()
    epoch_times: list = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0
        n_train = 0

        current_sp_lr = optimizer.param_groups[0]["lr"]
        current_tr_lr = optimizer.param_groups[1]["lr"]

        log(f"\n[Epoch {epoch:>2}/{epochs}] -- TRAINING (Total: {total_train_steps} steps | Spatial LR: {current_sp_lr:.2e}, Trans LR: {current_tr_lr:.2e}) ---")
        step_times: list = []

        for step, batch in enumerate(train_loader):
            step_start = time.time()
            faces = batch["face_frames"].to(device)
            pad_v  = batch["padding_mask_v"].to(device)
            labels = batch["labels"].to(device).float()

            optimizer.zero_grad()
            logits, _ = model(faces, padding_mask=pad_v)
            loss = criterion(logits.view(-1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * faces.size(0)
            n_train += faces.size(0)

            step_elapsed = time.time() - step_start
            step_times.append(step_elapsed)
            avg_step_t = sum(step_times[-50:]) / len(step_times[-50:])
            remaining_steps = total_train_steps - (step + 1)
            epoch_eta_train = avg_step_t * remaining_steps

            # Log every 50 steps, or early steps, or the last step
            if (step + 1) <= 10 or (step + 1) % 50 == 0 or (step + 1) == total_train_steps:
                log(
                    f"  Train Step [{step+1:>4}/{total_train_steps}] "
                    f"| Loss: {loss.item():.4f} "
                    f"| Avg Loss: {(running_train_loss / n_train):.4f} "
                    f"| Step: {step_elapsed:.2f}s "
                    f"| Epoch ETA (train): {fmt_time(epoch_eta_train)}"
                )

        avg_train_loss = running_train_loss / n_train if n_train > 0 else 0.0
        train_elapsed = time.time() - epoch_start
        log(f"\n  [OK] Training done in {fmt_time(train_elapsed)} | Avg Train Loss: {avg_train_loss:.4f}")

        # FULL 320-SAMPLE VALIDATION
        log(f"\n[Epoch {epoch:>2}/{epochs}] -- VALIDATION ({len(val_items)} samples, {total_val_steps} batches) ----------------")
        model.eval()
        running_val_loss = 0.0
        val_targets: list = []
        val_probs: list   = []
        val_step = 0
        val_step_times: list = []

        with torch.no_grad():
            for batch in val_loader:
                vstep_start = time.time()
                faces  = batch["face_frames"].to(device)
                pad_v  = batch["padding_mask_v"].to(device)
                labels = batch["labels"].to(device).float()

                logits, _ = model(faces, padding_mask=pad_v)
                loss = criterion(logits.view(-1), labels)
                running_val_loss += loss.item() * faces.size(0)

                probs = torch.sigmoid(logits).view(-1).cpu().numpy()
                val_probs.extend(probs.tolist())
                val_targets.extend(labels.cpu().numpy().tolist())

                vstep_elapsed = time.time() - vstep_start
                val_step_times.append(vstep_elapsed)
                val_step += 1
                avg_vt = sum(val_step_times[-20:]) / len(val_step_times[-20:])
                val_eta = avg_vt * (total_val_steps - val_step)

                if (val_step <= 5) or (val_step % 20 == 0) or (val_step == total_val_steps):
                    log(
                        f"  Val  Step [{val_step:>3}/{total_val_steps}] "
                        f"| Loss: {loss.item():.4f} "
                        f"| Step: {vstep_elapsed:.2f}s "
                        f"| Val ETA: {fmt_time(val_eta)}"
                    )

        avg_val_loss = running_val_loss / len(val_targets) if val_targets else 0.0
        metrics = calculate_deepfake_metrics(np.array(val_targets), np.array(val_probs))

        # Step LR scheduler after epoch completion
        scheduler.step()

        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        avg_epoch_t = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - epoch
        overall_eta = avg_epoch_t * remaining_epochs

        saved_marker = " ** SAVED BEST **" if metrics["roc_auc"] >= best_val_auc else ""
        summary = (
            f"\n{'-'*110}\n"
            f"  EPOCH {epoch:>2}/{epochs} SUMMARY\n"
            f"    Spatial LR : {current_sp_lr:.2e} | Trans LR: {current_tr_lr:.2e}\n"
            f"    Train Loss : {avg_train_loss:.4f}\n"
            f"    Val   Loss : {avg_val_loss:.4f}\n"
            f"    ROC-AUC    : {metrics['roc_auc']*100:.2f}%\n"
            f"    PR-AUC     : {metrics['pr_auc']*100:.2f}%\n"
            f"    Bal Acc    : {metrics['balanced_accuracy']*100:.2f}%\n"
            f"    Specificity: {metrics['specificity']*100:.2f}%\n"
            f"    Recall     : {metrics['recall']*100:.2f}%\n"
            f"    Epoch Time : {fmt_time(epoch_elapsed)}\n"
            f"    Overall ETA: {fmt_time(overall_eta)} ({remaining_epochs} epoch(s) left){saved_marker}\n"
            f"{'-'*110}"
        )
        log(summary)

        if metrics["roc_auc"] >= best_val_auc:
            best_val_auc = metrics["roc_auc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": best_val_auc,
                "metrics": metrics,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            }, out_path)

    total_time = time.time() - run_start
    final = (
        f"\n{'='*110}\n"
        f"Experiment 6B (Spatial + Temporal - Full Dataset + CosineAnnealingLR) -- COMPLETE\n"
        f"  Best Val ROC-AUC : {best_val_auc * 100:.2f}%\n"
        f"  Total Run Time   : {fmt_time(total_time)}\n"
        f"  Checkpoint       : {out_path}\n"
        f"{'='*110}"
    )
    log(final)
    log_file.close()


if __name__ == "__main__":
    train_spatial_temporal_6b(
        epochs=25,
        batch_size=4,
        spatial_lr=1e-5,
        transformer_lr=1e-4,
        classifier_lr=1e-4,
        min_lr=1e-6
    )
