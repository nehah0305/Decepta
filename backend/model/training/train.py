"""
End-to-End Training Pipeline for Visual Deepfake Detection System.

Trains the full visual pipeline end-to-end:
Loss -> Classifier -> Temporal Transformer -> Linear Projection -> Gated Fusion -> Spatial CNN & Frequency CNN.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DEFAULT_CONFIG, VisualPipelineConfig
from models.visual_model import VisualDeepfakeDetector
from training.dataset import VideoDeepfakeDataset, VideoSampleItem, collate_variable_video_batch
from training.losses import DeepfakeDetectionLoss
from training.validate import validate_epoch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_visual_model(
    train_samples: List[VideoSampleItem],
    val_samples: List[VideoSampleItem],
    config: VisualPipelineConfig = DEFAULT_CONFIG,
    save_checkpoint_path: Optional[Path] = None
) -> Dict[str, list]:
    """
    Executes end-to-end training of the visual deepfake detection pipeline.
    """
    device = torch.device(config.DEVICE)
    logger.info(f"Initializing training on device: {device} (Mode: {config.MODEL_MODE})")

    # Datasets and Loaders
    train_dataset = VideoDeepfakeDataset(
        samples=train_samples,
        coverage_ratio=config.FRAME_COVERAGE_RATIO,
        min_frames=config.MIN_FRAMES,
        max_frames=config.MAX_FRAMES,
        face_size=config.FACE_SIZE,
        device=config.DEVICE
    )
    val_dataset = VideoDeepfakeDataset(
        samples=val_samples,
        coverage_ratio=config.FRAME_COVERAGE_RATIO,
        min_frames=config.MIN_FRAMES,
        max_frames=config.MAX_FRAMES,
        face_size=config.FACE_SIZE,
        device=config.DEVICE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_variable_video_batch,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_variable_video_batch,
        num_workers=0
    )

    # Initialize complete visual model
    model = VisualDeepfakeDetector(
        spatial_dim=config.SPATIAL_FEATURE_DIM,
        frequency_dim=config.FREQUENCY_FEATURE_DIM,
        fusion_hidden_dim=config.FUSION_HIDDEN_DIM,
        fused_dim=config.FUSED_FEATURE_DIM,
        transformer_dim=config.TRANSFORMER_DIM,
        transformer_heads=config.TRANSFORMER_HEADS,
        transformer_layers=config.TRANSFORMER_LAYERS,
        dropout=config.TRANSFORMER_DROPOUT,
        mode=config.MODEL_MODE,
        frame_chunk_size=config.FRAME_BATCH_SIZE
    ).to(device)

    # Loss, Optimizer, and Scheduler
    criterion = DeepfakeDetectionLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,
        eta_min=1e-6
    )

    use_amp = config.USE_AMP and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auc": [],
        "val_f1": []
    }

    best_val_auc = 0.0

    logger.info(f"Starting {config.NUM_EPOCHS} epochs of end-to-end training...")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_train_samples = 0

        for step, batch in enumerate(train_loader):
            frames = batch["face_frames"].to(device)        # (B, N, 3, H, W)
            padding_mask = batch["padding_mask"].to(device)  # (B, N)
            labels = batch["labels"].to(device)              # (B,)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(frames, padding_mask=padding_mask)
                loss = criterion(outputs.logits.view(-1), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_VAL)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * frames.size(0)
            n_train_samples += frames.size(0)

        scheduler.step()

        avg_train_loss = train_loss / n_train_samples if n_train_samples > 0 else 0.0
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_f1"].append(val_metrics["f1"])

        logger.info(
            f"Epoch [{epoch:02d}/{config.NUM_EPOCHS:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Checkpoint best model
        if val_metrics["auc"] >= best_val_auc and save_checkpoint_path is not None:
            best_val_auc = val_metrics["auc"]
            save_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_val_auc,
                "config": config
            }, save_checkpoint_path)
            logger.info(f"Saved best model checkpoint (AUC: {best_val_auc:.4f}) to {save_checkpoint_path}")

    return history
