"""
End-to-End Multimodal Deepfake Detection Training Pipeline.

Optimizes:
  L_total = L_classification + λ_sync * L_sync (with canonical InfoNCE tau=0.07)

Supports:
- Stage 4 Multimodal Training (with optional Stage 1, Stage 2, and Stage 3 pretrained weights)
- Stage 5 Full Multimodal End-to-End Fine-Tuning
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch.utils.data import DataLoader, WeightedRandomSampler

# Ensure model root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG, VisualPipelineConfig
from models.multimodal_detector import MultimodalDeepfakeDetector
from training.multimodal_dataset import MultimodalVideoDataset, collate_multimodal_batch
from training.dataset import VideoSampleItem, split_videos_by_id
from training.losses import MultimodalCompoundLoss
from evaluation.metrics import calculate_deepfake_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_multimodal_epoch(
    model: MultimodalDeepfakeDetector,
    val_loader: DataLoader,
    criterion: MultimodalCompoundLoss,
    device: torch.device
) -> Dict[str, float]:
    """Evaluates multimodal model across validation dataset."""
    model.eval()
    total_loss = 0.0
    all_targets: list = []
    all_probs: list = []
    sync_scores: list = []

    with torch.no_grad():
        for batch in val_loader:
            faces = batch["face_frames"].to(device)
            mouths = batch["mouth_crops"].to(device)
            mels = batch["mel_windows"].to(device)
            mod_masks = batch["modality_masks"].to(device)
            pad_v = batch["padding_mask_v"].to(device)
            pad_a = batch["padding_mask_a"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                face_frames=faces,
                mouth_crops=mouths,
                mel_windows=mels,
                modality_mask=mod_masks,
                padding_mask_v=pad_v,
                padding_mask_a=pad_a
            )

            loss, _, _ = criterion(
                logits=outputs.logits.view(-1),
                targets=labels,
                pos_sync_sims=outputs.temporal_similarities
            )

            total_loss += loss.item() * faces.size(0)
            probs = outputs.probability.view(-1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

            if outputs.sync_score is not None:
                sync_scores.extend(outputs.sync_score.view(-1).cpu().numpy().tolist())

    n_samples = len(all_targets)
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
    metrics = calculate_deepfake_metrics(np.array(all_targets), np.array(all_probs))
    metrics["loss"] = round(float(avg_loss), 4)
    metrics["avg_sync_score"] = round(float(np.mean(sync_scores)), 4) if sync_scores else 0.5

    return metrics


def train_multimodal_model(
    train_samples: List[VideoSampleItem],
    val_samples: List[VideoSampleItem],
    config: VisualPipelineConfig = DEFAULT_CONFIG,
    save_checkpoint_path: Optional[Path] = None,
    visual_checkpoint: Optional[Union[str, Path]] = None,
    audio_checkpoint: Optional[Union[str, Path]] = None,
    sync_checkpoint: Optional[Union[str, Path]] = None,
    resume_checkpoint: Optional[Union[str, Path]] = None
) -> Dict[str, list]:
    """
    Executes end-to-end multimodal training optimizing classification and synchronization.
    Optionally initializes from Stage 1 (Visual), Stage 2 (Audio), and Stage 3 (Sync) weights.
    """
    device = torch.device(config.DEVICE)
    logger.info(f"--- STARTING MULTIMODAL TRAINING on {device} (Mode: {config.MODEL_MODE}) ---")

    train_dataset = MultimodalVideoDataset(
        samples=train_samples,
        coverage_ratio=config.FRAME_COVERAGE_RATIO,
        min_frames=config.MIN_FRAMES,
        max_frames=config.MAX_FRAMES,
        face_size=config.FACE_SIZE,
        mouth_size=config.MOUTH_ROI_SIZE,
        audio_window_sec=config.AUDIO_WINDOW_SECONDS,
        audio_hop_sec=config.AUDIO_HOP_SECONDS,
        device=config.DEVICE
    )
    val_dataset = MultimodalVideoDataset(
        samples=val_samples,
        coverage_ratio=config.FRAME_COVERAGE_RATIO,
        min_frames=config.MIN_FRAMES,
        max_frames=config.MAX_FRAMES,
        face_size=config.FACE_SIZE,
        mouth_size=config.MOUTH_ROI_SIZE,
        audio_window_sec=config.AUDIO_WINDOW_SECONDS,
        audio_hop_sec=config.AUDIO_HOP_SECONDS,
        device=config.DEVICE
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    num_workers = 0
    pin_memory = (device.type == "cuda")

    labels = [sample.label for sample in train_samples]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    logger.info(f"Using WeightedRandomSampler for training | Class Counts: REAL={class_counts[0] if len(class_counts)>0 else 0}, FAKE={class_counts[1] if len(class_counts)>1 else 0}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        collate_fn=collate_multimodal_batch,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_multimodal_batch,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    model = MultimodalDeepfakeDetector(
        visual_dim=config.TRANSFORMER_DIM,
        audio_dim=config.AUDIO_FEATURE_DIM,
        sync_dim=config.SYNC_FEATURE_DIM,
        fusion_dim=config.FUSION_DIM,
        mode=config.MODEL_MODE,
        dropout=config.TRANSFORMER_DROPOUT,
        frame_chunk_size=config.FRAME_BATCH_SIZE
    ).to(device)

    # -------------------------------------------------------------------------
    # Optional Pretrained Checkpoint Loading (Stages 1, 2, 3, or Resume)
    # -------------------------------------------------------------------------
    if resume_checkpoint and Path(resume_checkpoint).exists():
        chk = torch.load(resume_checkpoint, map_location=device)
        state_dict = chk.get("model_state_dict", chk)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded full model checkpoint for Stage 5 fine-tuning from: {resume_checkpoint}")
    else:
        if visual_checkpoint and Path(visual_checkpoint).exists():
            v_chk = torch.load(visual_checkpoint, map_location=device)
            v_dict = v_chk.get("model_state_dict", v_chk)
            model.visual_branch.load_state_dict(v_dict, strict=False)
            logger.info(f"Loaded Stage 1 Visual weights from: {visual_checkpoint}")

        if audio_checkpoint and Path(audio_checkpoint).exists():
            a_chk = torch.load(audio_checkpoint, map_location=device)
            a_dict = a_chk.get("audio_branch_state_dict", a_chk.get("model_state_dict", a_chk))
            model.audio_branch.load_state_dict(a_dict, strict=False)
            logger.info(f"Loaded Stage 2 Audio weights from: {audio_checkpoint}")

        if sync_checkpoint and Path(sync_checkpoint).exists():
            s_chk = torch.load(sync_checkpoint, map_location=device)
            if "mouth_encoder_state_dict" in s_chk:
                model.mouth_encoder.load_state_dict(s_chk["mouth_encoder_state_dict"], strict=False)
            if "sync_branch_state_dict" in s_chk:
                model.sync_branch.load_state_dict(s_chk["sync_branch_state_dict"], strict=False)
            logger.info(f"Loaded Stage 3 Sync weights from: {sync_checkpoint}")

    criterion = MultimodalCompoundLoss(
        lambda_sync=config.SYNC_LOSS_WEIGHT,
        label_smoothing=0.05,
        sync_loss_type=config.SYNC_LOSS_TYPE,
        sync_temperature=config.SYNC_TEMPERATURE
    ).to(device)

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
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auc": [],
        "val_f1": [],
        "val_sync_score": []
    }

    best_val_auc = 0.0

    total_steps = len(train_loader) * config.NUM_EPOCHS
    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_train_samples = 0

        for step, batch in enumerate(train_loader):
            faces = batch["face_frames"].to(device)
            mouths = batch["mouth_crops"].to(device)
            mels = batch["mel_windows"].to(device)
            mod_masks = batch["modality_masks"].to(device)
            pad_v = batch["padding_mask_v"].to(device)
            pad_a = batch["padding_mask_a"].to(device)
            labels = batch["labels"].to(device)

            if device.type == "cuda":
                torch.cuda.empty_cache()
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(
                    face_frames=faces,
                    mouth_crops=mouths,
                    mel_windows=mels,
                    modality_mask=mod_masks,
                    padding_mask_v=pad_v,
                    padding_mask_a=pad_a
                )

                loss, l_cls, l_sync = criterion(
                    logits=outputs.logits.view(-1),
                    targets=labels,
                    pos_sync_sims=outputs.temporal_similarities
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_VAL)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * faces.size(0)
            n_train_samples += faces.size(0)

            current_global_step = (epoch - 1) * len(train_loader) + (step + 1)
            pct_complete = (current_global_step / total_steps) * 100.0 if total_steps > 0 else 0.0
            elapsed = time.time() - start_time
            eta_seconds = (elapsed / current_global_step) * (total_steps - current_global_step) if current_global_step > 0 else 0.0

            logger.info(
                f"[PROGRESS: {pct_complete:5.1f}%] Epoch [{epoch:02d}/{config.NUM_EPOCHS:02d}] "
                f"Step [{step+1:04d}/{len(train_loader):04d}] | Loss: {loss.item():.4f} "
                f"(Cls: {l_cls.item():.4f}, Sync: {l_sync.item():.4f}) | ETA: {eta_seconds/60:.1f}m"
            )
            sys.stdout.flush()

        scheduler.step()
        avg_train_loss = train_loss / n_train_samples if n_train_samples > 0 else 0.0
        val_metrics = validate_multimodal_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["roc_auc"])
        history["val_f1"].append(val_metrics["f1_score"])
        history["val_sync_score"].append(val_metrics["avg_sync_score"])

        logger.info(
            f"Epoch [{epoch:02d}/{config.NUM_EPOCHS:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Val Sync Score: {val_metrics['avg_sync_score']:.4f}"
        )

        if val_metrics["roc_auc"] >= best_val_auc and save_checkpoint_path is not None:
            best_val_auc = val_metrics["roc_auc"]
            save_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_val_auc,
                "config": config
            }, save_checkpoint_path)
            logger.info(f"Saved best checkpoint (AUC: {best_val_auc:.4f}) to {save_checkpoint_path}")

    return history


def load_samples_from_csv(csv_path: Path, data_root: Path, split_name: str, max_samples: Optional[int] = None) -> List[VideoSampleItem]:
    import pandas as pd
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"] == split_name]
    if max_samples and max_samples > 0:
        df = df.head(max_samples)
    items = []
    for _, row in df.iterrows():
        rel_path = str(row["video_path"])
        full_path = data_root / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        items.append(
            VideoSampleItem(
                video_id=str(row.get("sample_id", Path(rel_path).stem)),
                video_path=str(full_path),
                label=int(row["label"]),
                split=split_name
            )
        )
    return items


def main():
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Multimodal Deepfake Detection Training (Stage 4 & Stage 5)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.NUM_EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum training/validation samples limit (optional)")
    parser.add_argument("--train-manifest", type=str, default="Datasets/metadata/train.csv", help="Train CSV manifest path")
    parser.add_argument("--val-manifest", type=str, default="Datasets/metadata/validation.csv", help="Validation CSV manifest path")
    parser.add_argument("--data-root", type=str, default="Datasets/raw/faceforensicspp/c23", help="Root folder for video files")
    parser.add_argument("--visual-checkpoint", type=str, default=None, help="Stage 1 Visual pretrained checkpoint (.pt)")
    parser.add_argument("--audio-checkpoint", type=str, default=None, help="Stage 2 Audio pretrained checkpoint (.pt)")
    parser.add_argument("--sync-checkpoint", type=str, default=None, help="Stage 3 Sync pretrained checkpoint (.pt)")
    parser.add_argument("--resume", type=str, default=None, help="Resume full multimodal checkpoint for fine-tuning")
    parser.add_argument("--output", type=str, default="checkpoints/multimodal_best.pt", help="Checkpoint save path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    train_manifest_path = project_root / args.train_manifest if not Path(args.train_manifest).is_absolute() else Path(args.train_manifest)
    val_manifest_path = project_root / args.val_manifest if not Path(args.val_manifest).is_absolute() else Path(args.val_manifest)
    data_root_path = project_root / args.data_root if not Path(args.data_root).is_absolute() else Path(args.data_root)

    train_s = load_samples_from_csv(train_manifest_path, data_root_path, "train", max_samples=args.max_samples)
    val_s = load_samples_from_csv(val_manifest_path, data_root_path, "validation", max_samples=args.max_samples // 4 if args.max_samples else None)

    if not train_s:
        logger.warning("Train manifest not found or empty. Fallback to demo video samples.")
        dummy_samples = [
            VideoSampleItem(video_id=f"demo_multi_{i}", video_path="sample_test_video.mp4", label=(i % 2))
            for i in range(8)
        ]
        train_s, val_s, _ = split_videos_by_id(dummy_samples, train_ratio=0.75, val_ratio=0.25, test_ratio=0.0)

    logger.info(f"Loaded {len(train_s)} training samples and {len(val_s)} validation samples.")

    cfg = VisualPipelineConfig(
        NUM_EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.lr
    )
    train_multimodal_model(
        train_s,
        val_s,
        config=cfg,
        save_checkpoint_path=Path(args.output),
        visual_checkpoint=args.visual_checkpoint,
        audio_checkpoint=args.audio_checkpoint,
        sync_checkpoint=args.sync_checkpoint,
        resume_checkpoint=args.resume
    )


if __name__ == "__main__":
    main()
