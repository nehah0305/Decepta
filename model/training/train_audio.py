"""
Standalone Stage 2 Audio Branch Deepfake Pretraining Script.

Pretrains the Audio Authenticity Branch independently on binary deepfake audio classification:
Audio -> 16 kHz Log-Mel -> Custom 2D Audio CNN -> Positional Encoding ->
TransformerEncoder -> Attention Pooling -> F_audio (768-D) -> Classifier Head -> P(Fake)

Saves best model weights to: checkpoints/audio_stage2_best.pt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure model root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG, VisualPipelineConfig
from models.audio_branch import AudioAuthenticityBranch
from preprocessing.audio_windowing import AudioWindowExtractor, AudioWindowData
from training.dataset import VideoSampleItem, split_videos_by_id
from training.losses import DeepfakeDetectionLoss
from evaluation.metrics import calculate_deepfake_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StandaloneAudioClassifier(nn.Module):
    """
    Standalone Audio Deepfake Detector combining AudioAuthenticityBranch
    and a binary classification head.
    """

    def __init__(
        self,
        in_mels: int = 128,
        audio_dim: int = 768,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.audio_branch = AudioAuthenticityBranch(
            in_mels=in_mels,
            d_model=audio_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            use_self_attention=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(audio_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1)
        )

    def forward(
        self,
        mel_windows: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mel_windows: (B, W, 128, T) or (W, 128, T) or (128, T)

        Returns:
            logits: (B, 1) or (1,)
            probs: (B, 1) or (1,)
            audio_feature: (B, 768)
        """
        out = self.audio_branch(mel_windows, padding_mask=padding_mask)
        f_a = out.audio_feature
        if f_a.dim() == 1:
            f_a_in = f_a.unsqueeze(0)
        else:
            f_a_in = f_a

        logits = self.classifier(f_a_in)
        probs = torch.sigmoid(logits)

        if f_a.dim() == 1:
            logits = logits.squeeze(0)
            probs = probs.squeeze(0)

        return logits, probs, f_a


class AudioVideoDataset(Dataset):
    """
    Dataset extracting 16 kHz Log-Mel spectrogram windows from videos for audio pretraining.
    """

    def __init__(
        self,
        samples: List[VideoSampleItem],
        config: VisualPipelineConfig = DEFAULT_CONFIG
    ):
        self.samples = samples
        self.config = config
        self.audio_extractor = AudioWindowExtractor(
            sample_rate=config.AUDIO_SAMPLE_RATE,
            window_seconds=config.AUDIO_WINDOW_SECONDS,
            hop_seconds=config.AUDIO_HOP_SECONDS,
            n_mels=config.AUDIO_N_MELS,
            n_fft=config.AUDIO_N_FFT,
            hop_length=config.AUDIO_HOP_LENGTH
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        vpath = Path(item.video_path)

        if not vpath.exists():
            return self._dummy_sample(item.video_id, item.label)

        try:
            res = self.audio_extractor.process_video_audio(vpath)
            if not res.audio_available or len(res.windows) == 0:
                return self._dummy_sample(item.video_id, item.label)

            mel_list = [torch.from_numpy(w.mel_spectrogram) for w in res.windows]
            mel_stack = torch.stack(mel_list, dim=0)  # (W, 128, T)

            return {
                "video_id": item.video_id,
                "mel_windows": mel_stack,
                "label": torch.tensor(item.label, dtype=torch.float32),
                "num_windows": len(res.windows),
                "has_audio": True
            }
        except Exception as e:
            logger.warning(f"Failed to load audio for {vpath.name}: {e}")
            return self._dummy_sample(item.video_id, item.label)

    def _dummy_sample(self, video_id: str, label: int) -> Dict:
        return {
            "video_id": video_id,
            "mel_windows": torch.zeros(1, 128, 251, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "num_windows": 0,
            "has_audio": False
        }


def collate_audio_batch(batch: List[Dict]) -> Dict:
    """Collates and dynamically pads variable audio window counts."""
    video_ids = [b["video_id"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    has_audios = [b["has_audio"] for b in batch]

    window_lens = [b["mel_windows"].size(0) for b in batch]
    max_windows = max(window_lens)
    _, m_dim, t_dim = batch[0]["mel_windows"].shape
    B = len(batch)

    padded_mels = torch.zeros(B, max_windows, m_dim, t_dim, dtype=torch.float32)
    padding_mask = torch.ones(B, max_windows, dtype=torch.bool)

    for i, b in enumerate(batch):
        w = window_lens[i]
        padded_mels[i, :w] = b["mel_windows"]
        if b["has_audio"]:
            padding_mask[i, :w] = False

    return {
        "video_ids": video_ids,
        "mel_windows": padded_mels,
        "padding_mask": padding_mask,
        "labels": labels,
        "has_audios": has_audios
    }


def validate_audio_epoch(
    model: StandaloneAudioClassifier,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluates standalone audio model on validation split."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            mels = batch["mel_windows"].to(device)
            pads = batch["padding_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, probs, _ = model(mels, padding_mask=pads)
            loss = criterion(logits.view(-1), labels)

            total_loss += loss.item() * mels.size(0)
            all_probs.extend(probs.view(-1).cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    n_samples = len(all_targets)
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
    metrics = calculate_deepfake_metrics(np.array(all_targets), np.array(all_probs))
    metrics["loss"] = round(float(avg_loss), 4)
    return metrics


def train_audio_stage2(
    train_samples: List[VideoSampleItem],
    val_samples: List[VideoSampleItem],
    config: VisualPipelineConfig = DEFAULT_CONFIG,
    save_path: Optional[Path] = None,
    epochs: int = 15,
    lr: float = 1e-4,
    batch_size: int = 4
) -> Dict[str, list]:
    """
    Executes standalone Stage 2 Audio Branch deepfake pretraining.
    """
    device = torch.device(config.DEVICE)
    if save_path is None:
        save_path = config.CHECKPOINTS_DIR / "audio_stage2_best.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- STARTING STAGE 2: AUDIO BRANCH PRETRAINING on {device} ---")
    logger.info(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)} | Epochs: {epochs}")

    train_dataset = AudioVideoDataset(train_samples, config=config)
    val_dataset = AudioVideoDataset(val_samples, config=config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_audio_batch,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_audio_batch,
        num_workers=0
    )

    model = StandaloneAudioClassifier(
        in_mels=config.AUDIO_N_MELS,
        audio_dim=config.AUDIO_FEATURE_DIM,
        nhead=config.AUDIO_TRANSFORMER_HEADS,
        num_layers=config.AUDIO_TRANSFORMER_LAYERS,
        dropout=config.AUDIO_TRANSFORMER_DROPOUT
    ).to(device)

    criterion = DeepfakeDetectionLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = config.USE_AMP and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_auc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            mels = batch["mel_windows"].to(device)
            pads = batch["padding_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _, _ = model(mels, padding_mask=pads)
                loss = criterion(logits.view(-1), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_VAL)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * mels.size(0)
            n_train += mels.size(0)

        scheduler.step()
        avg_train = train_loss / n_train if n_train > 0 else 0.0
        val_metrics = validate_audio_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["roc_auc"])
        history["val_acc"].append(val_metrics["accuracy"])

        logger.info(
            f"Stage 2 Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Val F1: {val_metrics['f1_score']:.4f}"
        )

        if val_metrics["roc_auc"] >= best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "audio_branch_state_dict": model.audio_branch.state_dict(),
                "val_auc": best_val_auc,
                "config": config
            }, save_path)
            logger.info(f"Saved best Stage 2 Audio checkpoint (AUC: {best_val_auc:.4f}) -> {save_path}")

    logger.info(f"--- STAGE 2 AUDIO PRETRAINING COMPLETE. Best Val AUC: {best_val_auc:.4f} ---")
    return history


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Standalone Audio Branch Deepfake Pretraining")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--output", type=str, default="checkpoints/audio_stage2_best.pt", help="Checkpoint save path")
    args = parser.parse_args()

    # Create dummy dataset demonstration if run standalone
    dummy_samples = [
        VideoSampleItem(video_id=f"demo_audio_{i}", video_path="sample_test_video.mp4", label=(i % 2))
        for i in range(8)
    ]
    train_s, val_s, _ = split_videos_by_id(dummy_samples, train_ratio=0.75, val_ratio=0.25, test_ratio=0.0)
    train_audio_stage2(train_s, val_s, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, save_path=Path(args.output))


if __name__ == "__main__":
    main()
