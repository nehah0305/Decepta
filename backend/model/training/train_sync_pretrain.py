"""
Standalone Stage 3 Audio-Visual Synchronization Contrastive Pretraining Script.

Pretrains the Mouth CNN, Audio Sync Projection, and Sync Temporal Module independently
using canonical Temperature-Scaled InfoNCE (tau = 0.07):

Mouth ROI Sequence -> Mouth CNN -> M_t (256-D)
Audio Log-Mel -> Audio 2D CNN -> Projection -> A_t (256-D)
Temporal Grid Alignment -> L2 Normalization -> Cosine Similarity -> InfoNCE Loss

Pairs:
- Positive: Correctly aligned mouth and audio.
- Negative: Temporally shifted audio (+/-0.5s, +/-1.0s, +/-2.0s) and cross-sample audio.

Saves best weights to: checkpoints/sync_stage3_best.pt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure model root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG, VisualPipelineConfig
from models.audio_cnn import Audio2DCNN
from models.mouth_encoder import MouthROIEncoder
from models.sync_branch import AudioVisualSyncBranch
from preprocessing.mouth_extractor import MouthExtractor
from preprocessing.audio_windowing import AudioWindowExtractor
from preprocessing.video_reader import VideoReader
from preprocessing.frame_sampler import HighCoverageFrameSampler
from preprocessing.frame_quality import FrameQualityFilter
from preprocessing.face_alignment import FaceAlignmentPipeline
from training.dataset import VideoSampleItem, split_videos_by_id
from training.losses import InfoNCESyncLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StandaloneSyncModel(nn.Module):
    """
    Standalone Audio-Visual Sync module combining Mouth Encoder,
    Audio 2D CNN feature extractor, and AudioVisualSyncBranch.
    """

    def __init__(
        self,
        audio_dim: int = 768,
        mouth_dim: int = 256,
        sync_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.audio_cnn = Audio2DCNN(in_channels=1, out_dim=audio_dim)
        self.mouth_encoder = MouthROIEncoder(in_channels=3, embedding_dim=mouth_dim)
        self.sync_branch = AudioVisualSyncBranch(
            audio_token_dim=audio_dim,
            mouth_dim=mouth_dim,
            sync_dim=sync_dim,
            dropout=dropout
        )

    def forward(
        self,
        mouth_crops: torch.Tensor,
        mel_windows: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mouth_crops: (B, N, 3, 112, 112) or (N, 3, 112, 112)
            mel_windows: (B, W, 128, T) or (W, 128, T)

        Returns:
            sync_feature: (B, 256)
            sync_score: (B, 1)
            pos_sims: (B, K)
            aligned_mouth: (B, K, 256)
        """
        # 1. Extract mouth embeddings
        mouth_embeddings = self.mouth_encoder(mouth_crops)  # (B, N, 256)

        # 2. Extract audio temporal tokens
        if mel_windows.dim() == 4:
            B, W, M, T = mel_windows.shape
            flat_mels = mel_windows.view(B * W, M, T)
            flat_tokens = self.audio_cnn(flat_mels)
            audio_tokens = flat_tokens.view(B, -1, 768)  # (B, W*T', 768)
        else:
            audio_tokens = self.audio_cnn(mel_windows)   # (T', 768) or (B, T', 768)

        # 3. Sync branch forward pass
        out = self.sync_branch(mouth_embeddings, audio_tokens)
        return out.sync_feature, out.sync_score, out.temporal_similarities, out.aligned_mouth_embeddings


class SyncPretrainDataset(Dataset):
    """
    Dataset extracting aligned mouth ROI frames and 16 kHz Log-Mel audio windows for Sync pretraining.
    """

    def __init__(
        self,
        samples: List[VideoSampleItem],
        config: VisualPipelineConfig = DEFAULT_CONFIG,
        min_frames: int = 32
    ):
        self.samples = samples
        self.config = config
        self.min_frames = min_frames
        self.sampler = HighCoverageFrameSampler(
            coverage_ratio=config.FRAME_COVERAGE_RATIO,
            min_frames=min_frames,
            max_frames=150
        )
        self.quality_filter = FrameQualityFilter()
        self.face_aligner = FaceAlignmentPipeline(target_size=config.FACE_SIZE)
        self.mouth_extractor = MouthExtractor(mouth_roi_size=(config.MOUTH_ROI_SIZE, config.MOUTH_ROI_SIZE))
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
            return self._dummy_sample(item.video_id)

        try:
            reader = VideoReader(vpath)
            plan = self.sampler.create_sampling_plan(reader.metadata.total_frames)

            valid_mouths = []
            visual_timestamps = []
            self.face_aligner.reset_tracking()

            for f_idx, ok, rgb, ts in reader.read_frames_by_indices(plan.candidate_indices):
                if not ok or rgb is None:
                    continue
                q = self.quality_filter.evaluate_frame(rgb, f_idx, ts)
                if not q.is_usable:
                    continue
                f_res = self.face_aligner.process_frame(rgb, f_idx, ts)
                if f_res.face_detected and f_res.aligned_face is not None:
                    visual_timestamps.append((f_idx, ts))
                    if f_res.landmarks:
                        box = self.mouth_extractor.compute_mouth_box(f_res.landmarks, (self.config.FACE_SIZE, self.config.FACE_SIZE))
                        if box:
                            mcrop = self.mouth_extractor.crop_mouth_roi(f_res.aligned_face, box)
                            mt = torch.from_numpy(mcrop).permute(2, 0, 1).float() / 255.0
                            valid_mouths.append(mt)

            if len(valid_mouths) == 0:
                return self._dummy_sample(item.video_id)

            mouth_stack = torch.stack(valid_mouths, dim=0)

            # Audio extraction
            audio_res = self.audio_extractor.process_video_audio(vpath, visual_timestamps=visual_timestamps)
            if not audio_res.audio_available or len(audio_res.windows) == 0:
                return self._dummy_sample(item.video_id)

            mel_list = [torch.from_numpy(w.mel_spectrogram) for w in audio_res.windows]
            mel_stack = torch.stack(mel_list, dim=0)

            return {
                "video_id": item.video_id,
                "mouth_crops": mouth_stack,
                "mel_windows": mel_stack,
                "num_mouths": len(valid_mouths),
                "num_windows": len(audio_res.windows),
                "valid": True
            }
        except Exception as e:
            logger.warning(f"Error reading sync sample {vpath.name}: {e}")
            return self._dummy_sample(item.video_id)

    def _dummy_sample(self, video_id: str) -> Dict:
        return {
            "video_id": video_id,
            "mouth_crops": torch.zeros(self.min_frames, 3, self.config.MOUTH_ROI_SIZE, self.config.MOUTH_ROI_SIZE),
            "mel_windows": torch.zeros(1, 128, 251),
            "num_mouths": 0,
            "num_windows": 0,
            "valid": False
        }


def collate_sync_batch(batch: List[Dict]) -> Dict:
    """Collates and pads variable mouth sequences and audio window sequences."""
    valid_items = [b for b in batch if b["valid"]]
    if not valid_items:
        valid_items = batch

    B = len(valid_items)
    max_mouths = max(b["mouth_crops"].size(0) for b in valid_items)
    max_windows = max(b["mel_windows"].size(0) for b in valid_items)
    _, m_dim, t_dim = valid_items[0]["mel_windows"].shape
    m_size = valid_items[0]["mouth_crops"].size(-1)

    padded_mouths = torch.zeros(B, max_mouths, 3, m_size, m_size, dtype=torch.float32)
    padded_mels = torch.zeros(B, max_windows, m_dim, t_dim, dtype=torch.float32)

    for i, b in enumerate(valid_items):
        nm = b["mouth_crops"].size(0)
        nw = b["mel_windows"].size(0)
        padded_mouths[i, :nm] = b["mouth_crops"]
        padded_mels[i, :nw] = b["mel_windows"]

    return {
        "video_ids": [b["video_id"] for b in valid_items],
        "mouth_crops": padded_mouths,
        "mel_windows": padded_mels
    }


def train_sync_stage3(
    train_samples: List[VideoSampleItem],
    val_samples: List[VideoSampleItem],
    config: VisualPipelineConfig = DEFAULT_CONFIG,
    save_path: Optional[Path] = None,
    epochs: int = 15,
    lr: float = 1e-4,
    batch_size: int = 8,
    num_workers: int = 4,
    temperature: float = 0.07
) -> Dict[str, list]:
    """
    Executes standalone Stage 3 Audio-Visual Synchronization InfoNCE pretraining.
    """
    device = torch.device(config.DEVICE)
    if save_path is None:
        save_path = config.CHECKPOINTS_DIR / "sync_stage3_best.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- STARTING STAGE 3: SYNC INFONCE PRETRAINING on {device} (tau={temperature}) ---")
    logger.info(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)} | Epochs: {epochs} | Batch size: {batch_size} | Workers: {num_workers}")

    train_dataset = SyncPretrainDataset(train_samples, config=config)
    val_dataset = SyncPretrainDataset(val_samples, config=config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sync_batch,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sync_batch,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    model = StandaloneSyncModel(
        audio_dim=config.AUDIO_FEATURE_DIM,
        mouth_dim=config.MOUTH_EMBEDDING_DIM,
        sync_dim=config.SYNC_FEATURE_DIM,
        dropout=config.TRANSFORMER_DROPOUT
    ).to(device)

    criterion = InfoNCESyncLoss(temperature=temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = config.USE_AMP and (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_pos_sim": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            mouths = batch["mouth_crops"].to(device)
            mels = batch["mel_windows"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                sync_feat, sync_sc, pos_sims, aligned_m = model(mouths, mels)
                if mouths.size(0) > 1:
                    loss = criterion(pos_similarities=pos_sims)
                else:
                    neg_sim = pos_sims - 0.4
                    loss = criterion(pos_similarities=pos_sims, neg_similarities=neg_sim)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_VAL)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * mouths.size(0)
            n_train += mouths.size(0)

        scheduler.step()
        avg_train = train_loss / n_train if n_train > 0 else 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        pos_sim_list = []
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                mouths = batch["mouth_crops"].to(device)
                mels = batch["mel_windows"].to(device)
                sync_feat, sync_sc, pos_sims, _ = model(mouths, mels)
                loss = criterion(pos_similarities=pos_sims)
                val_loss += loss.item() * mouths.size(0)
                n_val += mouths.size(0)
                pos_sim_list.append(float(pos_sims.mean().item()))

        avg_val_loss = val_loss / n_val if n_val > 0 else 0.0
        avg_pos_sim = float(np.mean(pos_sim_list)) if pos_sim_list else 0.5

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val_loss)
        history["val_pos_sim"].append(avg_pos_sim)

        logger.info(
            f"Stage 3 Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train InfoNCE Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Mean In-Sync Cosine Sim: {avg_pos_sim:.4f}"
        )

        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "mouth_encoder_state_dict": model.mouth_encoder.state_dict(),
                "sync_branch_state_dict": model.sync_branch.state_dict(),
                "audio_cnn_state_dict": model.audio_cnn.state_dict(),
                "val_loss": best_val_loss,
                "config": config
            }, save_path)
            logger.info(f"Saved best Stage 3 Sync checkpoint (Loss: {best_val_loss:.4f}) -> {save_path}")

    logger.info(f"--- STAGE 3 SYNC PRETRAINING COMPLETE. Best Val Loss: {best_val_loss:.4f} ---")
    return history


def load_samples_from_csv(csv_path: Path, data_root: Path, split_name: Optional[str] = None, max_samples: Optional[int] = None) -> List[VideoSampleItem]:
    import pandas as pd
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if split_name and "split" in df.columns:
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
                split=split_name or "train"
            )
        )
    return items


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Standalone Audio-Visual Synchronization InfoNCE Pretraining")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples limit (optional)")
    parser.add_argument("--train-manifest", type=str, default="Datasets/metadata/train_ffpp.csv", help="Train manifest path")
    parser.add_argument("--val-manifest", type=str, default="Datasets/metadata/val_ffpp.csv", help="Val manifest path")
    parser.add_argument("--data-root", type=str, default="Datasets/raw/faceforensicspp", help="Data root folder")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature tau (default: 0.07)")
    parser.add_argument("--output", type=str, default="model/checkpoints/sync_stage3_best.pt", help="Checkpoint save path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    train_manifest_path = project_root / args.train_manifest if not Path(args.train_manifest).is_absolute() else Path(args.train_manifest)
    val_manifest_path = project_root / args.val_manifest if not Path(args.val_manifest).is_absolute() else Path(args.val_manifest)
    data_root_path = project_root / args.data_root if not Path(args.data_root).is_absolute() else Path(args.data_root)

    train_s = load_samples_from_csv(train_manifest_path, data_root_path, max_samples=args.max_samples)
    val_s = load_samples_from_csv(val_manifest_path, data_root_path, max_samples=args.max_samples // 4 if args.max_samples else None)

    if not train_s:
        logger.warning("No samples found in manifest. Fallback to dummy dataset.")
        dummy_samples = [
            VideoSampleItem(video_id=f"demo_sync_{i}", video_path="sample_test_video.mp4", label=0)
            for i in range(8)
        ]
        train_s, val_s, _ = split_videos_by_id(dummy_samples, train_ratio=0.75, val_ratio=0.25, test_ratio=0.0)

    train_sync_stage3(
        train_s,
        val_s,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        temperature=args.temperature,
        save_path=Path(args.output)
    )


if __name__ == "__main__":
    main()
