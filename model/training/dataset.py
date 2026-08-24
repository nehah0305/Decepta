"""
Video-Level Dataset and Collation Module.

Ensures:
1. Strict Video-Level Dataset Partitioning (Zero frame-level data leakage between train/val/test).
2. High-Coverage Frame Sampling per video (~70% of video frames analyzed).
3. Variable sequence length collation with boolean attention padding masks.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import random
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.video_reader import VideoReader
from preprocessing.frame_sampler import HighCoverageFrameSampler
from preprocessing.frame_quality import FrameQualityFilter
from preprocessing.face_alignment import FaceAlignmentPipeline

logger = logging.getLogger(__name__)


@dataclass
class VideoSampleItem:
    """Metadata item representing a single video file and its ground truth label."""
    video_id: str
    video_path: str
    label: int  # 0 = Real, 1 = Fake
    split: str  # "train", "val", "test"


def split_videos_by_id(
    video_records: List[Dict[str, Union[str, int]]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[VideoSampleItem], List[VideoSampleItem], List[VideoSampleItem]]:
    """
    Partitions video files strictly by video identity to prevent data leakage.

    Args:
        video_records: List of dicts with {"video_path": str, "label": int, "video_id": Optional[str]}
        train_ratio: Proportion of unique videos for training.
        val_ratio: Proportion for validation.
        test_ratio: Proportion for testing.
        seed: Random seed for deterministic splitting.

    Returns:
        (train_items, val_items, test_items)
    """
    rng = random.Random(seed)
    shuffled = list(video_records)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_list: List[VideoSampleItem] = []
    val_list: List[VideoSampleItem] = []
    test_list: List[VideoSampleItem] = []

    for i, rec in enumerate(shuffled):
        vpath = str(rec["video_path"])
        vid = str(rec.get("video_id", Path(vpath).stem))
        label = int(rec["label"])

        if i < n_train:
            train_list.append(VideoSampleItem(video_id=vid, video_path=vpath, label=label, split="train"))
        elif i < n_train + n_val:
            val_list.append(VideoSampleItem(video_id=vid, video_path=vpath, label=label, split="val"))
        else:
            test_list.append(VideoSampleItem(video_id=vid, video_path=vpath, label=label, split="test"))

    return train_list, val_list, test_list


class VideoDeepfakeDataset(Dataset):
    """
    PyTorch Dataset that loads videos, extracts high-coverage aligned faces,
    and returns tensor sequences for end-to-end model training.
    """

    def __init__(
        self,
        samples: List[VideoSampleItem],
        coverage_ratio: float = 0.70,
        min_frames: int = 32,
        max_frames: Optional[int] = 300,
        face_size: int = 224,
        transform: Optional[Callable] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.samples = samples
        self.coverage_ratio = coverage_ratio
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.face_size = face_size
        self.transform = transform

        self.sampler = HighCoverageFrameSampler(
            coverage_ratio=coverage_ratio,
            min_frames=min_frames,
            max_frames=max_frames
        )
        self.quality_filter = FrameQualityFilter()
        self.face_aligner = FaceAlignmentPipeline(
            target_size=face_size,
            device=device
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int, float]]:
        item = self.samples[idx]
        vpath = Path(item.video_path)

        if not vpath.exists():
            # Return dummy tensor if video missing
            dummy = torch.zeros(self.min_frames, 3, self.face_size, self.face_size, dtype=torch.float32)
            return {
                "video_id": item.video_id,
                "face_frames": dummy,
                "label": torch.tensor(item.label, dtype=torch.float32),
                "num_valid_frames": 0,
                "coverage_ratio": 0.0
            }

        try:
            reader = VideoReader(vpath)
            meta = reader.metadata
            total_frames = meta.total_frames

            # High coverage candidate indices
            plan = self.sampler.create_sampling_plan(total_frames)
            candidate_indices = plan.candidate_indices

            valid_faces: List[torch.Tensor] = []
            self.face_aligner.reset_tracking()

            # Read and process candidate frames
            for frame_idx, success, frame_rgb, ts in reader.read_frames_by_indices(candidate_indices):
                if not success or frame_rgb is None:
                    continue

                q_res = self.quality_filter.evaluate_frame(frame_rgb, frame_idx, ts)
                if not q_res.is_usable:
                    continue

                face_res = self.face_aligner.process_frame(frame_rgb, frame_idx, ts)
                if face_res.face_detected and face_res.aligned_face is not None:
                    # Convert (224, 224, 3) uint8 -> (3, 224, 224) float [0, 1]
                    face_tensor = torch.from_numpy(face_res.aligned_face).permute(2, 0, 1).float() / 255.0
                    if self.transform:
                        face_tensor = self.transform(face_tensor)
                    valid_faces.append(face_tensor)

            if len(valid_faces) == 0:
                # Fallback if no faces detected
                valid_faces = [torch.zeros(3, self.face_size, self.face_size, dtype=torch.float32)]

            face_stack = torch.stack(valid_faces, dim=0)  # (N, 3, 224, 224)
            num_valid = len(valid_faces)
            coverage = num_valid / total_frames if total_frames > 0 else 0.0

            return {
                "video_id": item.video_id,
                "face_frames": face_stack,
                "label": torch.tensor(item.label, dtype=torch.float32),
                "num_valid_frames": num_valid,
                "coverage_ratio": coverage
            }

        except Exception as e:
            logger.error(f"Error loading video {item.video_path}: {e}")
            dummy = torch.zeros(self.min_frames, 3, self.face_size, self.face_size, dtype=torch.float32)
            return {
                "video_id": item.video_id,
                "face_frames": dummy,
                "label": torch.tensor(item.label, dtype=torch.float32),
                "num_valid_frames": 0,
                "coverage_ratio": 0.0
            }


def collate_variable_video_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collator that dynamically pads video sequences of different frame counts
    to the batch maximum length and generates a padding mask for the Temporal Transformer.
    """
    video_ids = [b["video_id"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    frame_counts = [b["face_frames"].size(0) for b in batch]
    max_frames = max(frame_counts)

    B = len(batch)
    C, H, W = batch[0]["face_frames"].shape[1:]

    # Padded batch tensor: (B, max_frames, 3, H, W)
    padded_frames = torch.zeros(B, max_frames, C, H, W, dtype=torch.float32)
    # Boolean mask: True indicates padded/ignored positions
    padding_mask = torch.ones(B, max_frames, dtype=torch.bool)

    for i, b in enumerate(batch):
        n = frame_counts[i]
        padded_frames[i, :n] = b["face_frames"]
        padding_mask[i, :n] = False  # False = real frame

    return {
        "video_ids": video_ids,
        "face_frames": padded_frames,
        "padding_mask": padding_mask,
        "labels": labels,
        "frame_counts": torch.tensor(frame_counts, dtype=torch.long)
    }
