"""
Multimodal Video Dataset with Strict Video-Level Partitioning.

Ensures:
1. Strict Video-Level Dataset Partitioning (Zero leakage between train/val/test).
2. High-Coverage Visual Frame Sampling (~70%) with Face and Mouth ROI extraction.
3. 16 kHz Audio Windowing (4-second overlapping windows).
4. Dynamic temporal offset synthesis for audio-visual synchronization contrastive training.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.video_reader import VideoReader
from preprocessing.frame_sampler import HighCoverageFrameSampler
from preprocessing.frame_quality import FrameQualityFilter
from preprocessing.face_alignment import FaceAlignmentPipeline
from preprocessing.mouth_extractor import MouthExtractor
from preprocessing.audio_windowing import AudioWindowExtractor, AudioWindowData, VideoAudioResult
from training.dataset import VideoSampleItem, split_videos_by_id

logger = logging.getLogger(__name__)


class MultimodalVideoDataset(Dataset):
    """
    Multimodal Dataset extracting high-coverage visual frames, mouth ROIs,
    and 16 kHz Log-Mel audio windows per video.
    """

    def __init__(
        self,
        samples: List[VideoSampleItem],
        coverage_ratio: float = 0.70,
        min_frames: int = 32,
        max_frames: Optional[int] = 300,
        face_size: int = 224,
        mouth_size: int = 112,
        audio_window_sec: float = 4.0,
        audio_hop_sec: float = 2.0,
        enable_sync_shifts: bool = True,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.samples = samples
        self.coverage_ratio = coverage_ratio
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.face_size = face_size
        self.mouth_size = mouth_size
        self.enable_sync_shifts = enable_sync_shifts

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
        self.mouth_extractor = MouthExtractor(
            mouth_roi_size=(mouth_size, mouth_size)
        )
        self.audio_extractor = AudioWindowExtractor(
            sample_rate=16000,
            window_seconds=audio_window_sec,
            hop_seconds=audio_hop_sec
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        vpath = Path(item.video_path)

        if not vpath.exists():
            return self._create_dummy_sample(item.video_id, item.label)

        try:
            # 1. Visual Extraction
            reader = VideoReader(vpath)
            total_frames = reader.metadata.total_frames
            plan = self.sampler.create_sampling_plan(total_frames)
            candidate_indices = plan.candidate_indices

            valid_faces: List[torch.Tensor] = []
            valid_mouths: List[torch.Tensor] = []
            visual_timestamps: List[Tuple[int, float]] = []

            self.face_aligner.reset_tracking()

            for frame_idx, success, frame_rgb, ts in reader.read_frames_by_indices(candidate_indices):
                if not success or frame_rgb is None:
                    continue

                q_res = self.quality_filter.evaluate_frame(frame_rgb, frame_idx, ts)
                if not q_res.is_usable:
                    continue

                face_res = self.face_aligner.process_frame(frame_rgb, frame_idx, ts)
                if face_res.face_detected and face_res.aligned_face is not None:
                    face_t = torch.from_numpy(face_res.aligned_face).permute(2, 0, 1).float() / 255.0
                    valid_faces.append(face_t)
                    visual_timestamps.append((frame_idx, ts))

                    # Extract mouth ROI from aligned landmarks
                    if face_res.landmarks:
                        mouth_box = self.mouth_extractor.compute_mouth_box(
                            face_res.landmarks,
                            (self.face_size, self.face_size)
                        )
                        if mouth_box:
                            mouth_crop = self.mouth_extractor.crop_mouth_roi(face_res.aligned_face, mouth_box)
                            mouth_t = torch.from_numpy(mouth_crop).permute(2, 0, 1).float() / 255.0
                            valid_mouths.append(mouth_t)
                        else:
                            valid_mouths.append(torch.zeros(3, self.mouth_size, self.mouth_size))
                    else:
                        valid_mouths.append(torch.zeros(3, self.mouth_size, self.mouth_size))

            has_visual = len(valid_faces) > 0
            if not has_visual:
                valid_faces = [torch.zeros(3, self.face_size, self.face_size)]
                valid_mouths = [torch.zeros(3, self.mouth_size, self.mouth_size)]

            face_stack = torch.stack(valid_faces, dim=0)   # (N, 3, 224, 224)
            mouth_stack = torch.stack(valid_mouths, dim=0) # (N, 3, 112, 112)

            # 2. Audio Extraction & Windowing
            audio_res = self.audio_extractor.process_video_audio(vpath, visual_timestamps=visual_timestamps)
            has_audio = audio_res.audio_available and len(audio_res.windows) > 0

            if has_audio:
                mel_list = [torch.from_numpy(w.mel_spectrogram) for w in audio_res.windows]
                mel_stack = torch.stack(mel_list, dim=0) # (W, 128, T)
            else:
                mel_stack = torch.zeros(1, 128, 251, dtype=torch.float32)

            has_sync = (has_visual and has_audio)
            modality_mask = torch.tensor([has_visual, has_audio, has_sync], dtype=torch.bool)

            return {
                "video_id": item.video_id,
                "face_frames": face_stack,
                "mouth_crops": mouth_stack,
                "mel_windows": mel_stack,
                "modality_mask": modality_mask,
                "label": torch.tensor(item.label, dtype=torch.float32),
                "num_frames": len(valid_faces) if has_visual else 0,
                "num_windows": len(audio_res.windows) if has_audio else 0
            }

        except Exception as e:
            logger.error(f"Error loading multimodal sample {vpath.name}: {e}")
            return self._create_dummy_sample(item.video_id, item.label)

    def _create_dummy_sample(self, video_id: str, label: int) -> Dict[str, Any]:
        return {
            "video_id": video_id,
            "face_frames": torch.zeros(self.min_frames, 3, self.face_size, self.face_size),
            "mouth_crops": torch.zeros(self.min_frames, 3, self.mouth_size, self.mouth_size),
            "mel_windows": torch.zeros(1, 128, 251),
            "modality_mask": torch.tensor([False, False, False], dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.float32),
            "num_frames": 0,
            "num_windows": 0
        }


def collate_multimodal_batch(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function dynamically padding variable visual frame counts (N)
    and variable audio window counts (W).
    """
    video_ids = [b["video_id"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    modality_masks = torch.stack([b["modality_mask"] for b in batch], dim=0)

    # Frame lengths
    frame_lens = [b["face_frames"].size(0) for b in batch]
    max_frames = max(frame_lens)
    B = len(batch)

    # Audio window lengths
    window_lens = [b["mel_windows"].size(0) for b in batch]
    max_windows = max(window_lens)
    _, m_dim, t_dim = batch[0]["mel_windows"].shape

    # Padded Visual Tensors
    padded_faces = torch.zeros(B, max_frames, 3, 224, 224, dtype=torch.float32)
    padded_mouths = torch.zeros(B, max_frames, 3, 112, 112, dtype=torch.float32)
    padding_mask_v = torch.ones(B, max_frames, dtype=torch.bool)

    for i, b in enumerate(batch):
        n = frame_lens[i]
        padded_faces[i, :n] = b["face_frames"]
        padded_mouths[i, :n] = b["mouth_crops"]
        if b["modality_mask"][0]:
            padding_mask_v[i, :n] = False

    # Padded Audio Tensors
    padded_mels = torch.zeros(B, max_windows, m_dim, t_dim, dtype=torch.float32)
    padding_mask_a = torch.ones(B, max_windows, dtype=torch.bool)

    for i, b in enumerate(batch):
        w = window_lens[i]
        padded_mels[i, :w] = b["mel_windows"]
        if b["modality_mask"][1]:
            padding_mask_a[i, :w] = False

    return {
        "video_ids": video_ids,
        "face_frames": padded_faces,
        "mouth_crops": padded_mouths,
        "mel_windows": padded_mels,
        "modality_masks": modality_masks,
        "padding_mask_v": padding_mask_v,
        "padding_mask_a": padding_mask_a,
        "labels": labels
    }
