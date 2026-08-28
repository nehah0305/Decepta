"""
Training Module for Visual Deepfake Detection System.
"""

from .dataset import VideoDeepfakeDataset, VideoSampleItem, collate_variable_video_batch, split_videos_by_id
from .multimodal_dataset import MultimodalVideoDataset, collate_multimodal_batch
from .losses import DeepfakeDetectionLoss, InfoNCESyncLoss, AudioVisualSyncLoss, MultimodalCompoundLoss
from .train import train_visual_model
from .train_audio import train_audio_stage2
from .train_sync_pretrain import train_sync_stage3
from .train_multimodal import train_multimodal_model, validate_multimodal_epoch
from .validate import compute_binary_metrics, validate_epoch

__all__ = [
    "VideoDeepfakeDataset",
    "VideoSampleItem",
    "collate_variable_video_batch",
    "split_videos_by_id",
    "MultimodalVideoDataset",
    "collate_multimodal_batch",
    "DeepfakeDetectionLoss",
    "InfoNCESyncLoss",
    "AudioVisualSyncLoss",
    "MultimodalCompoundLoss",
    "train_visual_model",
    "train_audio_stage2",
    "train_sync_stage3",
    "train_multimodal_model",
    "validate_epoch",
    "validate_multimodal_epoch",
    "compute_binary_metrics",
]

