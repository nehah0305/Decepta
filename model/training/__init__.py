"""
Training Module for Visual Deepfake Detection System.
"""

from .dataset import VideoDeepfakeDataset, VideoSampleItem, collate_variable_video_batch, split_videos_by_id
from .losses import DeepfakeDetectionLoss
from .train import train_visual_model
from .validate import compute_binary_metrics, validate_epoch

__all__ = [
    "VideoDeepfakeDataset",
    "VideoSampleItem",
    "collate_variable_video_batch",
    "split_videos_by_id",
    "DeepfakeDetectionLoss",
    "train_visual_model",
    "validate_epoch",
    "compute_binary_metrics",
]
