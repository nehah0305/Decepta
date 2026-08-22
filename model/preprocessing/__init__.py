"""
Preprocessing module for Multimodal Deepfake Detection System.

Provides tools to independently extract video frames and audio streams from video files,
maintaining strict temporal synchronization and preserving forensic artifact integrity.
"""

from .audio_processor import AudioProcessingResult, AudioProcessor
from .ffmpeg_utils import (
    AudioStreamInfo,
    FFmpegExecutionError,
    FFmpegNotFoundError,
    MediaInfo,
    VideoStreamInfo,
    get_ffmpeg_path,
    probe_media,
    run_ffmpeg_command,
)
from .pipeline import (
    MultimodalPreprocessor,
    PreprocessingOutput,
    preprocess_video,
)
from .video_processor import FrameInfo, VideoProcessingResult, VideoProcessor

__all__ = [
    "MultimodalPreprocessor",
    "preprocess_video",
    "PreprocessingOutput",
    "VideoProcessor",
    "VideoProcessingResult",
    "FrameInfo",
    "AudioProcessor",
    "AudioProcessingResult",
    "MediaInfo",
    "VideoStreamInfo",
    "AudioStreamInfo",
    "get_ffmpeg_path",
    "probe_media",
    "run_ffmpeg_command",
    "FFmpegNotFoundError",
    "FFmpegExecutionError",
]
