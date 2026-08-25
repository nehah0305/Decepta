"""
Preprocessing module for Multimodal Deepfake Detection System.

Provides tools to independently extract video frames and audio streams,
perform MTCNN face and 5-point landmark detection, canonical face alignment,
temporal mouth ROI sequence extraction, and 16 kHz 4s audio standardization
with Mel-spectrogram (.npy) generation.
"""

from .audio_preprocessor import (
    AudioPreprocessor,
    AudioPreprocessorConfig,
    AudioPreprocessorResult,
    preprocess_audio_file,
)
from .audio_windowing import AudioWindowData, AudioWindowExtractor, VideoAudioResult
from .audio_processor import AudioProcessingResult, AudioProcessor
from .video_processor import FrameInfo, VideoProcessingResult, VideoProcessor
from .dataset_preprocessor import (
    VideoFacePreprocessor,
    VideoPreprocessingOutput,
    preprocess_frame_sequence,
)
from .face_aligner import AlignedFaceResult, FaceAligner
from .face_detector import FaceDetection, FaceDetector, FrameFaceResult
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
from .mouth_extractor import MouthExtractor, MouthROIResult
from .pipeline import (
    MultimodalPreprocessor,
    PreprocessingOutput,
    preprocess_video,
)
from .video_reader import VideoMetadata, VideoReader
from .frame_sampler import HighCoverageFrameSampler, SamplingPlan
from .frame_quality import FrameQualityFilter, FrameQualityResult
from .face_alignment import AlignedFaceData, FaceAlignmentPipeline

__all__ = [
    # Visual High-Coverage Pipeline
    "VideoReader",
    "VideoMetadata",
    "HighCoverageFrameSampler",
    "SamplingPlan",
    "FrameQualityFilter",
    "FrameQualityResult",
    "FaceAlignmentPipeline",
    "AlignedFaceData",
    # Audio Windowing Pipeline
    "AudioWindowExtractor",
    "AudioWindowData",
    "VideoAudioResult",
    # Video & Audio Modality Extraction
    "MultimodalPreprocessor",
    "preprocess_video",
    "PreprocessingOutput",
    "VideoProcessor",
    "VideoProcessingResult",
    "FrameInfo",
    "AudioProcessor",
    "AudioProcessingResult",
    # Face & Mouth ROI Pipeline
    "FaceDetector",
    "FaceDetection",
    "FrameFaceResult",
    "FaceAligner",
    "AlignedFaceResult",
    "MouthExtractor",
    "MouthROIResult",
    "VideoFacePreprocessor",
    "VideoPreprocessingOutput",
    "preprocess_frame_sequence",
    # Audio Preprocessing Stage (16 kHz, 4s, Mel-Spectrogram)
    "AudioPreprocessor",
    "AudioPreprocessorConfig",
    "AudioPreprocessorResult",
    "preprocess_audio_file",
    # Utilities
    "MediaInfo",
    "VideoStreamInfo",
    "AudioStreamInfo",
    "get_ffmpeg_path",
    "probe_media",
    "run_ffmpeg_command",
    "FFmpegNotFoundError",
    "FFmpegExecutionError",
]

