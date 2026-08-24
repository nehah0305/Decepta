"""
Multimodal Preprocessing Pipeline for Deepfake Detection System.

Orchestrates independent video frame extraction and audio stream extraction,
preserving temporal alignment and recording complete forensic metadata in JSON format.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .audio_processor import AudioProcessingResult, AudioProcessor
from .ffmpeg_utils import get_ffmpeg_path
from .video_processor import VideoProcessingResult, VideoProcessor


@dataclass
class PreprocessingOutput:
    """Consolidated multimodal preprocessing output data model."""
    status: str  # "success" | "warning" | "error"
    input_video_path: str
    output_directory: str
    frames_directory: str
    audio_file_path: Optional[str]
    metadata_file_path: str
    
    # Video metadata
    original_fps: float
    extracted_fps: float
    resolution: Dict[str, int]  # {"width": int, "height": int}
    video_duration_seconds: float
    total_original_frames: int
    total_extracted_frames: int
    frame_filenames: List[str]
    frame_timestamps: Dict[str, float]
    
    # Audio metadata
    has_audio: bool
    audio_sample_rate: Optional[int]
    audio_channels: Optional[int]
    audio_duration_seconds: Optional[float]
    audio_format: Optional[str]
    
    # Execution metadata
    processing_time_seconds: float
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object to dictionary."""
        return asdict(self)

    def save_json(self, path: Union[str, Path]) -> str:
        """Save metadata dictionary to a JSON file."""
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return str(target)


class MultimodalPreprocessor:
    """
    Multimodal Preprocessor for deepfake detection systems.
    Separates video and audio modalities into output/frames/ and output/audio/
    while preserving temporal alignment and forensic integrity.
    """

    def __init__(
        self,
        extracted_fps: float = 25.0,
        audio_sample_rate: Optional[int] = None,
        audio_channels: Optional[int] = None,
        ffmpeg_path: Optional[str] = None
    ):
        """
        Args:
            extracted_fps: Frame extraction rate in FPS (default: 25.0).
            audio_sample_rate: Audio sampling rate in Hz (e.g. 16000, 44100, or None for native).
            audio_channels: Audio channel count (e.g. 1, 2, or None for native).
            ffmpeg_path: Custom path to FFmpeg binary if needed.
        """
        self.extracted_fps = extracted_fps
        self.audio_sample_rate = audio_sample_rate
        self.audio_channels = audio_channels
        self.ffmpeg_path = ffmpeg_path or get_ffmpeg_path()

        self.video_processor = VideoProcessor(
            target_fps=self.extracted_fps,
            ffmpeg_path=self.ffmpeg_path
        )
        self.audio_processor = AudioProcessor(
            sample_rate=self.audio_sample_rate,
            channels=self.audio_channels,
            ffmpeg_path=self.ffmpeg_path
        )

    def process(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path] = "output",
        extracted_fps: Optional[float] = None,
        audio_sample_rate: Optional[int] = None,
        save_metadata: bool = True
    ) -> PreprocessingOutput:
        """
        Processes a single input video, extracting video frames to output/frames/
        and audio to output/audio/audio.wav.

        Args:
            video_path: Path to input video file.
            output_dir: Root output directory (default: "output").
            extracted_fps: Override frame rate for this run.
            audio_sample_rate: Override audio sample rate for this run.
            save_metadata: Whether to write metadata.json inside output_dir.

        Returns:
            PreprocessingOutput object.
        """
        start_time = time.time()
        resolved_video_path = str(Path(video_path).resolve())
        root_output_path = Path(output_dir).resolve()

        frames_dir = root_output_path / "frames"
        audio_dir = root_output_path / "audio"
        audio_output_file = audio_dir / "audio.wav"
        metadata_file = root_output_path / "metadata.json"

        # 1. Video Processing - Frame Extraction
        target_fps = extracted_fps if extracted_fps is not None else self.extracted_fps
        video_result: VideoProcessingResult = self.video_processor.extract_frames(
            video_path=resolved_video_path,
            output_dir=frames_dir,
            target_fps=target_fps
        )

        # 2. Audio Processing - Audio Extraction
        sr = audio_sample_rate if audio_sample_rate is not None else self.audio_sample_rate
        audio_result: AudioProcessingResult = self.audio_processor.extract_audio(
            video_path=resolved_video_path,
            output_file_path=audio_output_file,
            sample_rate=sr,
            allow_no_audio=True
        )

        elapsed_time = round(time.time() - start_time, 4)
        iso_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # 3. Consolidate Metadata
        output = PreprocessingOutput(
            status="success" if audio_result.has_audio else "warning",
            input_video_path=resolved_video_path,
            output_directory=str(root_output_path),
            frames_directory=str(frames_dir),
            audio_file_path=audio_result.output_audio_path,
            metadata_file_path=str(metadata_file),
            original_fps=video_result.original_fps,
            extracted_fps=video_result.extracted_fps,
            resolution=video_result.resolution,
            video_duration_seconds=video_result.duration_seconds,
            total_original_frames=video_result.total_original_frames,
            total_extracted_frames=video_result.total_extracted_frames,
            frame_filenames=video_result.frame_filenames,
            frame_timestamps=video_result.frame_timestamps,
            has_audio=audio_result.has_audio,
            audio_sample_rate=audio_result.sample_rate,
            audio_channels=audio_result.channels,
            audio_duration_seconds=audio_result.duration_seconds if audio_result.has_audio else 0.0,
            audio_format=audio_result.format if audio_result.has_audio else None,
            processing_time_seconds=elapsed_time,
            timestamp_utc=iso_timestamp
        )

        if save_metadata:
            output.save_json(metadata_file)

        return output


def preprocess_video(
    video_path: Union[str, Path],
    output_dir: Union[str, Path] = "output",
    fps: float = 25.0,
    audio_sample_rate: Optional[int] = None,
    save_metadata: bool = True
) -> PreprocessingOutput:
    """
    Convenience function to run multimodal preprocessing on a video file.

    Args:
        video_path: Path to the input video.
        output_dir: Directory where 'frames/', 'audio/', and 'metadata.json' are saved.
        fps: Configurable extraction frame rate (default: 25.0 FPS).
        audio_sample_rate: Target sample rate for WAV extraction (optional).
        save_metadata: Whether to save output/metadata.json.

    Returns:
        PreprocessingOutput with full forensic metadata.
    """
    preprocessor = MultimodalPreprocessor(
        extracted_fps=fps,
        audio_sample_rate=audio_sample_rate
    )
    return preprocessor.process(
        video_path=video_path,
        output_dir=output_dir,
        save_metadata=save_metadata
    )
