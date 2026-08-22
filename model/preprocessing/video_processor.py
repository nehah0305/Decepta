"""
Video Processor Module for Multimodal Deepfake Detection Preprocessing.

Extracts frames at a configurable frame rate using FFmpeg in lossless PNG format,
preserving chronological order, timestamps, and forensic image artifacts.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .ffmpeg_utils import (
    FFmpegExecutionError,
    MediaInfo,
    VideoStreamInfo,
    get_ffmpeg_path,
    probe_media,
    run_ffmpeg_command,
)


@dataclass
class FrameInfo:
    """Metadata for a single extracted frame."""
    frame_index: int
    filename: str
    relative_path: str
    timestamp_seconds: float


@dataclass
class VideoProcessingResult:
    """Consolidated result of video frame extraction."""
    input_video_path: str
    output_dir: str
    original_fps: float
    extracted_fps: float
    resolution: Dict[str, int]  # {"width": int, "height": int}
    duration_seconds: float
    total_original_frames: int
    total_extracted_frames: int
    frame_filenames: List[str]
    frame_timestamps: Dict[str, float]  # filename -> timestamp in seconds
    frames_metadata: List[FrameInfo] = field(default_factory=list)


class VideoProcessor:
    """
    Handles frame extraction from video streams using FFmpeg.
    Ensures strict lossless PNG preservation, sequential naming, and metadata tracking.
    """

    def __init__(
        self,
        target_fps: float = 5.0,
        ffmpeg_path: Optional[str] = None,
        filename_pattern: str = "frame_%06d.png"
    ):
        """
        Args:
            target_fps: Target frame rate for extraction (e.g., 5.0, 10.0 FPS).
            ffmpeg_path: Optional custom path to FFmpeg binary.
            filename_pattern: Pattern for frame filenames (must produce .png files).
        """
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if not filename_pattern.endswith(".png"):
            raise ValueError(f"filename_pattern must have .png extension, got {filename_pattern}")

        self.target_fps = float(target_fps)
        self.ffmpeg_path = ffmpeg_path or get_ffmpeg_path()
        self.filename_pattern = filename_pattern

    def extract_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        target_fps: Optional[float] = None
    ) -> VideoProcessingResult:
        """
        Extracts frames from the video at the specified target FPS.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory where extracted frames will be stored (e.g. output/frames/).
            target_fps: Override target frame rate if specified.

        Returns:
            VideoProcessingResult with full metadata.
        """
        resolved_video_path = str(Path(video_path).resolve())
        fps_to_use = float(target_fps) if target_fps is not None else self.target_fps

        if not os.path.isfile(resolved_video_path):
            raise FileNotFoundError(f"Input video file not found: {resolved_video_path}")

        # Probe video information
        media_info: MediaInfo = probe_media(resolved_video_path, ffmpeg_path=self.ffmpeg_path)
        if not media_info.has_video or media_info.video_stream is None:
            raise ValueError(f"The input file does not contain a valid video stream: {resolved_video_path}")

        video_stream: VideoStreamInfo = media_info.video_stream

        # Ensure frames output directory exists
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Clear any existing frames in output directory to prevent mixing
        for old_file in out_path.glob("frame_*.png"):
            try:
                old_file.unlink()
            except OSError:
                pass

        # Build FFmpeg command
        # Forensic Safety Considerations:
        # - "-vf fps=fps={fps_to_use}:round=near": Extract frames at exact target rate without resizing or smoothing.
        # - "-compression_level 2": Standard PNG lossless compression without modifying pixels.
        # - "-pred mixed": standard lossless prediction filter.
        # - No sharpening, denoising, scaling, or color transformations are applied.
        output_template = str(out_path / self.filename_pattern)

        ffmpeg_args = [
            "-hide_banner",
            "-y",  # overwrite output files if any
            "-i", resolved_video_path,
            "-vf", f"fps=fps={fps_to_use}:round=near",
            "-compression_level", "2",
            "-pred", "mixed",
            output_template
        ]

        run_ffmpeg_command(ffmpeg_args, ffmpeg_path=self.ffmpeg_path, check=True)

        # Discover and sort all extracted PNG frames chronologically
        extracted_files = sorted(out_path.glob("frame_*.png"))
        if not extracted_files:
            raise RuntimeError(f"No frames were extracted to {out_path}")

        # Build timestamps and metadata for each frame
        frame_filenames: List[str] = []
        frame_timestamps: Dict[str, float] = {}
        frames_metadata: List[FrameInfo] = []

        total_extracted = len(extracted_files)
        for idx, file_path in enumerate(extracted_files, start=1):
            fname = file_path.name
            # Timestamp calculated based on target extracted frame rate:
            # Frame 1 corresponds to t = 0.0s, Frame 2 to t = 1/fps, etc.
            timestamp = (idx - 1) / fps_to_use
            # Cap timestamp at video duration if duration is known
            if video_stream.duration > 0 and timestamp > video_stream.duration:
                timestamp = video_stream.duration

            timestamp = round(timestamp, 4)

            frame_filenames.append(fname)
            frame_timestamps[fname] = timestamp
            frames_metadata.append(
                FrameInfo(
                    frame_index=idx,
                    filename=fname,
                    relative_path=str(Path("frames") / fname),
                    timestamp_seconds=timestamp
                )
            )

        return VideoProcessingResult(
            input_video_path=resolved_video_path,
            output_dir=str(out_path.resolve()),
            original_fps=video_stream.fps,
            extracted_fps=fps_to_use,
            resolution={"width": video_stream.width, "height": video_stream.height},
            duration_seconds=round(video_stream.duration, 4),
            total_original_frames=video_stream.total_frames,
            total_extracted_frames=total_extracted,
            frame_filenames=frame_filenames,
            frame_timestamps=frame_timestamps,
            frames_metadata=frames_metadata
        )
