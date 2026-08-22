"""
FFmpeg Utilities for Multimodal Deepfake Detection Preprocessing Pipeline.

Handles automatic FFmpeg binary discovery, media file probing, and command execution.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class VideoStreamInfo:
    """Metadata for a video stream."""
    codec: str
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int
    pix_fmt: Optional[str] = None
    bitrate_kbps: Optional[int] = None


@dataclass
class AudioStreamInfo:
    """Metadata for an audio stream."""
    codec: str
    sample_rate: int
    channels: int
    duration: float
    bitrate_kbps: Optional[int] = None


@dataclass
class MediaInfo:
    """Consolidated media file information."""
    file_path: str
    duration: float
    video_stream: Optional[VideoStreamInfo] = None
    audio_stream: Optional[AudioStreamInfo] = None
    has_video: bool = False
    has_audio: bool = False


class FFmpegNotFoundError(FileNotFoundError):
    """Raised when FFmpeg binary cannot be located."""
    pass


class FFmpegExecutionError(RuntimeError):
    """Raised when an FFmpeg command fails to execute."""
    pass


def get_ffmpeg_path(custom_path: Optional[Union[str, Path]] = None) -> str:
    """
    Finds the path to the FFmpeg executable.
    
    Search order:
    1. custom_path argument if provided.
    2. FFMPEG_PATH or FFMPEG_BINARY environment variable.
    3. System PATH (shutil.which).
    4. imageio-ffmpeg bundled binary.

    Returns:
        Absolute path to the ffmpeg executable.

    Raises:
        FFmpegNotFoundError: If ffmpeg cannot be found.
    """
    if custom_path:
        path = Path(custom_path)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        raise FFmpegNotFoundError(f"Provided custom FFmpeg path is invalid: {custom_path}")

    # Check environment variables
    for env_var in ("FFMPEG_PATH", "FFMPEG_BINARY"):
        env_path = os.environ.get(env_var)
        if env_path and os.path.isfile(env_path):
            return str(Path(env_path).resolve())

    # Check system PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return str(Path(system_ffmpeg).resolve())

    # Check imageio_ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and os.path.isfile(ffmpeg_exe):
            return str(Path(ffmpeg_exe).resolve())
    except ImportError:
        pass

    raise FFmpegNotFoundError(
        "FFmpeg executable not found. Please install FFmpeg or set FFMPEG_PATH environment variable."
    )


def run_ffmpeg_command(
    cmd_args: List[str],
    ffmpeg_path: Optional[str] = None,
    check: bool = True,
    capture_output: bool = True
) -> subprocess.CompletedProcess:
    """
    Executes an FFmpeg command with standard error logging.

    Args:
        cmd_args: Arguments to pass to FFmpeg (excluding the executable name).
        ffmpeg_path: Path to FFmpeg executable (auto-discovered if None).
        check: Whether to raise FFmpegExecutionError on non-zero exit code.
        capture_output: Whether to capture stdout and stderr.

    Returns:
        CompletedProcess object containing returncode, stdout, and stderr.
    """
    exe = ffmpeg_path or get_ffmpeg_path()
    full_cmd = [exe] + cmd_args

    process = subprocess.run(
        full_cmd,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if check and process.returncode != 0:
        error_msg = (
            f"FFmpeg command failed with exit code {process.returncode}.\n"
            f"Command: {' '.join(full_cmd)}\n"
            f"Error Output:\n{process.stderr}"
        )
        raise FFmpegExecutionError(error_msg)

    return process


def _parse_time_to_seconds(time_str: str) -> float:
    """Converts HH:MM:SS.ms string to seconds."""
    parts = time_str.strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = float(parts[0]), float(parts[1])
        return minutes * 60 + seconds
    return float(parts[0])


def probe_media(file_path: Union[str, Path], ffmpeg_path: Optional[str] = None) -> MediaInfo:
    """
    Probes a media file to retrieve comprehensive video and audio stream information.
    Uses FFmpeg information output combined with cv2 fallback if needed.

    Args:
        file_path: Path to the input video/audio file.
        ffmpeg_path: Optional path to FFmpeg binary.

    Returns:
        MediaInfo object with detailed metadata.
    """
    resolved_path = str(Path(file_path).resolve())
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Input media file not found: {resolved_path}")

    # Run ffmpeg -i <file> to get stream metadata (output is in stderr)
    process = run_ffmpeg_command(
        ["-hide_banner", "-i", resolved_path],
        ffmpeg_path=ffmpeg_path,
        check=False,
        capture_output=True
    )

    stderr_text = process.stderr

    duration = 0.0
    # Match container duration: Duration: 00:00:05.10, start: 0.000000, bitrate: 1024 kb/s
    duration_match = re.search(r"Duration:\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)", stderr_text)
    if duration_match:
        duration = _parse_time_to_seconds(duration_match.group(1))

    # Parse Video stream
    # Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1280x720 [SAR 1:1 DAR 16:9], 30 fps, 30 tbr, ...
    video_stream_match = re.search(
        r"Stream\s*#\d+:\d+.*?: Video:\s*([^,]+),\s*([^,]+),\s*(\d+)x(\d+).*?,\s*([\d\.]+)\s*(?:fps|tbr)",
        stderr_text
    )

    video_info: Optional[VideoStreamInfo] = None
    if video_stream_match:
        codec = video_stream_match.group(1).strip()
        pix_fmt = video_stream_match.group(2).strip()
        width = int(video_stream_match.group(3))
        height = int(video_stream_match.group(4))
        fps = float(video_stream_match.group(5))
        total_frames = int(round(duration * fps)) if duration > 0 and fps > 0 else 0

        video_info = VideoStreamInfo(
            codec=codec,
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            total_frames=total_frames,
            pix_fmt=pix_fmt
        )
    else:
        # Fallback to general video stream matching if layout differs
        alt_video_match = re.search(
            r"Stream\s*#\d+:\d+.*?: Video:\s*([^,\n]+).*?,\s*(\d+)x(\d+)",
            stderr_text
        )
        if alt_video_match:
            codec = alt_video_match.group(1).strip()
            width = int(alt_video_match.group(2))
            height = int(alt_video_match.group(3))
            
            fps_match = re.search(r"([\d\.]+)\s*fps", stderr_text)
            fps = float(fps_match.group(1)) if fps_match else 30.0
            total_frames = int(round(duration * fps)) if duration > 0 else 0

            video_info = VideoStreamInfo(
                codec=codec,
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                total_frames=total_frames
            )

    # If OpenCV is available, refine total_frames, fps, and resolution
    try:
        import cv2
        cap = cv2.VideoCapture(resolved_path)
        if cap.isOpened():
            cv_fps = cap.get(cv2.CAP_PROP_FPS)
            cv_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cv_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cv_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if cv_fps > 0 and cv_width > 0 and cv_height > 0:
                if video_info:
                    if cv_fps > 0:
                        video_info.fps = cv_fps
                    if cv_frames > 0:
                        video_info.total_frames = cv_frames
                    if cv_width > 0 and cv_height > 0:
                        video_info.width = cv_width
                        video_info.height = cv_height
                else:
                    video_info = VideoStreamInfo(
                        codec="unknown",
                        width=cv_width,
                        height=cv_height,
                        fps=cv_fps,
                        duration=duration if duration > 0 else (cv_frames / cv_fps if cv_fps > 0 else 0.0),
                        total_frames=cv_frames
                    )
    except Exception:
        pass

    # Parse Audio stream
    # Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 128 kb/s
    audio_info: Optional[AudioStreamInfo] = None
    audio_stream_match = re.search(
        r"Stream\s*#\d+:\d+.*?: Audio:\s*([^,]+),\s*(\d+)\s*Hz,\s*([^,]+)",
        stderr_text
    )

    if audio_stream_match:
        a_codec = audio_stream_match.group(1).strip()
        sample_rate = int(audio_stream_match.group(2))
        channels_str = audio_stream_match.group(3).strip().lower()

        if "stereo" in channels_str:
            channels = 2
        elif "mono" in channels_str:
            channels = 1
        elif "5.1" in channels_str:
            channels = 6
        elif "7.1" in channels_str:
            channels = 8
        else:
            # Try matching digits e.g. "2 channels"
            chan_digit_match = re.search(r"(\d+)", channels_str)
            channels = int(chan_digit_match.group(1)) if chan_digit_match else 2

        audio_info = AudioStreamInfo(
            codec=a_codec,
            sample_rate=sample_rate,
            channels=channels,
            duration=duration
        )

    return MediaInfo(
        file_path=resolved_path,
        duration=duration,
        video_stream=video_info,
        audio_stream=audio_info,
        has_video=video_info is not None,
        has_audio=audio_info is not None
    )
