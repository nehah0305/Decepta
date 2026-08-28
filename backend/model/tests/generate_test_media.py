"""
Test Media Generator for Multimodal Deepfake Detection Preprocessing.

Generates synthetic video files containing both video and audio streams
for automated testing and verification.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.ffmpeg_utils import get_ffmpeg_path, run_ffmpeg_command


def create_synthetic_multimodal_video(
    output_path: Union[str, Path],
    duration: float = 3.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    include_audio: bool = True,
    audio_sample_rate: int = 44100,
    ffmpeg_path: Optional[str] = None
) -> str:
    """
    Creates a synthetic MP4 video with a test pattern and synchronized sine wave audio.

    Args:
        output_path: Target path for the synthetic video file.
        duration: Duration of video in seconds.
        fps: Video frame rate.
        width: Video width.
        height: Video height.
        include_audio: Whether to include an audio stream.
        audio_sample_rate: Audio sampling rate in Hz.
        ffmpeg_path: Optional path to FFmpeg binary.

    Returns:
        Absolute path to the created video.
    """
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    exe = ffmpeg_path or get_ffmpeg_path()

    # Video filter: test pattern with frame counters and timestamps
    video_filter = f"testsrc=duration={duration}:size={width}x{height}:rate={fps}"

    cmd = [
        "-hide_banner",
        "-y",
        "-f", "lavfi",
        "-i", video_filter
    ]

    if include_audio:
        # Audio filter: 440 Hz A4 tone with exact duration and sample rate
        audio_filter = f"sine=frequency=440:sample_rate={audio_sample_rate}:duration={duration}"
        cmd.extend([
            "-f", "lavfi",
            "-i", audio_filter,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest"
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p"
        ])

    cmd.append(str(target))

    run_ffmpeg_command(cmd, ffmpeg_path=exe, check=True)
    return str(target)


if __name__ == "__main__":
    out = create_synthetic_multimodal_video("sample_test_video.mp4", duration=2.0, fps=25)
    print(f"Generated test video: {out}")
