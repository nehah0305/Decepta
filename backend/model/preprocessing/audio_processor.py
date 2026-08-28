"""
Audio Processor Module for Multimodal Deepfake Detection Preprocessing.

Extracts the audio stream from video files independently into lossless WAV format,
maintaining temporal alignment with the video stream.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

from .ffmpeg_utils import (
    AudioStreamInfo,
    FFmpegExecutionError,
    MediaInfo,
    get_ffmpeg_path,
    probe_media,
    run_ffmpeg_command,
)


@dataclass
class AudioProcessingResult:
    """Consolidated result of audio extraction."""
    input_video_path: str
    output_audio_path: Optional[str]
    has_audio: bool
    sample_rate: Optional[int]
    channels: Optional[int]
    codec: Optional[str]
    duration_seconds: float
    format: str = "wav"


class AudioProcessor:
    """
    Handles audio extraction from video files using FFmpeg.
    Saves uncompressed PCM 16-bit WAV with strict temporal alignment.
    """

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        ffmpeg_path: Optional[str] = None
    ):
        """
        Args:
            sample_rate: Optional target sample rate in Hz (e.g. 16000 for speech models, 44100, or None to keep native).
            channels: Optional target channel count (e.g. 1 for mono, 2 for stereo, or None to keep native).
            ffmpeg_path: Optional custom path to FFmpeg binary.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.ffmpeg_path = ffmpeg_path or get_ffmpeg_path()

    def extract_audio(
        self,
        video_path: Union[str, Path],
        output_file_path: Union[str, Path],
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        allow_no_audio: bool = True
    ) -> AudioProcessingResult:
        """
        Extracts the audio stream from the video file and saves it as a WAV file.

        Args:
            video_path: Path to the input video file.
            output_file_path: Path where audio.wav will be stored (e.g. output/audio/audio.wav).
            sample_rate: Target sample rate override (e.g. 16000, 44100, None for native).
            channels: Target channels override (e.g. 1, 2, None for native).
            allow_no_audio: If False, raises ValueError when video has no audio stream.

        Returns:
            AudioProcessingResult with metadata.
        """
        resolved_video_path = str(Path(video_path).resolve())
        out_file = Path(output_file_path).resolve()

        if not os.path.isfile(resolved_video_path):
            raise FileNotFoundError(f"Input video file not found: {resolved_video_path}")

        if out_file.suffix.lower() != ".wav":
            raise ValueError(f"Audio output file must have .wav extension, got: {out_file.name}")

        # Probe video for audio stream
        media_info: MediaInfo = probe_media(resolved_video_path, ffmpeg_path=self.ffmpeg_path)

        if not media_info.has_audio or media_info.audio_stream is None:
            if not allow_no_audio:
                raise ValueError(f"The input video does not contain an audio stream: {resolved_video_path}")
            return AudioProcessingResult(
                input_video_path=resolved_video_path,
                output_audio_path=None,
                has_audio=False,
                sample_rate=None,
                channels=None,
                codec=None,
                duration_seconds=0.0
            )

        audio_stream: AudioStreamInfo = media_info.audio_stream
        sr_to_use = sample_rate if sample_rate is not None else self.sample_rate
        ch_to_use = channels if channels is not None else self.channels

        # Ensure output directory exists
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Build FFmpeg command for pristine audio extraction:
        # - "-vn": Disable video stream
        # - "-acodec pcm_s16le": Uncompressed 16-bit Linear PCM in WAV container
        # - "-ar {sr}": Resample if target sample rate provided
        # - "-ac {ch}": Channel count if provided
        # - Temporal alignment: starts from 0:00:00 exactly synchronized with video frame 1
        ffmpeg_args = [
            "-hide_banner",
            "-y",
            "-i", resolved_video_path,
            "-vn",
            "-acodec", "pcm_s16le"
        ]

        if sr_to_use is not None:
            ffmpeg_args.extend(["-ar", str(sr_to_use)])
        if ch_to_use is not None:
            ffmpeg_args.extend(["-ac", str(ch_to_use)])

        ffmpeg_args.append(str(out_file))

        run_ffmpeg_command(ffmpeg_args, ffmpeg_path=self.ffmpeg_path, check=True)

        if not out_file.is_file() or out_file.stat().st_size == 0:
            raise RuntimeError(f"Failed to create extracted audio file at {out_file}")

        # Probe extracted WAV to confirm exact output properties
        extracted_probe = probe_media(str(out_file), ffmpeg_path=self.ffmpeg_path)
        final_sr = extracted_probe.audio_stream.sample_rate if extracted_probe.audio_stream else (sr_to_use or audio_stream.sample_rate)
        final_ch = extracted_probe.audio_stream.channels if extracted_probe.audio_stream else (ch_to_use or audio_stream.channels)
        final_dur = extracted_probe.duration if extracted_probe.duration > 0 else media_info.duration

        return AudioProcessingResult(
            input_video_path=resolved_video_path,
            output_audio_path=str(out_file),
            has_audio=True,
            sample_rate=final_sr,
            channels=final_ch,
            codec="pcm_s16le",
            duration_seconds=round(final_dur, 4),
            format="wav"
        )
