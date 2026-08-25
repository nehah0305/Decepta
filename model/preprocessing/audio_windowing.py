"""
Audio Extraction, Windowing, and Log-Mel Spectrogram Module.

Implements:
1. Resampling to 16 kHz Mono.
2. 4-second overlapping windows (hop = 2s, 64,000 samples/window) for comprehensive temporal coverage.
3. Dynamic Log-Mel Spectrogram extraction (128 mels, n_fft=1024, hop=256).
4. Audio-to-Visual timestamp mapping.
5. Fault-tolerant execution (missing audio flagged with audio_available = False).
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import librosa
import numpy as np
import soundfile as sf
import torch

from .audio_processor import AudioProcessor
from .ffmpeg_utils import probe_media

logger = logging.getLogger(__name__)


@dataclass
class AudioWindowData:
    """Represents a single 4-second audio window and its spectral representations."""
    video_id: str
    window_id: int
    start_time: float
    end_time: float
    waveform: np.ndarray          # 1D float32 array of shape (64000,)
    mel_spectrogram: np.ndarray   # 2D float32 array of shape (128, T)
    corresponding_frame_indices: List[int] = None

    def to_dict(self) -> Dict:
        return {
            "video_id": self.video_id,
            "window_id": self.window_id,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "waveform_samples": len(self.waveform) if self.waveform is not None else 0,
            "mel_shape": list(self.mel_spectrogram.shape) if self.mel_spectrogram is not None else [],
            "corresponding_frame_indices": self.corresponding_frame_indices or []
        }


@dataclass
class VideoAudioResult:
    """Overall audio extraction result for a video."""
    video_id: str
    audio_available: bool
    total_duration_seconds: float
    sample_rate: int
    num_windows: int
    windows: List[AudioWindowData]
    status: str  # "success", "no_audio", "read_error"


class AudioWindowExtractor:
    """
    Extracts and standardizes audio from videos into 4-second overlapping Log-Mel windows.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_seconds: float = 4.0,
        hop_seconds: float = 2.0,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        top_db: float = 80.0
    ):
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.window_samples = int(round(sample_rate * window_seconds)) # 64000
        self.hop_samples = int(round(sample_rate * hop_seconds))       # 32000
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.top_db = top_db

        self.audio_processor = AudioProcessor()

    def compute_log_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Transforms 16 kHz 1D waveform into (128, T) Log-Mel Spectrogram (dB scale).
        """
        # Ensure exact window size
        if len(waveform) < self.window_samples:
            waveform = np.pad(waveform, (0, self.window_samples - len(waveform)), mode="constant")
        elif len(waveform) > self.window_samples:
            waveform = waveform[:self.window_samples]

        mel_power = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=2.0
        )

        # Log (dB) scale
        mel_db = librosa.power_to_db(mel_power, ref=np.max, top_db=self.top_db)
        if not np.isfinite(mel_db).all():
            mel_db = np.nan_to_num(mel_db, nan=-self.top_db, posinf=0.0, neginf=-self.top_db)

        # Normalization to roughly [-1, 1] range for stable neural processing
        mel_norm = (mel_db + (self.top_db / 2.0)) / (self.top_db / 2.0)
        return mel_norm.astype(np.float32)

    def extract_audio_from_video(
        self,
        video_path: Union[str, Path],
        temp_audio_dir: Optional[Path] = None
    ) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Extracts 16 kHz mono waveform from video using FFmpeg / soundfile.
        """
        vpath = Path(video_path).resolve()
        if not vpath.exists():
            return False, None, 0.0

        try:
            # Check audio stream existence
            media_info = probe_media(vpath)
            if not media_info.has_audio or media_info.audio_stream is None:
                return False, None, 0.0

            # Extract via temporary WAV
            if temp_audio_dir is None:
                temp_audio_dir = Path("outputs/temp_audio")
            temp_audio_dir.mkdir(parents=True, exist_ok=True)
            temp_wav = temp_audio_dir / f"{vpath.stem}_16k.wav"

            res = self.audio_processor.extract_audio(
                video_path=vpath,
                output_file_path=temp_wav,
                sample_rate=self.sample_rate,
                channels=1,
                allow_no_audio=True
            )

            if not res.has_audio or res.output_audio_path is None or not Path(res.output_audio_path).exists():
                return False, None, 0.0

            waveform, sr = sf.read(res.output_audio_path, dtype="float32")
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)

            duration = len(waveform) / float(self.sample_rate)
            return True, waveform.astype(np.float32), duration

        except Exception as e:
            logger.warning(f"Audio extraction failed for {vpath.name}: {e}")
            return False, None, 0.0

    def process_video_audio(
        self,
        video_path: Union[str, Path],
        visual_timestamps: Optional[List[Tuple[int, float]]] = None
    ) -> VideoAudioResult:
        """
        Extracts, windows, and aligns audio from a video file.

        Args:
            video_path: Path to video file.
            visual_timestamps: Optional list of (frame_index, timestamp_seconds) for alignment.

        Returns:
            VideoAudioResult object containing all 4-second overlapping windows.
        """
        vpath = Path(video_path).resolve()
        vid_id = vpath.stem

        has_audio, full_waveform, duration = self.extract_audio_from_video(vpath)
        if not has_audio or full_waveform is None or len(full_waveform) == 0:
            return VideoAudioResult(
                video_id=vid_id,
                audio_available=False,
                total_duration_seconds=0.0,
                sample_rate=self.sample_rate,
                num_windows=0,
                windows=[],
                status="no_audio"
            )

        total_samples = len(full_waveform)
        windows: List[AudioWindowData] = []
        window_id = 0

        # If audio is shorter than window_samples, create single zero-padded window
        if total_samples <= self.window_samples:
            pad_len = self.window_samples - total_samples
            padded = np.pad(full_waveform, (0, pad_len), mode="constant")
            mel = self.compute_log_mel_spectrogram(padded)

            frame_indices = []
            if visual_timestamps:
                frame_indices = [idx for idx, ts in visual_timestamps if 0.0 <= ts <= self.window_seconds]

            windows.append(
                AudioWindowData(
                    video_id=vid_id,
                    window_id=0,
                    start_time=0.0,
                    end_time=self.window_seconds,
                    waveform=padded,
                    mel_spectrogram=mel,
                    corresponding_frame_indices=frame_indices
                )
            )
        else:
            # Overlapping windowing: 0-4s, 2-6s, 4-8s, etc.
            start_sample = 0
            while start_sample < total_samples:
                end_sample = start_sample + self.window_samples
                start_time = start_sample / float(self.sample_rate)
                end_time = end_sample / float(self.sample_rate)

                if end_sample <= total_samples:
                    segment = full_waveform[start_sample:end_sample]
                else:
                    # Pad final window
                    segment = full_waveform[start_sample:]
                    segment = np.pad(segment, (0, self.window_samples - len(segment)), mode="constant")

                mel = self.compute_log_mel_spectrogram(segment)

                frame_indices = []
                if visual_timestamps:
                    frame_indices = [idx for idx, ts in visual_timestamps if start_time <= ts <= end_time]

                windows.append(
                    AudioWindowData(
                        video_id=vid_id,
                        window_id=window_id,
                        start_time=start_time,
                        end_time=end_time,
                        waveform=segment,
                        mel_spectrogram=mel,
                        corresponding_frame_indices=frame_indices
                    )
                )

                window_id += 1
                start_sample += self.hop_samples

                # Avoid tiny redundant tail window if covered
                if start_sample >= total_samples:
                    break

        return VideoAudioResult(
            video_id=vid_id,
            audio_available=True,
            total_duration_seconds=duration,
            sample_rate=self.sample_rate,
            num_windows=len(windows),
            windows=windows,
            status="success"
        )
