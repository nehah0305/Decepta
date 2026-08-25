"""
Audio Preprocessing Module for Multimodal Deepfake Detection.

Implements the standardized audio pipeline:
Extracted Audio -> Load -> Convert to Mono -> Resample to 16 kHz ->
Trim/Pad to 4s (64,000 samples) -> Save WAV & Mel-Spectrogram (.npy) -> Audio Metadata.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import librosa


@dataclass
class AudioPreprocessorConfig:
    """Configuration parameters for audio standardization and Mel-spectrogram extraction."""
    target_sample_rate: int = 16000
    target_duration_seconds: float = 4.0
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    f_min: float = 0.0
    f_max: float = 8000.0
    power: float = 2.0
    top_db: float = 80.0

    @property
    def target_samples(self) -> int:
        """Total required samples = sample_rate * duration."""
        return int(round(self.target_sample_rate * self.target_duration_seconds))


@dataclass
class AudioPreprocessorResult:
    """Consolidated result of audio preprocessing."""
    video_id: str
    input_audio: str
    output_directory: str
    original_sample_rate: int
    original_channels: int
    processed_sample_rate: int
    processed_channels: int
    original_duration: float
    processed_duration: float
    num_samples: int
    mel_parameters: Dict[str, Any]
    processing_status: str  # "success" | "error"
    waveform_path: str
    spectrogram_path: str
    metadata_path: str
    mel_shape: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        """Converts result object to metadata dictionary matching specification."""
        return {
            "video_id": self.video_id,
            "input_audio": Path(self.input_audio).name,
            "original_sample_rate": self.original_sample_rate,
            "original_channels": self.original_channels,
            "processed_sample_rate": self.processed_sample_rate,
            "processed_channels": self.processed_channels,
            "original_duration": round(float(self.original_duration), 2),
            "processed_duration": round(float(self.processed_duration), 2),
            "num_samples": self.num_samples,
            "mel_parameters": self.mel_parameters,
            "processing_status": self.processing_status
        }


class AudioPreprocessor:
    """
    Standardizes raw audio waveforms to 16 kHz mono 4.0s (64,000 samples)
    and computes numerical Mel-spectrograms (.npy) without altering forensic artifacts.
    """

    def __init__(self, config: Optional[AudioPreprocessorConfig] = None):
        self.config = config or AudioPreprocessorConfig()

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int, int]:
        """
        Loads an audio file and returns (waveform, sample_rate, num_channels).
        
        Returns:
            waveform: numpy array of shape (samples,) or (samples, channels).
            sample_rate: sampling frequency in Hz.
            num_channels: integer channel count.
        """
        resolved_path = Path(audio_path).resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {resolved_path}")

        try:
            waveform, sr = sf.read(str(resolved_path), dtype="float32")
        except Exception:
            # Fallback to librosa.load
            waveform, sr = librosa.load(str(resolved_path), sr=None, mono=False)
            if waveform.ndim > 1:
                waveform = waveform.T  # shape to (samples, channels)

        if waveform.size == 0:
            raise ValueError(f"Audio file contains no samples (empty): {resolved_path}")

        if waveform.ndim == 1:
            channels = 1
        else:
            channels = waveform.shape[1] if waveform.shape[0] >= waveform.shape[1] else waveform.shape[0]
            if waveform.shape[0] < waveform.shape[1]:
                waveform = waveform.T

        return waveform, sr, channels

    def convert_to_mono(self, waveform: np.ndarray) -> np.ndarray:
        """
        Converts multi-channel waveform to mono by averaging across channels.
        If already mono, returns 1D array unchanged.
        """
        if waveform.ndim == 1:
            return waveform.astype(np.float32)

        if waveform.ndim == 2:
            # Average across channel dimension
            mono_waveform = np.mean(waveform, axis=1)
            return mono_waveform.astype(np.float32)

        raise ValueError(f"Unsupported waveform dimensions: {waveform.shape}")

    def resample_waveform(self, waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resamples mono waveform to target sample rate using librosa/soxr.
        """
        if orig_sr == target_sr:
            return waveform

        resampled = librosa.resample(
            y=waveform,
            orig_sr=orig_sr,
            target_sr=target_sr,
            res_type="soxr_hq"
        )
        return resampled.astype(np.float32)

    def trim_or_pad_duration(self, waveform: np.ndarray, target_samples: int) -> np.ndarray:
        """
        Ensures waveform contains exactly target_samples (e.g. 64,000 samples for 4.0s @ 16kHz).
        - If len > target_samples: trims to target_samples.
        - If len < target_samples: zero-pads to target_samples.
        - If len == target_samples: unchanged.
        """
        current_samples = len(waveform)

        if current_samples == target_samples:
            return waveform

        if current_samples > target_samples:
            # Trim to exact duration
            return waveform[:target_samples]

        # Zero-pad to exact duration
        pad_length = target_samples - current_samples
        padded_waveform = np.pad(waveform, (0, pad_length), mode="constant", constant_values=0.0)
        return padded_waveform.astype(np.float32)

    def generate_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Generates numerical Mel-spectrogram in decibel (log-power) scale from 16kHz mono 4s waveform.

        Returns:
            mel_db: 2D numpy float32 array of shape (n_mels, time_frames).
        """
        if len(waveform) != self.config.target_samples:
            raise ValueError(
                f"Expected waveform of {self.config.target_samples} samples, got {len(waveform)}"
            )

        mel_power = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.config.target_sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels,
            fmin=self.config.f_min,
            fmax=self.config.f_max,
            power=self.config.power
        )

        # Convert power to decibels (log scale)
        mel_db = librosa.power_to_db(mel_power, ref=np.max, top_db=self.config.top_db)

        # Validate no NaN or Inf values
        if not np.isfinite(mel_db).all():
            mel_db = np.nan_to_num(mel_db, nan=-self.config.top_db, posinf=0.0, neginf=-self.config.top_db)

        return mel_db.astype(np.float32)

    def process(
        self,
        input_audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        video_id: str = "video_001",
        save_metadata: bool = True
    ) -> AudioPreprocessorResult:
        """
        Executes end-to-end audio preprocessing:
        Loads raw audio -> Mono -> 16 kHz -> 4.0s (64k samples) -> Saves WAV & Mel (.npy) -> Metadata.

        Args:
            input_audio_path: Path to extracted audio file (e.g., audio.wav).
            output_dir: Root output directory (e.g., processed_dataset/video_001/).
            video_id: Video identifier.
            save_metadata: Whether to save audio_metadata.json.

        Returns:
            AudioPreprocessorResult with verified properties.
        """
        resolved_input = str(Path(input_audio_path).resolve())
        out_root = Path(output_dir).resolve()

        # Output subdirectories
        out_audio_dir = out_root / "audio"
        out_spec_dir = out_root / "spectrogram"
        out_audio_dir.mkdir(parents=True, exist_ok=True)
        out_spec_dir.mkdir(parents=True, exist_ok=True)

        target_wav_file = out_audio_dir / "audio_16k_4s.wav"
        target_npy_file = out_spec_dir / "mel.npy"
        target_meta_file = out_root / "audio_metadata.json"

        # 1. Load Audio
        raw_waveform, orig_sr, orig_channels = self.load_audio(resolved_input)
        orig_duration = float(len(raw_waveform)) / float(orig_sr)

        # 2. Convert to Mono
        mono_waveform = self.convert_to_mono(raw_waveform)

        # 3. Resample to 16 kHz
        resampled_waveform = self.resample_waveform(
            mono_waveform,
            orig_sr=orig_sr,
            target_sr=self.config.target_sample_rate
        )

        # 4. Trim or Pad to Exactly 4.0 Seconds (64,000 samples)
        standardized_waveform = self.trim_or_pad_duration(
            resampled_waveform,
            target_samples=self.config.target_samples
        )

        # 5. Save Standardized 16 kHz, 4s Mono WAV
        # Using 16-bit uncompressed PCM format
        sf.write(
            str(target_wav_file),
            standardized_waveform,
            self.config.target_sample_rate,
            subtype="PCM_16"
        )

        # Verification of saved WAV
        final_info = sf.info(str(target_wav_file))
        assert final_info.samplerate == self.config.target_sample_rate, f"Sample rate mismatch: {final_info.samplerate}"
        assert final_info.channels == 1, f"Channels mismatch: {final_info.channels}"
        assert final_info.frames == self.config.target_samples, f"Frames mismatch: {final_info.frames}"

        # 6. Generate Mel-Spectrogram
        mel_spectrogram = self.generate_mel_spectrogram(standardized_waveform)

        # Save numerical Mel-spectrogram as .npy
        np.save(str(target_npy_file), mel_spectrogram)

        # 7. Metadata Generation
        mel_params = {
            "n_mels": self.config.n_mels,
            "n_fft": self.config.n_fft,
            "hop_length": self.config.hop_length,
            "win_length": self.config.win_length
        }

        result = AudioPreprocessorResult(
            video_id=video_id,
            input_audio=resolved_input,
            output_directory=str(out_root),
            original_sample_rate=orig_sr,
            original_channels=orig_channels,
            processed_sample_rate=self.config.target_sample_rate,
            processed_channels=1,
            original_duration=orig_duration,
            processed_duration=self.config.target_duration_seconds,
            num_samples=self.config.target_samples,
            mel_parameters=mel_params,
            processing_status="success",
            waveform_path=str(target_wav_file),
            spectrogram_path=str(target_npy_file),
            metadata_path=str(target_meta_file),
            mel_shape=mel_spectrogram.shape
        )

        if save_metadata:
            with open(target_meta_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)

        return result


def preprocess_audio_file(
    input_audio_path: Union[str, Path],
    output_dir: Union[str, Path],
    video_id: str = "video_001",
    config: Optional[AudioPreprocessorConfig] = None
) -> AudioPreprocessorResult:
    """
    Convenience function to preprocess an audio file to 16 kHz 4s WAV and Mel-spectrogram (.npy).
    """
    preprocessor = AudioPreprocessor(config=config)
    return preprocessor.process(
        input_audio_path=input_audio_path,
        output_dir=output_dir,
        video_id=video_id
    )
