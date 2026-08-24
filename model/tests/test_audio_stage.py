"""
Unit and Integration Test Suite for the Audio Preprocessing Stage.
"""

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing import (
    AudioPreprocessor,
    AudioPreprocessorConfig,
    preprocess_audio_file,
)


class TestAudioPreprocessingStage(unittest.TestCase):
    """Test suite for Audio Loading, Mono Conversion, 16 kHz Resampling, 4s Standardization, and Mel-Spectrogram."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent / "test_scratch_audio"
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create a 6.0-second 48 kHz stereo audio file (> 4s)
        cls.audio_long_stereo = cls.test_dir / "long_stereo_48k.wav"
        t_long = np.linspace(0, 6.0, int(48000 * 6.0), endpoint=False, dtype=np.float32)
        sig1 = 0.5 * np.sin(2 * np.pi * 440.0 * t_long)
        sig2 = 0.5 * np.sin(2 * np.pi * 880.0 * t_long)
        stereo_data = np.stack([sig1, sig2], axis=1)
        sf.write(str(cls.audio_long_stereo), stereo_data, 48000, subtype="PCM_16")

        # 2. Create a 2.0-second 44.1 kHz mono audio file (< 4s)
        cls.audio_short_mono = cls.test_dir / "short_mono_44k.wav"
        t_short = np.linspace(0, 2.0, int(44100 * 2.0), endpoint=False, dtype=np.float32)
        mono_data = 0.6 * np.sin(2 * np.pi * 500.0 * t_short)
        sf.write(str(cls.audio_short_mono), mono_data, 44100, subtype="PCM_16")

        # 3. Create an empty file
        cls.empty_audio = cls.test_dir / "empty.wav"
        with open(cls.empty_audio, "wb") as f:
            pass

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_load_audio_stereo_and_mono(self):
        """Test audio loading across sample rates and channel counts."""
        preprocessor = AudioPreprocessor()
        
        # Stereo 48k
        wave, sr, ch = preprocessor.load_audio(self.audio_long_stereo)
        self.assertEqual(sr, 48000)
        self.assertEqual(ch, 2)
        self.assertEqual(wave.shape[0], int(48000 * 6.0))

        # Mono 44.1k
        wave_m, sr_m, ch_m = preprocessor.load_audio(self.audio_short_mono)
        self.assertEqual(sr_m, 44100)
        self.assertEqual(ch_m, 1)
        self.assertEqual(wave_m.shape[0], int(44100 * 2.0))

    def test_02_convert_to_mono(self):
        """Test conversion of stereo signal to mono without distortion."""
        preprocessor = AudioPreprocessor()
        stereo_arr = np.ones((1000, 2), dtype=np.float32)
        stereo_arr[:, 0] = 0.4
        stereo_arr[:, 1] = 0.6

        mono_arr = preprocessor.convert_to_mono(stereo_arr)
        self.assertEqual(mono_arr.ndim, 1)
        self.assertEqual(len(mono_arr), 1000)
        self.assertAlmostEqual(float(mono_arr[0]), 0.5, places=5)

    def test_03_resample_waveform_16k(self):
        """Test resampling from 48kHz to exactly 16kHz."""
        preprocessor = AudioPreprocessor()
        t = np.linspace(0, 1.0, 48000, endpoint=False, dtype=np.float32)
        sig = np.sin(2 * np.pi * 440 * t)

        resampled = preprocessor.resample_waveform(sig, orig_sr=48000, target_sr=16000)
        self.assertEqual(len(resampled), 16000)
        self.assertTrue(np.isfinite(resampled).all())

    def test_04_trim_and_pad_duration(self):
        """Test 4-second (64,000 samples) trimming and zero-padding."""
        preprocessor = AudioPreprocessor()

        # Case A: 80,000 samples (> 64,000) -> Trim
        sig_long = np.ones(80000, dtype=np.float32)
        trimmed = preprocessor.trim_or_pad_duration(sig_long, target_samples=64000)
        self.assertEqual(len(trimmed), 64000)

        # Case B: 30,000 samples (< 64,000) -> Zero Pad
        sig_short = np.ones(30000, dtype=np.float32)
        padded = preprocessor.trim_or_pad_duration(sig_short, target_samples=64000)
        self.assertEqual(len(padded), 64000)
        # Verify first 30,000 are ones, remaining 34,000 are zeroes
        self.assertEqual(padded[0], 1.0)
        self.assertEqual(padded[29999], 1.0)
        self.assertEqual(padded[30000], 0.0)
        self.assertEqual(padded[63999], 0.0)

    def test_05_mel_spectrogram_dimensions_and_validity(self):
        """Test Mel-spectrogram shape consistency and finite values."""
        preprocessor = AudioPreprocessor()
        sig_64k = np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, 64000, endpoint=False, dtype=np.float32))

        mel = preprocessor.generate_mel_spectrogram(sig_64k)

        # Default: n_mels=128, hop_length=256 -> time_frames = 64000 // 256 + 1 = 251
        self.assertEqual(mel.shape, (128, 251))
        self.assertTrue(np.isfinite(mel).all(), "Mel-spectrogram contains NaN or Inf values")
        self.assertFalse(np.isnan(mel).any())

    def test_06_end_to_end_long_stereo_audio(self):
        """Test full pipeline on long stereo audio (>4s): output folders, WAV header, npy matrix, metadata."""
        out_dir = self.test_dir / "output_long"
        preprocessor = AudioPreprocessor()

        result = preprocessor.process(
            input_audio_path=self.audio_long_stereo,
            output_dir=out_dir,
            video_id="video_001"
        )

        # 1. Verify Result Object
        self.assertEqual(result.video_id, "video_001")
        self.assertEqual(result.original_sample_rate, 48000)
        self.assertEqual(result.original_channels, 2)
        self.assertEqual(result.processed_sample_rate, 16000)
        self.assertEqual(result.processed_channels, 1)
        self.assertEqual(result.processed_duration, 4.0)
        self.assertEqual(result.num_samples, 64000)
        self.assertEqual(result.mel_shape, (128, 251))

        # 2. Verify Output Folder Structure
        self.assertTrue((out_dir / "audio").is_dir())
        self.assertTrue((out_dir / "spectrogram").is_dir())
        self.assertTrue((out_dir / "audio" / "audio_16k_4s.wav").is_file())
        self.assertTrue((out_dir / "spectrogram" / "mel.npy").is_file())
        self.assertTrue((out_dir / "audio_metadata.json").is_file())

        # 3. Verify Saved WAV Properties
        info = sf.info(str(out_dir / "audio" / "audio_16k_4s.wav"))
        self.assertEqual(info.samplerate, 16000)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.frames, 64000)
        self.assertEqual(info.duration, 4.0)

        # 4. Verify Saved .npy Mel Matrix
        loaded_mel = np.load(str(out_dir / "spectrogram" / "mel.npy"))
        self.assertEqual(loaded_mel.shape, (128, 251))
        self.assertTrue(np.isfinite(loaded_mel).all())

        # 5. Verify Metadata JSON Content
        with open(out_dir / "audio_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta["video_id"], "video_001")
        self.assertEqual(meta["input_audio"], "long_stereo_48k.wav")
        self.assertEqual(meta["original_sample_rate"], 48000)
        self.assertEqual(meta["original_channels"], 2)
        self.assertEqual(meta["processed_sample_rate"], 16000)
        self.assertEqual(meta["processed_channels"], 1)
        self.assertEqual(meta["processed_duration"], 4.0)
        self.assertEqual(meta["num_samples"], 64000)
        self.assertEqual(meta["mel_parameters"]["n_mels"], 128)
        self.assertEqual(meta["processing_status"], "success")

    def test_07_end_to_end_short_mono_audio(self):
        """Test full pipeline on short mono audio (<4s) with zero padding."""
        out_dir = self.test_dir / "output_short"
        preprocessor = AudioPreprocessor()

        result = preprocessor.process(
            input_audio_path=self.audio_short_mono,
            output_dir=out_dir,
            video_id="video_002"
        )

        self.assertEqual(result.num_samples, 64000)
        self.assertEqual(result.processed_sample_rate, 16000)
        self.assertEqual(result.processed_channels, 1)

        # Verify saved WAV
        info = sf.info(str(out_dir / "audio" / "audio_16k_4s.wav"))
        self.assertEqual(info.frames, 64000)
        self.assertEqual(info.samplerate, 16000)

    def test_08_error_handling(self):
        """Test handling of missing or empty audio file."""
        preprocessor = AudioPreprocessor()
        with self.assertRaises(FileNotFoundError):
            preprocessor.process("non_existent_audio_xyz.wav", self.test_dir / "err_out")

        with self.assertRaises(Exception):
            preprocessor.process(self.empty_audio, self.test_dir / "empty_out")


if __name__ == "__main__":
    unittest.main()
