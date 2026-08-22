"""
Comprehensive Test Suite for Multimodal Deepfake Detection Preprocessing Pipeline.
"""

import json
import os
import shutil
import struct
import sys
import unittest
from pathlib import Path
from PIL import Image

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing import (
    AudioProcessor,
    MultimodalPreprocessor,
    VideoProcessor,
    get_ffmpeg_path,
    preprocess_video,
    probe_media,
)
from tests.generate_test_media import create_synthetic_multimodal_video


class TestMultimodalPreprocessing(unittest.TestCase):
    """Test suite for video and audio preprocessing pipeline."""

    @classmethod
    def setUpClass(cls):
        """Generate test videos with and without audio."""
        cls.test_dir = Path(__file__).parent / "test_scratch"
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        cls.video_with_audio = cls.test_dir / "test_with_audio.mp4"
        cls.video_no_audio = cls.test_dir / "test_no_audio.mp4"

        # Create a 2.0 second 30 FPS video with audio (60 original frames)
        create_synthetic_multimodal_video(
            cls.video_with_audio,
            duration=2.0,
            fps=30,
            width=320,
            height=240,
            include_audio=True,
            audio_sample_rate=44100
        )

        # Create a 1.5 second 20 FPS video without audio (30 original frames)
        create_synthetic_multimodal_video(
            cls.video_no_audio,
            duration=1.5,
            fps=20,
            width=320,
            height=240,
            include_audio=False
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up generated test media and temporary directories."""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_ffmpeg_binary_discovery(self):
        """Ensure FFmpeg executable is located successfully."""
        ffmpeg_exe = get_ffmpeg_path()
        self.assertTrue(os.path.isfile(ffmpeg_exe), f"FFmpeg binary not found at: {ffmpeg_exe}")

    def test_02_probe_media_with_audio(self):
        """Test probing media with both video and audio streams."""
        info = probe_media(self.video_with_audio)
        self.assertTrue(info.has_video)
        self.assertTrue(info.has_audio)
        self.assertIsNotNone(info.video_stream)
        self.assertIsNotNone(info.audio_stream)
        self.assertEqual(info.video_stream.width, 320)
        self.assertEqual(info.video_stream.height, 240)
        self.assertAlmostEqual(info.video_stream.fps, 30.0, delta=1.0)
        self.assertAlmostEqual(info.duration, 2.0, delta=0.5)
        self.assertEqual(info.audio_stream.sample_rate, 44100)

    def test_03_probe_media_without_audio(self):
        """Test probing media with video only."""
        info = probe_media(self.video_no_audio)
        self.assertTrue(info.has_video)
        self.assertFalse(info.has_audio)
        self.assertIsNotNone(info.video_stream)
        self.assertIsNone(info.audio_stream)

    def test_04_video_frame_extraction_png_and_naming(self):
        """Test video frame extraction: 5 FPS, exclusively PNG, sequential naming."""
        out_frames_dir = self.test_dir / "output_frames_5fps"
        processor = VideoProcessor(target_fps=5.0)
        result = processor.extract_frames(self.video_with_audio, out_frames_dir)

        self.assertEqual(result.extracted_fps, 5.0)
        self.assertGreater(result.total_extracted_frames, 0)
        # For a 2.0s video at 5 FPS, we expect around 10 frames
        self.assertIn(result.total_extracted_frames, [9, 10, 11])

        # Verify all files are named sequentially frame_000001.png, frame_000002.png, ...
        extracted_files = sorted(out_frames_dir.glob("*"))
        self.assertEqual(len(extracted_files), result.total_extracted_frames)

        for idx, file_path in enumerate(extracted_files, start=1):
            expected_name = f"frame_{idx:06d}.png"
            self.assertEqual(file_path.name, expected_name)
            self.assertTrue(file_path.name.endswith(".png"))

            # Verify actual image content is valid PNG and dimensions match
            with Image.open(file_path) as img:
                self.assertEqual(img.format, "PNG")
                self.assertEqual(img.size, (320, 240))

        # Verify timestamps in result
        self.assertEqual(result.frame_timestamps["frame_000001.png"], 0.0)
        self.assertAlmostEqual(result.frame_timestamps["frame_000002.png"], 0.2, places=2)

    def test_05_video_frame_extraction_10fps(self):
        """Test video frame extraction at 10 FPS."""
        out_frames_dir = self.test_dir / "output_frames_10fps"
        processor = VideoProcessor(target_fps=10.0)
        result = processor.extract_frames(self.video_with_audio, out_frames_dir)

        self.assertEqual(result.extracted_fps, 10.0)
        # For a 2.0s video at 10 FPS, expect around 20 frames
        self.assertIn(result.total_extracted_frames, [19, 20, 21])
        self.assertEqual(result.frame_filenames[0], "frame_000001.png")
        self.assertAlmostEqual(result.frame_timestamps["frame_000002.png"], 0.1, places=2)

    def test_06_audio_extraction_wav_format(self):
        """Test audio extraction: WAV format, uncompressed PCM 16-bit, temporal sync."""
        out_audio_file = self.test_dir / "output_audio" / "audio.wav"
        processor = AudioProcessor()
        result = processor.extract_audio(self.video_with_audio, out_audio_file)

        self.assertTrue(result.has_audio)
        self.assertIsNotNone(result.output_audio_path)
        self.assertTrue(out_audio_file.is_file())
        self.assertEqual(result.format, "wav")

        # Verify WAV header RIFF structure
        with open(out_audio_file, "rb") as f:
            riff = f.read(4)
            self.assertEqual(riff, b"RIFF")
            f.seek(8)
            wave = f.read(4)
            self.assertEqual(wave, b"WAVE")

    def test_07_audio_extraction_resampling(self):
        """Test audio extraction with custom target sample rate (e.g. 16000 Hz for speech models)."""
        out_audio_file = self.test_dir / "output_audio_16k" / "audio.wav"
        processor = AudioProcessor(sample_rate=16000, channels=1)
        result = processor.extract_audio(self.video_with_audio, out_audio_file)

        self.assertTrue(result.has_audio)
        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(result.channels, 1)

    def test_08_audio_extraction_video_without_audio(self):
        """Test audio extraction on video that contains no audio stream."""
        out_audio_file = self.test_dir / "output_audio_none" / "audio.wav"
        processor = AudioProcessor()
        result = processor.extract_audio(self.video_no_audio, out_audio_file, allow_no_audio=True)

        self.assertFalse(result.has_audio)
        self.assertIsNone(result.output_audio_path)

    def test_09_full_pipeline_end_to_end(self):
        """Test full MultimodalPreprocessor end-to-end processing with metadata.json generation."""
        out_dir = self.test_dir / "full_pipeline_out"
        preprocessor = MultimodalPreprocessor(extracted_fps=5.0)
        result = preprocessor.process(self.video_with_audio, output_dir=out_dir)

        self.assertEqual(result.status, "success")
        self.assertTrue(Path(result.frames_directory).is_dir())
        self.assertTrue(Path(result.audio_file_path).is_file())
        self.assertTrue(Path(result.metadata_file_path).is_file())

        # Read back saved JSON metadata
        with open(result.metadata_file_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Validate all required fields
        self.assertEqual(metadata["status"], "success")
        self.assertEqual(metadata["extracted_fps"], 5.0)
        self.assertEqual(metadata["resolution"]["width"], 320)
        self.assertEqual(metadata["resolution"]["height"], 240)
        self.assertGreater(metadata["total_extracted_frames"], 0)
        self.assertEqual(len(metadata["frame_filenames"]), metadata["total_extracted_frames"])
        self.assertEqual(len(metadata["frame_timestamps"]), metadata["total_extracted_frames"])
        self.assertTrue(metadata["has_audio"])
        self.assertEqual(metadata["audio_format"], "wav")

    def test_10_invalid_input_handling(self):
        """Test pipeline behavior with non-existent file."""
        preprocessor = MultimodalPreprocessor()
        with self.assertRaises(FileNotFoundError):
            preprocessor.process("non_existent_video_12345.mp4", self.test_dir / "invalid_out")


if __name__ == "__main__":
    unittest.main()
