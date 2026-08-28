"""
Comprehensive Test Suite for Video Face & Mouth Preprocessing Pipeline.
"""

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing import (
    FaceAligner,
    FaceDetection,
    FaceDetector,
    MouthExtractor,
    VideoFacePreprocessor,
    preprocess_frame_sequence,
)


def create_synthetic_face_image(
    width: int = 300,
    height: int = 300,
    eye_left: tuple = (100, 110),
    eye_right: tuple = (200, 110),
    nose: tuple = (150, 160),
    mouth_left: tuple = (110, 210),
    mouth_right: tuple = (190, 210)
) -> Image.Image:
    """Creates a synthetic face image with clear geometric facial landmarks."""
    img = Image.new("RGB", (width, height), color=(230, 200, 175))
    draw = ImageDraw.Draw(img)

    # Face contour
    draw.ellipse([40, 30, 260, 270], fill=(240, 210, 185), outline=(180, 140, 110), width=2)

    # Eyes
    for eye in [eye_left, eye_right]:
        draw.ellipse([eye[0] - 18, eye[1] - 10, eye[0] + 18, eye[1] + 10], fill=(255, 255, 255), outline=(50, 50, 50))
        draw.ellipse([eye[0] - 8, eye[1] - 8, eye[0] + 8, eye[1] + 8], fill=(40, 30, 20))

    # Nose
    draw.polygon([nose, (nose[0] - 12, nose[1] + 20), (nose[0] + 12, nose[1] + 20)], fill=(210, 170, 140))

    # Mouth
    draw.ellipse([mouth_left[0] - 5, mouth_left[1] - 10, mouth_right[0] + 5, mouth_right[1] + 15], fill=(190, 80, 80), outline=(130, 40, 40))

    return img


class TestFaceAndMouthPreprocessing(unittest.TestCase):
    """Test suite for Face Detection, Alignment, and Mouth ROI extraction."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent / "test_scratch_face"
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        # Create input/frames directory with 16 sampled PNG frames
        cls.input_frames_dir = cls.test_dir / "input_frames"
        cls.input_frames_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, 17):
            frame_img = create_synthetic_face_image(
                eye_left=(100, 110 + (i % 3)),
                eye_right=(200, 110 - (i % 2)),
                mouth_left=(110, 210 + (i % 2)),
                mouth_right=(190, 210 - (i % 2))
            )
            frame_img.save(cls.input_frames_dir / f"frame_{i:02d}.png", format="PNG")

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_face_aligner_math(self):
        """Test affine similarity transform and landmark warping."""
        aligner = FaceAligner(output_size=(224, 224))
        detection = FaceDetection(
            bbox=[40.0, 30.0, 260.0, 270.0],
            confidence=0.99,
            landmarks={
                "left_eye": [100.0, 110.0],
                "right_eye": [200.0, 110.0],
                "nose": [150.0, 160.0],
                "mouth_left": [110.0, 210.0],
                "mouth_right": [190.0, 210.0]
            }
        )

        test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        result = aligner.align(test_img, detection)

        self.assertTrue(result.success)
        self.assertEqual(result.aligned_face_image.shape, (224, 224, 3))
        self.assertIsNotNone(result.aligned_landmarks)

        # In canonical alignment: left eye is near 0.35*224 = 78.4, right eye near 0.65*224 = 145.6
        aligned_lm = result.aligned_landmarks
        self.assertAlmostEqual(aligned_lm["left_eye"][0], 78.4, delta=2.0)
        self.assertAlmostEqual(aligned_lm["right_eye"][0], 145.6, delta=2.0)
        # Both eyes horizontally level (same y coordinate)
        self.assertAlmostEqual(aligned_lm["left_eye"][1], aligned_lm["right_eye"][1], delta=1.0)

    def test_02_face_aligner_no_face(self):
        """Test face aligner when detection is None."""
        aligner = FaceAligner(output_size=(224, 224))
        test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        result = aligner.align(test_img, None)

        self.assertFalse(result.success)
        self.assertEqual(result.status, "failed_no_face")
        self.assertIsNone(result.aligned_face_image)

    def test_03_mouth_extractor_temporal_smoothing(self):
        """Test mouth ROI extraction with temporal coordinate smoothing."""
        extractor = MouthExtractor(mouth_roi_size=(96, 96))

        # 5 consecutive frames with slight landmark jitter
        aligned_faces = [np.ones((224, 224, 3), dtype=np.uint8) * 100 for _ in range(5)]
        landmarks_seq = [
            {
                "left_eye": [78.4, 85.1],
                "right_eye": [145.6, 85.1],
                "nose": [112.0, 125.0],
                "mouth_left": [88.0 + (i % 2) * 2, 160.0 + (i % 2)],
                "mouth_right": [136.0 - (i % 2) * 2, 160.0 - (i % 2)]
            }
            for i in range(5)
        ]

        results = extractor.extract_sequence(aligned_faces, landmarks_seq)

        self.assertEqual(len(results), 5)
        for res in results:
            self.assertTrue(res.success)
            self.assertEqual(res.mouth_image.shape, (96, 96, 3))
            self.assertIsNotNone(res.smoothed_coordinates)
            self.assertIsNotNone(res.raw_coordinates)

    def test_04_primary_face_selection_strategy(self):
        """Test primary face selection among multiple candidates."""
        detector = FaceDetector()
        face_small_high_conf = FaceDetection(
            bbox=[10.0, 10.0, 40.0, 40.0],  # area = 900
            confidence=0.99,
            landmarks={"left_eye": [0,0], "right_eye": [0,0], "nose": [0,0], "mouth_left": [0,0], "mouth_right": [0,0]}
        )
        face_large_good_conf = FaceDetection(
            bbox=[50.0, 50.0, 250.0, 250.0],  # area = 40000
            confidence=0.95,
            landmarks={"left_eye": [0,0], "right_eye": [0,0], "nose": [0,0], "mouth_left": [0,0], "mouth_right": [0,0]}
        )

        selected = detector.select_primary_face([face_small_high_conf, face_large_good_conf])
        # Large primary speaker face should be selected over tiny background face
        self.assertEqual(selected, face_large_good_conf)

    def test_05_end_to_end_dataset_preprocessing(self):
        """Test complete VideoFacePreprocessor on 16 sampled frames."""
        out_dir = self.test_dir / "processed_video_001"
        preprocessor = VideoFacePreprocessor(face_size=(224, 224), mouth_size=(96, 96))

        result = preprocessor.process_frames_directory(
            frames_dir=self.input_frames_dir,
            output_dir=out_dir,
            video_id="video_001",
            label="real"
        )

        # 1. Verify schema and basic metadata
        self.assertEqual(result.video_id, "video_001")
        self.assertEqual(result.label, "real")
        self.assertEqual(result.num_frames, 16)
        self.assertEqual(len(result.frame_indices), 16)
        self.assertEqual(len(result.face_detected), 16)
        self.assertEqual(len(result.face_bbox), 16)
        self.assertEqual(len(result.face_confidence), 16)
        self.assertEqual(len(result.facial_landmarks), 16)
        self.assertEqual(len(result.alignment_status), 16)
        self.assertEqual(len(result.mouth_roi_coordinates), 16)
        self.assertEqual(len(result.mouth_roi_status), 16)

        # 2. Verify Directory Structure
        self.assertTrue((out_dir / "frames").is_dir())
        self.assertTrue((out_dir / "aligned_faces").is_dir())
        self.assertTrue((out_dir / "mouth_rois").is_dir())
        self.assertTrue((out_dir / "landmarks").is_dir())
        self.assertTrue((out_dir / "metadata.json").is_file())
        self.assertTrue((out_dir / "landmarks" / "landmarks.json").is_file())

        # 3. Verify all 16 frames are copied and named properly
        saved_frames = sorted((out_dir / "frames").glob("*.png"))
        self.assertEqual(len(saved_frames), 16)

        # 4. Verify aligned_faces and mouth_rois are PNG
        for face_file in (out_dir / "aligned_faces").glob("*.png"):
            with Image.open(face_file) as img:
                self.assertEqual(img.format, "PNG")
                self.assertEqual(img.size, (224, 224))

        for mouth_file in (out_dir / "mouth_rois").glob("*.png"):
            with Image.open(mouth_file) as img:
                self.assertEqual(img.format, "PNG")
                self.assertEqual(img.size, (96, 96))

        # 5. Verify JSON structure
        with open(out_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta["video_id"], "video_001")
        self.assertEqual(meta["num_frames"], 16)
        self.assertIn("face_detected", meta)
        self.assertIn("facial_landmarks", meta)
        self.assertIn("mouth_roi_coordinates", meta)


if __name__ == "__main__":
    unittest.main()
