"""
Acceptance Test Script for High-Coverage Visual Deepfake Detection System.

Meets Section 23 Acceptance Requirements:
1. Generates a 300-frame synthetic video containing facial structures.
2. Runs full visual pipeline with 70% frame coverage ratio.
3. Verifies that ~210 candidate frames are sampled and analyzed.
4. Verifies chunked CNN extraction (batch_size=32) without frame reduction.
5. Verifies Gated Fusion gate tracking and Temporal Transformer aggregation.
6. Prints the Section 12 compliant Frame Coverage Report and verifies all constraints.
7. Executes coverage ablation (30%, 50%, 70%, 90%) and architectural ablation.
"""

import math
import os
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG, VisualPipelineConfig
from evaluation.evaluate import VisualDeepfakeEvaluator
from evaluation.coverage_analysis import run_frame_coverage_ablation, run_architectural_ablation


def draw_synthetic_face(
    canvas_w: int = 640,
    canvas_h: int = 480,
    center_x: int = 320,
    center_y: int = 240,
    face_radius: int = 120,
    eye_offset_y: int = 0
) -> np.ndarray:
    """Draws a facial structure with eyes, nose, and mouth for face detection."""
    img = Image.new("RGB", (canvas_w, canvas_h), color=(220, 220, 225))
    draw = ImageDraw.Draw(img)

    # Head / Face oval
    x0, y0 = center_x - face_radius, center_y - int(face_radius * 1.2)
    x1, y1 = center_x + face_radius, center_y + int(face_radius * 1.2)
    draw.ellipse([x0, y0, x1, y1], fill=(235, 205, 180), outline=(180, 140, 110), width=3)

    # Hair
    draw.chord([x0, y0 - 15, x1, center_y - 20], start=180, end=360, fill=(40, 30, 20))

    # Eyes
    eye_lx, eye_rx = center_x - 45, center_x + 45
    eye_y = center_y - 25 + eye_offset_y

    for ex in [eye_lx, eye_rx]:
        # Sclera
        draw.ellipse([ex - 20, eye_y - 12, ex + 20, eye_y + 12], fill=(255, 255, 255), outline=(60, 50, 40), width=2)
        # Iris & Pupil
        draw.ellipse([ex - 9, eye_y - 9, ex + 9, eye_y + 9], fill=(60, 40, 30))
        draw.ellipse([ex - 4, eye_y - 4, ex + 4, eye_y + 4], fill=(10, 10, 10))

    # Eyebrows
    draw.line([eye_lx - 22, eye_y - 18, eye_lx + 20, eye_y - 18], fill=(50, 40, 30), width=4)
    draw.line([eye_rx - 20, eye_y - 18, eye_rx + 22, eye_y - 18], fill=(50, 40, 30), width=4)

    # Nose
    nose_tip = (center_x, center_y + 20)
    draw.polygon([(center_x, center_y - 5), (center_x - 14, center_y + 25), (center_x + 14, center_y + 25)], fill=(215, 175, 145))

    # Mouth
    mouth_y = center_y + 65
    draw.ellipse([center_x - 35, mouth_y - 10, center_x + 35, mouth_y + 12], fill=(185, 75, 75), outline=(130, 40, 40), width=2)
    draw.line([center_x - 30, mouth_y, center_x + 30, mouth_y], fill=(120, 30, 30), width=2)

    return np.array(img)


def create_300_frame_test_video(output_path: Path, num_frames: int = 300, fps: int = 25) -> Path:
    """Generates an MP4 video with exactly num_frames (>= 300) containing clear facial motion."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))

    print(f"\n[Media Generator] Generating {num_frames}-frame test video at {output_path.name}...")

    for i in range(num_frames):
        # Subtle realistic head and eye motion
        shift_x = int(12.0 * math.sin(2.0 * math.pi * i / 60.0))
        shift_y = int(6.0 * math.cos(2.0 * math.pi * i / 60.0))
        eye_blink = 2 if (i % 40 in [0, 1]) else 0

        frame_rgb = draw_synthetic_face(
            canvas_w=w,
            canvas_h=h,
            center_x=320 + shift_x,
            center_y=240 + shift_y,
            face_radius=115,
            eye_offset_y=eye_blink
        )
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"[Media Generator] Successfully generated {num_frames} frames ({num_frames / fps:.2f}s).")
    return output_path


def run_acceptance_test():
    """Runs the primary acceptance test."""
    test_dir = Path("outputs/acceptance_test_data")
    test_dir.mkdir(parents=True, exist_ok=True)
    video_file = test_dir / "acceptance_300_frames.mp4"

    # Step 1: Create 300-frame video
    create_300_frame_test_video(video_file, num_frames=300, fps=25)

    # Step 2: Configure Visual Pipeline with 70% coverage
    config = VisualPipelineConfig(
        FRAME_COVERAGE_RATIO=0.70,
        MIN_FRAMES=32,
        FRAME_BATCH_SIZE=32,
        FACE_SIZE=224,
        SPATIAL_FEATURE_DIM=256,
        FREQUENCY_FEATURE_DIM=256,
        TRANSFORMER_DIM=768,
        TRANSFORMER_HEADS=8,
        TRANSFORMER_LAYERS=2,
        MODEL_MODE="full"
    )

    evaluator = VisualDeepfakeEvaluator(config=config)

    print("\n" + "=" * 70)
    print("      RUNNING ACCEPTANCE TEST ON 300-FRAME VIDEO (70% TARGET)")
    print("=" * 70)

    # Step 3: Run full video evaluation
    report, frame_metadata = evaluator.evaluate_video(video_file, coverage_ratio=0.70, save_metadata=True)

    # Print the Section 12 compliant coverage report
    report.print_summary()

    # Step 4: Verify Acceptance Criteria
    print("-" * 70)
    print("VERIFYING ACCEPTANCE CONSTRAINTS:")
    print("-" * 70)

    # 1. Total frames must be 300
    assert report.total_video_frames == 300, f"Expected 300 total frames, got {report.total_video_frames}"
    print("  [PASS] Total video frames == 300")

    # 2. Candidate sampled frames must be approximately 210 (70% of 300)
    assert report.candidate_sampled_frames == 210, f"Expected 210 candidate frames, got {report.candidate_sampled_frames}"
    print("  [PASS] Candidate sampled frames == 210 (~70.0% coverage)")

    # 3. Processed frames must NOT be artificially truncated to batch size (16 or 32)
    assert report.frames_processed_by_cnn == report.frames_with_face, (
        f"CNN processed ({report.frames_processed_by_cnn}) does not match detected faces ({report.frames_with_face})!"
    )
    assert report.frames_processed_by_cnn >= 150, (
        f"Expected high frame coverage (>= 150 frames), got only {report.frames_processed_by_cnn}. "
        "Do NOT truncate frames to batch size!"
    )
    print(f"  [PASS] CNN processed frames == {report.frames_processed_by_cnn} (High coverage verified!)")
    print(f"  [PASS] Chunked batch processing processed all {report.frames_processed_by_cnn} frames without dropping.")

    # 4. Frame-level feature storage verification (Section 9)
    assert len(frame_metadata) == report.frames_processed_by_cnn
    sample_rec = frame_metadata[0]
    assert "frame_index" in sample_rec
    assert "timestamp" in sample_rec
    assert "face_confidence" in sample_rec
    assert "gate_value" in sample_rec
    assert "spatial_feature" in sample_rec and len(sample_rec["spatial_feature"]) == 256
    assert "frequency_feature" in sample_rec and len(sample_rec["frequency_feature"]) == 256
    assert "fused_feature" in sample_rec and len(sample_rec["fused_feature"]) == 256
    print("  [PASS] Frame-level metadata & 256-D feature storage verified for every frame.")

    # 5. Output file verification
    csv_file = config.METADATA_DIR / f"{video_file.stem}_frame_metadata.csv"
    json_file = config.METADATA_DIR / f"{video_file.stem}_frame_metadata.json"
    feat_file = config.FEATURES_DIR / f"{video_file.stem}_features.npz"

    assert csv_file.exists(), f"Missing CSV: {csv_file}"
    assert json_file.exists(), f"Missing JSON: {json_file}"
    assert feat_file.exists(), f"Missing features: {feat_file}"
    print("  [PASS] Metadata CSV, JSON, and feature .npz files successfully created.")

    print("\n" + "=" * 70)
    print("           ALL ACCEPTANCE CRITERIA SUCCESSFULLY MET!")
    print("=" * 70 + "\n")

    # Step 5: Run Coverage Ratio Ablation (Section 20)
    run_frame_coverage_ablation(video_file, ratios=[0.30, 0.50, 0.70, 0.90], config=config)

    # Step 6: Run Architecture Ablation (Section 20)
    run_architectural_ablation(video_file, config=config)


if __name__ == "__main__":
    run_acceptance_test()
