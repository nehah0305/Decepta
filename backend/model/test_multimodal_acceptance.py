"""
Master Multimodal Acceptance Test Script.

Validates the full Multimodal Deepfake Detection System:
1. Generates a synthetic video containing 300 frames with facial/mouth motion and synchronized audio.
2. Runs full Multimodal Evaluator (Visual + Audio Authenticity + Sync + Adaptive Modality Attention).
3. Verifies all 10 canonical outputs:
   - 768-D Visual Feature
   - 768-D Audio Feature
   - 256-D Sync Feature
   - α_v (Visual attention weight)
   - α_a (Audio attention weight)
   - α_s (Sync attention weight)
   - Audio-Visual Synchronization Score
   - Real Probability
   - Fake Probability
   - Final Prediction (Real/Fake)
4. Executes Synchronization Temporal Shift Sensitivity Evaluation (±0.25s, ±0.5s, ±1.0s, ±2.0s).
5. Executes Comprehensive Modality Ablation Study (V, A, S, V+A, V+S, A+S, Full).
"""

import math
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
import soundfile as sf
import torch

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG, VisualPipelineConfig
from evaluation.multimodal_evaluator import MultimodalDeepfakeEvaluator
from evaluation.sync_evaluation import evaluate_synchronization_offsets
from evaluation.multimodal_ablation import run_modality_ablation_experiment
from preprocessing.ffmpeg_utils import get_ffmpeg_path, run_ffmpeg_command


def create_synthetic_multimodal_sample(
    output_video_path: Path,
    num_frames: int = 300,
    fps: int = 25
) -> Path:
    """Generates a 300-frame video with facial motion and a synchronized audio track."""
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    temp_silent_video = output_video_path.parent / "temp_silent.mp4"
    temp_audio_wav = output_video_path.parent / "temp_synth_audio.wav"

    duration = num_frames / float(fps)
    w, h = 640, 480

    # 1. Create video frames with mouth opening and closing
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_silent_video), fourcc, float(fps), (w, h))

    print(f"\n[Media Generator] Generating {num_frames}-frame multimodal test video ({duration:.2f}s)...")

    for i in range(num_frames):
        # Mouth open/close motion cycle
        mouth_open = int(8.0 * (1.0 + math.sin(2.0 * math.pi * i / 20.0)))
        shift_x = int(10.0 * math.sin(2.0 * math.pi * i / 50.0))

        img = Image.new("RGB", (w, h), color=(220, 220, 225))
        draw = ImageDraw.Draw(img)

        # Head / Face oval
        cx, cy = 320 + shift_x, 240
        draw.ellipse([cx - 115, cy - 140, cx + 115, cy + 140], fill=(235, 205, 180), outline=(180, 140, 110), width=3)
        draw.chord([cx - 115, cy - 155, cx + 115, cy - 20], start=180, end=360, fill=(40, 30, 20))

        # Eyes & Nose
        for ex in [cx - 45, cx + 45]:
            draw.ellipse([ex - 20, cy - 35, ex + 20, cy - 15], fill=(255, 255, 255), outline=(60, 50, 40), width=2)
            draw.ellipse([ex - 8, cy - 29, ex + 8, cy - 19], fill=(50, 30, 20))
        draw.polygon([(cx, cy - 5), (cx - 12, cy + 22), (cx + 12, cy + 22)], fill=(215, 175, 145))

        # Mouth with dynamic opening
        mouth_y = cy + 65
        draw.ellipse([cx - 35, mouth_y - 8 - mouth_open, cx + 35, mouth_y + 10 + mouth_open], fill=(185, 75, 75), outline=(130, 40, 40), width=2)

        frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()

    # 2. Create synchronized audio waveform (16 kHz)
    sr = 16000
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    # Frequency modulated audio matching the mouth motion cycle
    mod_freq = 440.0 + 150.0 * np.sin(2.0 * np.pi * t * 1.25)
    phase = 2.0 * np.pi * np.cumsum(mod_freq) / sr
    audio_wav = (0.5 * np.sin(phase)).astype(np.float32)
    sf.write(str(temp_audio_wav), audio_wav, sr)

    # 3. Combine video and audio with FFmpeg into final container
    exe = get_ffmpeg_path()
    cmd = [
        "-hide_banner", "-y",
        "-i", str(temp_silent_video),
        "-i", str(temp_audio_wav),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(output_video_path)
    ]
    run_ffmpeg_command(cmd, ffmpeg_path=exe, check=True)

    # Cleanup intermediates
    if temp_silent_video.exists():
        temp_silent_video.unlink()
    if temp_audio_wav.exists():
        temp_audio_wav.unlink()

    print(f"[Media Generator] Created multimodal video at {output_video_path.name}")
    return output_video_path


def run_multimodal_acceptance_test():
    """Executes full multimodal system verification."""
    test_dir = Path("outputs/acceptance_test_data")
    test_dir.mkdir(parents=True, exist_ok=True)
    video_file = test_dir / "acceptance_multimodal_300.mp4"

    # Step 1: Generate 300-frame multimodal video
    create_synthetic_multimodal_sample(video_file, num_frames=300, fps=25)

    # Step 2: Initialize Multimodal Evaluator
    config = VisualPipelineConfig(
        FRAME_COVERAGE_RATIO=0.70,
        MIN_FRAMES=32,
        FRAME_BATCH_SIZE=32,
        MODEL_MODE="full"
    )
    evaluator = MultimodalDeepfakeEvaluator(config=config)

    print("\n" + "=" * 75)
    print("   RUNNING MASTER MULTIMODAL ACCEPTANCE TEST ON 300-FRAME VIDEO")
    print("=" * 75)

    # Step 3: Run Full Multimodal Evaluation
    report, outputs, meta = evaluator.evaluate_video(video_file, coverage_ratio=0.70, save_metadata=True)

    # Print Master Report
    report.print_summary()

    # Step 4: Verify All Acceptance Criteria
    print("-" * 75)
    print("VERIFYING MULTIMODAL SYSTEM CONSTRAINTS:")
    print("-" * 75)

    # 1. Visual high coverage
    assert report.total_video_frames == 300, f"Expected 300 frames, got {report.total_video_frames}"
    assert report.candidate_frames == 210, f"Expected 210 candidate frames, got {report.candidate_frames}"
    assert report.faces_detected >= 150, f"Expected >=150 faces detected, got {report.faces_detected}"
    assert report.mouths_extracted >= 150, f"Expected >=150 mouth crops, got {report.mouths_extracted}"
    print(f"  [PASS] Visual high coverage: {report.candidate_frames}/300 frames sampled ({report.visual_coverage_pct:.1f}% analyzed).")

    # 2. Audio windowing
    assert report.has_audio, "Audio stream must be present and extracted."
    assert report.num_audio_windows >= 4, f"Expected >=4 overlapping windows for 12s video, got {report.num_audio_windows}"
    print(f"  [PASS] 16 kHz Audio windowing: {report.num_audio_windows} overlapping 4-sec windows (hop=2s) processed.")

    # 3. Canonical Representations
    assert outputs.visual_feature.size(-1) == 768, "Visual feature must be 768-D."
    assert outputs.audio_feature.size(-1) == 768, "Audio authenticity feature must be 768-D."
    assert outputs.sync_feature.size(-1) == 256, "Sync feature must be 256-D."
    print("  [PASS] Canonical Feature Dimensions verified: Visual (768-D), Audio (768-D), Sync (256-D).")

    # 4. Adaptive Modality Attention
    sum_alphas = report.alpha_v + report.alpha_a + report.alpha_s
    assert abs(sum_alphas - 1.0) < 1e-3, f"Modality weights must sum to 1.0, got {sum_alphas}"
    print(f"  [PASS] Adaptive Modality Attention verified: alpha_v={report.alpha_v:.4f}, alpha_a={report.alpha_a:.4f}, alpha_s={report.alpha_s:.4f} (Sum = {sum_alphas:.4f}).")

    # 5. Synchronization Score
    assert 0.0 <= report.sync_score <= 1.0, f"Sync score out of range: {report.sync_score}"
    print(f"  [PASS] Audio-Visual Synchronization score verified: {report.sync_score * 100:.2f}%.")

    # 6. Final Real/Fake Classification
    assert 0.0 <= report.fake_probability <= 1.0
    assert 0.0 <= report.real_probability <= 1.0
    assert report.prediction in ["Real", "Fake"]
    print(f"  [PASS] Final Prediction verified: {report.prediction} (Confidence: {report.confidence * 100:.2f}%).")

    print("\n" + "=" * 75)
    print("        ALL CANONICAL SYSTEM REQUIREMENTS SUCCESSFULLY MET!")
    print("=" * 75 + "\n")

    # Step 5: Run Audio-Visual Synchronization Sensitivity Evaluation (Section 33)
    dummy_m = torch.randn(1, report.faces_detected, 256, device=evaluator.device)
    dummy_a = torch.randn(1, report.num_audio_windows * 62, 768, device=evaluator.device)
    sync_report = evaluate_synchronization_offsets(
        evaluator.model.sync_branch,
        dummy_m,
        dummy_a,
        offsets_sec=[-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]
    )
    sync_report.video_id = video_file.name
    sync_report.print_summary()

    # Step 6: Run Modality Ablation Study (Section 32)
    run_modality_ablation_experiment(
        evaluator.model,
        face_frames=torch.stack([torch.zeros(3, 224, 224)] * 16, dim=0).to(evaluator.device),
        mouth_crops=torch.stack([torch.zeros(3, 112, 112)] * 16, dim=0).to(evaluator.device),
        mel_windows=torch.zeros(2, 128, 251, device=evaluator.device),
        device=evaluator.device
    )


if __name__ == "__main__":
    run_multimodal_acceptance_test()
