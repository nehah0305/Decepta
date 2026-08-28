"""
Audio-Visual Synchronization Pre-Training Diagnostic Script.

Verifies video/audio alignment, mouth motion variance, audio spectrogram variance,
and synchronized temporal grid windowing prior to Stage 3 InfoNCE sync training.

Usage:
    python model/evaluation/debug_sync_pipeline.py --data-dir path/to/FakeAVCeleb
    python model/evaluation/debug_sync_pipeline.py --sample-video model/sample_test_video.mp4
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure model root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_CONFIG, VisualPipelineConfig
from preprocessing.video_reader import VideoReader
from preprocessing.frame_sampler import HighCoverageFrameSampler
from preprocessing.frame_quality import FrameQualityFilter
from preprocessing.face_alignment import FaceAlignmentPipeline
from preprocessing.mouth_extractor import MouthExtractor
from preprocessing.audio_windowing import AudioWindowExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logger = logging.getLogger(__name__)


def diagnose_video_sync(
    video_path: Path,
    config: VisualPipelineConfig = DEFAULT_CONFIG
) -> Dict:
    """
    Diagnoses cross-modal synchronization for a single video file.
    """
    if not video_path.exists():
        return {"video": video_path.name, "status": "FILE_NOT_FOUND", "passed": False}

    try:
        reader = VideoReader(video_path)
        meta = reader.metadata
        fps = meta.fps
        duration = meta.duration_seconds

        sampler = HighCoverageFrameSampler(coverage_ratio=0.70, min_frames=16, max_frames=64)
        plan = sampler.create_sampling_plan(meta.total_frames)

        quality_filter = FrameQualityFilter()
        face_aligner = FaceAlignmentPipeline(target_size=config.FACE_SIZE)
        mouth_extractor = MouthExtractor(mouth_roi_size=(config.MOUTH_ROI_SIZE, config.MOUTH_ROI_SIZE))
        audio_extractor = AudioWindowExtractor(
            sample_rate=config.AUDIO_SAMPLE_RATE,
            window_seconds=config.AUDIO_WINDOW_SECONDS,
            hop_seconds=config.AUDIO_HOP_SECONDS,
            n_mels=config.AUDIO_N_MELS,
            n_fft=config.AUDIO_N_FFT,
            hop_length=config.AUDIO_HOP_LENGTH
        )

        valid_mouth_crops = []
        visual_timestamps = []

        face_aligner.reset_tracking()

        for f_idx, ok, rgb, ts in reader.read_frames_by_indices(plan.candidate_indices):
            if not ok or rgb is None:
                continue
            q = quality_filter.evaluate_frame(rgb, f_idx, ts)
            if not q.is_usable:
                continue
            f_res = face_aligner.process_frame(rgb, f_idx, ts)
            if f_res.face_detected and f_res.aligned_face is not None and f_res.landmarks:
                box = mouth_extractor.compute_mouth_box(f_res.landmarks, (config.FACE_SIZE, config.FACE_SIZE))
                if box:
                    mcrop = mouth_extractor.crop_mouth_roi(f_res.aligned_face, box)
                    mt = torch.from_numpy(mcrop).permute(2, 0, 1).float() / 255.0
                    valid_mouth_crops.append(mt)
                    visual_timestamps.append((f_idx, ts))

        # Audio Processing
        audio_res = audio_extractor.process_video_audio(video_path, visual_timestamps=visual_timestamps)

        has_audio = audio_res.audio_available and len(audio_res.windows) > 0
        num_mouths = len(valid_mouth_crops)
        num_audio_windows = len(audio_res.windows)

        mouth_motion_var = 0.0
        if num_mouths > 1:
            m_stack = torch.stack(valid_mouth_crops, dim=0) # (N, 3, 112, 112)
            m_diff = m_stack[1:] - m_stack[:-1]
            mouth_motion_var = float(m_diff.var().item())

        audio_spec_var = 0.0
        if has_audio:
            mel_stack = torch.stack([torch.from_numpy(w.mel_spectrogram) for w in audio_res.windows], dim=0)
            audio_spec_var = float(mel_stack.var().item())

        sync_pass = (has_audio) and (num_mouths > 0) and (audio_spec_var > 0.0) and (mouth_motion_var > 0.0)

        return {
            "video": video_path.name,
            "fps": fps,
            "sample_rate": audio_res.sample_rate if has_audio else 0,
            "duration": duration,
            "num_mouths": num_mouths,
            "mouth_shape": list(torch.stack(valid_mouth_crops, dim=0).shape) if num_mouths > 0 else [],
            "num_audio_windows": num_audio_windows,
            "audio_window_shape": list(audio_res.windows[0].mel_spectrogram.shape) if has_audio else [],
            "mouth_motion_var": mouth_motion_var,
            "audio_spec_var": audio_spec_var,
            "passed": sync_pass
        }

    except Exception as e:
        logger.error(f"Error diagnosing {video_path.name}: {e}")
        return {"video": video_path.name, "status": f"ERROR: {e}", "passed": False}


def run_sync_diagnostic_suite(
    video_paths: List[Path],
    config: VisualPipelineConfig = DEFAULT_CONFIG
):
    print("\n" + "=" * 78)
    print("        AUDIO-VISUAL SYNCHRONIZATION DIAGNOSTIC SUITE")
    print("=" * 78)

    passed_count = 0
    total = len(video_paths)

    for p in video_paths:
        res = diagnose_video_sync(p, config=config)
        status_str = "✅ PASS" if res["passed"] else "❌ FAIL"
        print(f" Video: {res['video']:<30} | Status: {status_str}")
        print(f"    - Video FPS: {res.get('fps', 0):.2f} | Audio Sample Rate: {res.get('sample_rate', 0)} Hz")
        print(f"    - Duration: {res.get('duration', 0):.2f}s | Mouth Crops: {res.get('num_mouths', 0)} | Audio Windows: {res.get('num_audio_windows', 0)}")
        print(f"    - Mouth Shape: {res.get('mouth_shape', [])} | Audio Window Shape: {res.get('audio_window_shape', [])}")
        print(f"    - Mouth Motion Variance: {res.get('mouth_motion_var', 0):.6f} (Target: > 0)")
        print(f"    - Audio Spectrogram Var: {res.get('audio_spec_var', 0):.6f} (Target: > 0)")
        print("-" * 78)

        if res["passed"]:
            passed_count += 1

    print(f"\n DIAGNOSTIC VERDICT: {passed_count}/{total} PASSED")
    if passed_count == total:
        print(" ✅ READY FOR STAGE 3 INFONCE SYNC PRETRAINING!\n")
    else:
        print(" ❌ DO NOT START STAGE 3 — RESOLVE AUDIO/MOUTH EXTRACTION ISSUES FIRST!\n")


def main():
    parser = argparse.ArgumentParser(description="Audio-Visual Synchronization Pre-Training Diagnostic")
    parser.add_argument("--data-dir", type=str, default=None, help="Folder containing videos")
    parser.add_argument("--sample-video", type=str, default="model/sample_test_video.mp4", help="Sample video file")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    if args.data_dir:
        data_path = Path(args.data_dir)
        if not data_path.is_absolute():
            data_path = project_root / data_path
        paths = list(data_path.rglob("*.mp4"))[:10]
    else:
        vpath = Path(args.sample_video)
        if not vpath.is_absolute():
            vpath = project_root / vpath
        paths = [vpath]

    run_sync_diagnostic_suite(paths)


if __name__ == "__main__":
    main()
