"""
Dataset-Independent Audio Integrity Audit Script.

Audits a dataset directory or CSV manifest for audio track presence,
waveform statistics, and Mel spectrogram quality to prevent silent-zero-tensor training.

Usage:
    python model/evaluation/debug_audio_dataset.py --data-dir Datasets/raw/faceforensicspp
    python model/evaluation/debug_audio_dataset.py --data-dir path/to/FakeAVCeleb
    python model/evaluation/debug_audio_dataset.py --manifest Datasets/metadata/train.csv
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure model root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_CONFIG, VisualPipelineConfig
from preprocessing.audio_windowing import AudioWindowExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def audit_dataset_audio(
    video_paths: List[Path],
    dataset_name: str = "Dataset",
    sample_limit: Optional[int] = 200,
    config: VisualPipelineConfig = DEFAULT_CONFIG
) -> Dict:
    """
    Audits video files for audio stream presence, waveform statistics, and Mel spectrogram quality.
    """
    logger.info(f"--- STARTING AUDIO INTEGRITY AUDIT ON: {dataset_name} ---")
    logger.info(f"Total Video Paths Discovered: {len(video_paths)}")

    if sample_limit and sample_limit < len(video_paths):
        # Sample uniformly across dataset
        indices = np.linspace(0, len(video_paths) - 1, sample_limit, dtype=int)
        sampled_paths = [video_paths[i] for i in indices]
    else:
        sampled_paths = video_paths

    logger.info(f"Auditing Subset of {len(sampled_paths)} Videos...")

    extractor = AudioWindowExtractor(
        sample_rate=config.AUDIO_SAMPLE_RATE,
        window_seconds=config.AUDIO_WINDOW_SECONDS,
        hop_seconds=config.AUDIO_HOP_SECONDS,
        n_mels=config.AUDIO_N_MELS,
        n_fft=config.AUDIO_N_FFT,
        hop_length=config.AUDIO_HOP_LENGTH
    )

    videos_with_audio = 0
    videos_without_audio = 0
    missing_files = 0

    durations = []
    sample_rates = []
    channels_list = []

    wf_means = []
    wf_stds = []
    wf_mins = []
    wf_maxs = []

    mel_means = []
    mel_stds = []
    mel_mins = []
    mel_maxs = []

    total_mel_windows = 0
    zero_spectrogram_windows = 0

    start_time = time.time()

    for idx, p in enumerate(sampled_paths):
        if not p.exists():
            missing_files += 1
            continue

        try:
            res = extractor.process_video_audio(p)
            if res.audio_available and len(res.windows) > 0:
                videos_with_audio += 1
                sample_rates.append(res.sample_rate)

                # Estimate duration from windows or samples
                total_samples = len(res.audio_waveform)
                dur = total_samples / float(res.sample_rate) if res.sample_rate > 0 else 0.0
                durations.append(dur)

                # Waveform stats
                wf = res.audio_waveform
                wf_means.append(float(wf.mean()))
                wf_stds.append(float(wf.std()))
                wf_mins.append(float(wf.min()))
                wf_maxs.append(float(wf.max()))

                # Mel window stats
                for w in res.windows:
                    total_mel_windows += 1
                    m = w.mel_spectrogram
                    mel_means.append(float(m.mean()))
                    mel_stds.append(float(m.std()))
                    mel_mins.append(float(m.min()))
                    mel_maxs.append(float(m.max()))
                    if np.all(m == 0):
                        zero_spectrogram_windows += 1
            else:
                videos_without_audio += 1
        except Exception as e:
            videos_without_audio += 1

    elapsed = time.time() - start_time
    valid_audited = len(sampled_paths) - missing_files
    audio_coverage_pct = (videos_with_audio / valid_audited * 100.0) if valid_audited > 0 else 0.0
    zero_window_pct = (zero_spectrogram_windows / total_mel_windows * 100.0) if total_mel_windows > 0 else 100.0

    avg_duration = float(np.mean(durations)) if durations else 0.0
    avg_sr = float(np.mean(sample_rates)) if sample_rates else 0.0

    avg_wf_mean = float(np.mean(wf_means)) if wf_means else 0.0
    avg_wf_std = float(np.mean(wf_stds)) if wf_stds else 0.0
    avg_wf_min = float(np.mean(wf_mins)) if wf_mins else 0.0
    avg_wf_max = float(np.mean(wf_maxs)) if wf_maxs else 0.0

    avg_mel_mean = float(np.mean(mel_means)) if mel_means else 0.0
    avg_mel_std = float(np.mean(mel_stds)) if mel_stds else 0.0
    avg_mel_min = float(np.mean(mel_mins)) if mel_mins else 0.0
    avg_mel_max = float(np.mean(mel_maxs)) if mel_maxs else 0.0

    audit_pass = (audio_coverage_pct >= 95.0) and (zero_window_pct < 5.0) and (avg_wf_std > 0.0)

    # Print Formatted Report
    line = "=" * 78
    print("\n" + line)
    print(f"      AUDIO INTEGRITY AUDIT REPORT — {dataset_name.upper()}")
    print(line)
    print(f" Total Videos Discovered:       {len(video_paths)}")
    print(f" Videos Audited in Sample:      {len(sampled_paths)}")
    print(f" Missing Files:                 {missing_files}")
    print(f" Videos WITH Valid Audio:       {videos_with_audio} ({audio_coverage_pct:.1f}%)")
    print(f" Videos WITHOUT Audio Track:    {videos_without_audio} ({100.0 - audio_coverage_pct:.1f}%)")
    print(f" Audio Coverage Percentage:     {audio_coverage_pct:.2f}%  (Target: > 95%)")
    print("-" * 78)
    print(" AUDIO PROPERTIES:")
    print(f"    - Average Video Duration:    {avg_duration:.2f} seconds")
    print(f"    - Sample Rate:              {avg_sr:.0f} Hz")
    print(f"    - Audio Channels:           1 (Mono resampled)")
    print("-" * 78)
    print(" WAVEFORM STATISTICS:")
    print(f"    - Waveform Min:             {avg_wf_min:.6f}")
    print(f"    - Waveform Max:             {avg_wf_max:.6f}")
    print(f"    - Waveform Mean:            {avg_wf_mean:.6f}")
    print(f"    - Waveform Std:             {avg_wf_std:.6f}  (Target: > 0.0)")
    print("-" * 78)
    print(" MEL SPECTROGRAM STATISTICS:")
    print(f"    - Total Windows Analyzed:   {total_mel_windows}")
    print(f"    - Zero-Value Windows:       {zero_spectrogram_windows} ({zero_window_pct:.2f}%)  (Target: ~0%)")
    print(f"    - Mel Spectrogram Min:      {avg_mel_min:.6f}")
    print(f"    - Mel Spectrogram Max:      {avg_mel_max:.6f}")
    print(f"    - Mel Spectrogram Mean:     {avg_mel_mean:.6f}")
    print(f"    - Mel Spectrogram Std:      {avg_mel_std:.6f}  (Target: > 0.0)")
    print("-" * 78)
    print(f" AUDIT VERDICT:                 {'✅ PASSED (READY FOR TRAINING)' if audit_pass else '❌ FAILED (DO NOT TRAIN AUDIO/SYNC)'}")
    print(line + "\n")

    return {
        "dataset_name": dataset_name,
        "total_videos": len(video_paths),
        "audited_videos": len(sampled_paths),
        "videos_with_audio": videos_with_audio,
        "videos_without_audio": videos_without_audio,
        "audio_coverage_pct": audio_coverage_pct,
        "average_duration_sec": avg_duration,
        "sample_rate": avg_sr,
        "waveform_stats": {
            "min": avg_wf_min,
            "max": avg_wf_max,
            "mean": avg_wf_mean,
            "std": avg_wf_std
        },
        "mel_stats": {
            "min": avg_mel_min,
            "max": avg_mel_max,
            "mean": avg_mel_mean,
            "std": avg_mel_std
        },
        "zero_window_pct": zero_window_pct,
        "audit_passed": audit_pass
    }


def main():
    parser = argparse.ArgumentParser(description="Dataset-Independent Audio Integrity Audit")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing .mp4/.avi/.mov video files")
    parser.add_argument("--manifest", type=str, default=None, help="Path to CSV manifest file")
    parser.add_argument("--sample-limit", type=int, default=200, help="Number of samples to audit (default: 200)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    video_paths = []
    dataset_name = "Custom Dataset"

    if args.data_dir:
        data_path = Path(args.data_dir)
        if not data_path.is_absolute():
            data_path = project_root / data_path
        dataset_name = data_path.name
        video_paths = sorted(list(data_path.rglob("*.mp4")) + list(data_path.rglob("*.avi")) + list(data_path.rglob("*.mov")))

    elif args.manifest:
        man_path = Path(args.manifest)
        if not man_path.is_absolute():
            man_path = project_root / man_path
        dataset_name = man_path.stem
        df = pd.read_csv(man_path)
        data_root = project_root / "Datasets/raw"
        for _, row in df.iterrows():
            rel_path = str(row["video_path"])
            full_path = data_root / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
            video_paths.append(full_path)

    else:
        # Default: audit FF++ raw folder
        data_path = project_root / "Datasets/raw/faceforensicspp"
        dataset_name = "FaceForensics++"
        video_paths = sorted(list(data_path.rglob("*.mp4")))

    if not video_paths:
        logger.error("No video files found to audit!")
        sys.exit(1)

    audit_dataset_audio(video_paths, dataset_name=dataset_name, sample_limit=args.sample_limit)


if __name__ == "__main__":
    main()
