"""
Coverage and Ablation Experiment Analysis Suite.

Evaluates and compares:
1. Impact of High Frame Coverage (30%, 50%, 70%, 90% coverage ratios)
2. Architecture Ablations:
   A. Spatial CNN only
   B. Frequency CNN only
   C. Spatial + Frequency without gated fusion
   D. Spatial + Frequency + Gated Fusion (Mean pooling without Transformer)
   E. Full Model + Temporal Transformer
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch

from config import DEFAULT_CONFIG, VisualPipelineConfig
from models.visual_model import VisualDeepfakeDetector
from evaluation.evaluate import VisualDeepfakeEvaluator
from utils.logging import FrameCoverageReport

logger = logging.getLogger(__name__)


@dataclass
class CoverageExperimentResult:
    """Summary record for a frame coverage ratio experiment."""
    coverage_ratio: float
    total_video_frames: int
    sampled_candidate_frames: int
    cnn_processed_frames: int
    effective_coverage_pct: float
    avg_gate_value: float
    prediction: str
    confidence: float
    inference_time_sec: float


def run_frame_coverage_ablation(
    video_path: Union[str, Path],
    ratios: List[float] = [0.30, 0.50, 0.70, 0.90],
    config: VisualPipelineConfig = DEFAULT_CONFIG
) -> List[CoverageExperimentResult]:
    """
    Evaluates the exact same video across multiple frame coverage levels.
    """
    evaluator = VisualDeepfakeEvaluator(config=config)
    results: List[CoverageExperimentResult] = []

    print("\n" + "=" * 75)
    print(f"      HIGH-COVERAGE FRAME SAMPLING ABLATION ANALYSIS: {Path(video_path).name}")
    print("=" * 75)

    for ratio in ratios:
        t0 = time.time()
        report, _ = evaluator.evaluate_video(
            video_path=video_path,
            coverage_ratio=ratio,
            save_metadata=False
        )
        elapsed = time.time() - t0

        res = CoverageExperimentResult(
            coverage_ratio=ratio,
            total_video_frames=report.total_video_frames,
            sampled_candidate_frames=report.candidate_sampled_frames,
            cnn_processed_frames=report.frames_processed_by_cnn,
            effective_coverage_pct=report.coverage_percentage,
            avg_gate_value=report.avg_gate_value,
            prediction=report.prediction,
            confidence=report.confidence,
            inference_time_sec=round(elapsed, 3)
        )
        results.append(res)

    # Print summary comparative table
    df = pd.DataFrame([
        {
            "Coverage Ratio Target": f"{r.coverage_ratio:.0%}",
            "Total Frames": r.total_video_frames,
            "Sampled Frames": r.sampled_candidate_frames,
            "CNN Processed": r.cnn_processed_frames,
            "Effective Coverage": f"{r.effective_coverage_pct:.1f}%",
            "Avg Gate (Spatial)": f"{r.avg_gate_value:.3f}",
            "Prediction": r.prediction,
            "Confidence": f"{r.confidence * 100:.1f}%",
            "Time (s)": f"{r.inference_time_sec:.2f}s"
        }
        for r in results
    ])

    print(df.to_string(index=False))
    print("=" * 75 + "\n")
    return results


def run_architectural_ablation(
    video_path: Union[str, Path],
    modes: List[str] = ["spatial_only", "frequency_only", "no_gate", "frame_average", "full"],
    config: VisualPipelineConfig = DEFAULT_CONFIG
) -> Dict[str, FrameCoverageReport]:
    """
    Evaluates an input video under different architectural configurations.
    """
    reports: Dict[str, FrameCoverageReport] = {}

    print("\n" + "=" * 75)
    print(f"      ARCHITECTURAL ABLATION ANALYSIS: {Path(video_path).name}")
    print("=" * 75)

    for mode in modes:
        mode_cfg = VisualPipelineConfig(
            FRAME_COVERAGE_RATIO=config.FRAME_COVERAGE_RATIO,
            MODEL_MODE=mode,
            DEVICE=config.DEVICE
        )
        evaluator = VisualDeepfakeEvaluator(config=mode_cfg)
        rep, _ = evaluator.evaluate_video(video_path, save_metadata=False)
        reports[mode] = rep

    df = pd.DataFrame([
        {
            "Model Mode": mode,
            "Processed Frames": rep.frames_processed_by_cnn,
            "Effective Coverage": f"{rep.coverage_percentage:.1f}%",
            "Avg Gate": f"{rep.avg_gate_value:.3f}",
            "Prediction": rep.prediction,
            "Confidence": f"{rep.confidence * 100:.1f}%"
        }
        for mode, rep in reports.items()
    ])

    print(df.to_string(index=False))
    print("=" * 75 + "\n")
    return reports
