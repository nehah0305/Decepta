"""
Evaluation and Analysis Module for Visual Deepfake Detection System.
"""

from .coverage_analysis import CoverageExperimentResult, run_architectural_ablation, run_frame_coverage_ablation
from .evaluate import VisualDeepfakeEvaluator
from .metrics import calculate_deepfake_metrics

__all__ = [
    "VisualDeepfakeEvaluator",
    "calculate_deepfake_metrics",
    "CoverageExperimentResult",
    "run_frame_coverage_ablation",
    "run_architectural_ablation",
]
