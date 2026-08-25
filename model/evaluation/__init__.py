"""
Evaluation and Analysis Module for Visual Deepfake Detection System.
"""

from .coverage_analysis import CoverageExperimentResult, run_architectural_ablation, run_frame_coverage_ablation
from .evaluate import VisualDeepfakeEvaluator
from .metrics import calculate_deepfake_metrics
from .sync_evaluation import SyncEvaluationReport, evaluate_synchronization_offsets
from .multimodal_ablation import ModalityAblationResult, run_modality_ablation_experiment
from .multimodal_evaluator import MultimodalDeepfakeEvaluator, MultimodalEvaluationReport

__all__ = [
    "VisualDeepfakeEvaluator",
    "calculate_deepfake_metrics",
    "CoverageExperimentResult",
    "run_frame_coverage_ablation",
    "run_architectural_ablation",
    "SyncEvaluationReport",
    "evaluate_synchronization_offsets",
    "ModalityAblationResult",
    "run_modality_ablation_experiment",
    "MultimodalDeepfakeEvaluator",
    "MultimodalEvaluationReport",
]

