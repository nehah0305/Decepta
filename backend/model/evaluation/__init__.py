# What: This file implements the   init   module in backend/model/evaluation/__init__.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

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

