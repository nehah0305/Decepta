"""
Evaluation Metrics for Visual Deepfake Detection System.
"""

from typing import Dict, Tuple
import numpy as np


def calculate_deepfake_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes precision, recall, specificity, F1, accuracy, and ROC-AUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    try:
        if len(np.unique(y_true)) > 1:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc = float(roc_auc_score(y_true, y_probs))
            ap = float(average_precision_score(y_true, y_probs))
        else:
            auc = 0.5
            ap = 0.5
    except Exception:
        auc = 0.5
        ap = 0.5

    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "accuracy": round(float(accuracy), 4),
        "balanced_accuracy": round(float(balanced_accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "pr_auc": round(float(ap), 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "total_samples": total
    }
