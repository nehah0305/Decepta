"""
Validation Module for Video Deepfake Detection Model.

Evaluates video-level predictions and computes classification performance metrics:
- Binary Cross-Entropy Loss
- Classification Accuracy
- Area Under the ROC Curve (ROC-AUC)
- Precision, Recall, and F1 Score
"""

import logging
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Computes comprehensive binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # ROC AUC calculation (Mann-Whitney U statistic formulation)
    try:
        if len(np.unique(y_true)) > 1:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_true, y_prob))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "auc": round(float(auc), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Runs full evaluation over the validation set.
    """
    model.eval()
    total_loss = 0.0
    all_targets: list = []
    all_probs: list = []

    with torch.no_grad():
        for batch in val_loader:
            frames = batch["face_frames"].to(device)       # (B, N, 3, H, W)
            padding_mask = batch["padding_mask"].to(device) # (B, N)
            labels = batch["labels"].to(device)             # (B,)

            outputs = model(frames, padding_mask=padding_mask)
            logits = outputs.logits.view(-1)
            loss = criterion(logits, labels)

            total_loss += loss.item() * frames.size(0)
            probs = outputs.probability.view(-1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    n_samples = len(all_targets)
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0

    metrics = compute_binary_metrics(np.array(all_targets), np.array(all_probs))
    metrics["loss"] = round(float(avg_loss), 4)

    return metrics
