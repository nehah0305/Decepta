"""
Reusable Evaluation Script for Visual Models on the Full Balanced Validation Partition (320 Samples).
Evaluates models across ALL validation samples with zero step truncation.
"""

import os
import sys
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.multimodal_dataset import MultimodalVideoDataset, collate_multimodal_batch
from training.dataset import VideoSampleItem
from evaluation.metrics import calculate_deepfake_metrics
from training.train_spatial_alone import PureSpatialDetector
from training.train_fft_alone import PureFFTDetector
from training.train_spatial_pretrained import PretrainedSpatialDetector
from training.train_spatial_pretrained_stage_b import PretrainedSpatialDetectorFineTune


def evaluate_checkpoint(model_type: str, checkpoint_path: str, batch_size: int = 4):
    project_root = Path(__file__).resolve().parents[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_file = project_root / checkpoint_path
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

    print(f"\n===========================================================")
    print(f"   FULL VALIDATION EVALUATION: {model_type.upper()}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"===========================================================")

    # Instantiate model architecture
    if model_type == "spatial_scratch":
        model = PureSpatialDetector(feature_dim=256, dropout=0.1)
    elif model_type == "fft_scratch":
        model = PureFFTDetector(feature_dim=256, dropout=0.1)
    elif model_type == "spatial_frozen":
        model = PretrainedSpatialDetector(feature_dim=256, dropout=0.1, freeze_backbone=True)
    elif model_type == "spatial_finetuned":
        model = PretrainedSpatialDetectorFineTune(feature_dim=256, dropout=0.1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)
    ckpt = torch.load(ckpt_file, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load 320-sample validation dataset
    val_df = pd.read_csv(project_root / "Datasets/metadata/val_ffpp.csv")
    d_root = project_root / "Datasets/raw/faceforensicspp"

    val_items = [
        VideoSampleItem(
            video_id=str(r.get("sample_id", Path(r["video_path"]).stem)),
            video_path=str(d_root / r["video_path"] if not Path(r["video_path"]).is_absolute() else Path(r["video_path"])),
            label=int(r["label"]),
            split="val"
        )
        for _, r in val_df.iterrows()
    ]

    val_dataset = MultimodalVideoDataset(
        samples=val_items,
        min_frames=16,
        max_frames=16,
        face_size=224,
        device=device.type
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal_batch)

    val_targets = []
    val_probs = []
    val_logits = []

    with torch.no_grad():
        for batch in val_loader:
            faces = batch["face_frames"].to(device)
            pad_v = batch["padding_mask_v"].to(device)
            labels = batch["labels"].to(device).float()

            logits, _ = model(faces, padding_mask=pad_v)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()

            val_logits.extend(logits.view(-1).cpu().numpy().tolist())
            val_probs.extend(probs.tolist())
            val_targets.extend(labels.cpu().numpy().tolist())

    y_true = np.array(val_targets)
    y_probs = np.array(val_probs)
    y_logits = np.array(val_logits)

    metrics = calculate_deepfake_metrics(y_true, y_probs)

    print(f"Total Evaluated Validation Samples: {len(y_true)}")
    print(f"Label Counts: {Counter(y_true)}")
    print(f"Prediction Stats -> Prob Mean: {y_probs.mean():.4f}, Std: {y_probs.std():.4f}, Min: {y_probs.min():.4f}, Max: {y_probs.max():.4f}")
    print(f"REAL Mean Logit: {y_logits[y_true==0].mean():.4f} | FAKE Mean Logit: {y_logits[y_true==1].mean():.4f}")
    print("-" * 65)
    print(f"ROC-AUC           : {metrics['roc_auc']*100:.2f}%")
    print(f"PR-AUC            : {metrics['pr_auc']*100:.2f}%")
    print(f"Accuracy          : {metrics['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']*100:.2f}%")
    print(f"Specificity (REAL): {metrics['specificity']*100:.2f}%")
    print(f"Recall (FAKE)     : {metrics['recall']*100:.2f}%")
    print(f"Precision         : {metrics['precision']*100:.2f}%")
    print(f"F1-Score          : {metrics['f1_score']*100:.2f}%")
    print(f"Confusion Matrix  : TP={metrics['true_positives']}, FP={metrics['false_positives']}, TN={metrics['true_negatives']}, FN={metrics['false_negatives']}")
    print("===========================================================\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Visual Model Checkpoint on 320 Val Samples")
    parser.add_argument("--model_type", type=str, required=True, choices=["spatial_scratch", "fft_scratch", "spatial_frozen", "spatial_finetuned"])
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    evaluate_checkpoint(args.model_type, args.checkpoint)
