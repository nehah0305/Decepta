"""
Test Dataset Evaluation Script for Multimodal Deepfake Detection.

Evaluates trained checkpoint on test set manifest (test.csv) and reports:
1. Overall Model Accuracy (%)
2. Precision, Recall, Specificity, F1-Score
3. ROC-AUC and PR-AUC
4. Confusion Matrix (TP, FP, TN, FN)
5. Average Audio-Visual Synchronization Score
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure model root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG, VisualPipelineConfig
from models.multimodal_detector import MultimodalDeepfakeDetector
from training.multimodal_dataset import MultimodalVideoDataset, collate_multimodal_batch
from training.dataset import VideoSampleItem
from evaluation.metrics import calculate_deepfake_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_test_samples(csv_path: Path, data_root: Path, max_samples: Optional[int] = None) -> List[VideoSampleItem]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Test manifest not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    
    if max_samples and max_samples > 0:
        df = df.head(max_samples)

    items = []
    for _, row in df.iterrows():
        rel_path = str(row["video_path"])
        full_path = data_root / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        items.append(
            VideoSampleItem(
                video_id=str(row.get("sample_id", Path(rel_path).stem)),
                video_path=str(full_path),
                label=int(row["label"]),
                split="test"
            )
        )
    return items


def evaluate_test_dataset(
    checkpoint_path: Path,
    test_manifest_path: Path,
    data_root_path: Path,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    config: VisualPipelineConfig = DEFAULT_CONFIG
):
    device = torch.device(config.DEVICE)
    logger.info(f"Using device: {device} for test evaluation.")

    # 1. Load Test Dataset Samples
    test_samples = load_test_samples(test_manifest_path, data_root_path, max_samples=max_samples)
    logger.info(f"Loaded {len(test_samples)} test samples from {test_manifest_path.name}.")

    if len(test_samples) == 0:
        logger.error("No valid test samples found. Exiting.")
        return

    # 2. Build Dataset and DataLoader
    test_dataset = MultimodalVideoDataset(
        samples=test_samples,
        coverage_ratio=config.FRAME_COVERAGE_RATIO,
        min_frames=config.MIN_FRAMES,
        max_frames=config.MAX_FRAMES,
        face_size=config.FACE_SIZE,
        mouth_size=config.MOUTH_ROI_SIZE,
        audio_window_sec=config.AUDIO_WINDOW_SECONDS,
        audio_hop_sec=config.AUDIO_HOP_SECONDS,
        device=config.DEVICE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_multimodal_batch,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # 3. Instantiate Multimodal Detector Model
    model = MultimodalDeepfakeDetector(
        visual_dim=config.TRANSFORMER_DIM,
        audio_dim=config.AUDIO_FEATURE_DIM,
        sync_dim=config.SYNC_FEATURE_DIM,
        fusion_dim=config.FUSION_DIM,
        mode=config.MODEL_MODE,
        dropout=config.TRANSFORMER_DROPOUT,
        frame_chunk_size=config.FRAME_BATCH_SIZE
    ).to(device)

    # 4. Load Checkpoint Weights
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    chk = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = chk.get("model_state_dict", chk)
    model.load_state_dict(state_dict, strict=False)
    logger.info(f"Successfully loaded model checkpoint from: {checkpoint_path}")

    model.eval()

    # 5. Evaluation Loop
    all_targets = []
    all_probs = []
    all_preds = []
    sync_scores = []
    alpha_v_list = []
    alpha_a_list = []
    alpha_s_list = []

    start_time = time.time()

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            if device.type == "cuda":
                torch.cuda.empty_cache()

            faces = batch["face_frames"].to(device)
            mouths = batch["mouth_crops"].to(device)
            mels = batch["mel_windows"].to(device)
            mod_masks = batch["modality_masks"].to(device)
            pad_v = batch["padding_mask_v"].to(device)
            pad_a = batch["padding_mask_a"].to(device)
            labels = batch["labels"].to(device)

            use_amp = config.USE_AMP and (device.type == "cuda")
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(
                    face_frames=faces,
                    mouth_crops=mouths,
                    mel_windows=mels,
                    modality_mask=mod_masks,
                    padding_mask_v=pad_v,
                    padding_mask_a=pad_a
                )

            probs = outputs.probability.view(-1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            targets = labels.cpu().numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

            if outputs.sync_score is not None:
                sync_scores.extend(outputs.sync_score.view(-1).cpu().numpy().tolist())

            if outputs.alpha_v is not None:
                alpha_v_list.extend(outputs.alpha_v.view(-1).cpu().numpy().tolist())
            if outputs.alpha_a is not None:
                alpha_a_list.extend(outputs.alpha_a.view(-1).cpu().numpy().tolist())
            if outputs.alpha_s is not None:
                alpha_s_list.extend(outputs.alpha_s.view(-1).cpu().numpy().tolist())

            logger.info(f"Evaluated Test Step [{step+1:03d}/{len(test_loader):03d}] | Processed {len(all_targets)} samples")

    elapsed = time.time() - start_time

    # 6. Calculate Metrics
    y_true = np.array(all_targets)
    y_probs = np.array(all_probs)
    metrics = calculate_deepfake_metrics(y_true, y_probs)

    avg_sync = float(np.mean(sync_scores)) if sync_scores else 0.5
    avg_a_v = float(np.mean(alpha_v_list)) if alpha_v_list else 0.33
    avg_a_a = float(np.mean(alpha_a_list)) if alpha_a_list else 0.33
    avg_a_s = float(np.mean(alpha_s_list)) if alpha_s_list else 0.33

    # 7. Print Master Evaluation Report
    sep = "=" * 70
    print("\n" + sep)
    print("      MULTIMODAL DEEPFAKE DETECTION — TEST ACCURACY REPORT")
    print(sep)
    print(f" Checkpoint Evaluated:       {checkpoint_path.name}")
    print(f" Test Manifest Evaluated:     {test_manifest_path.name}")
    print(f" Total Test Samples:          {metrics['total_samples']}")
    print(f" Time Elapsed:                {elapsed:.2f}s ({elapsed/metrics['total_samples']:.2f}s per video)")
    print("-" * 70)
    print(" 1. CLASSIFICATION METRICS:")
    print(f"    - ACCURACY:               {metrics['accuracy'] * 100:>6.2f}%")
    print(f"    - ROC-AUC Score:          {metrics['roc_auc'] * 100:>6.2f}%")
    print(f"    - PR-AUC Score:           {metrics['pr_auc'] * 100:>6.2f}%")
    print(f"    - Precision:              {metrics['precision'] * 100:>6.2f}%")
    print(f"    - Recall (Sensitivity):   {metrics['recall'] * 100:>6.2f}%")
    print(f"    - Specificity:            {metrics['specificity'] * 100:>6.2f}%")
    print(f"    - F1-Score:               {metrics['f1_score'] * 100:>6.2f}%")
    print("-" * 70)
    print(" 2. CONFUSION MATRIX SUMMARY:")
    print(f"    - True Positives (TP):    {metrics['true_positives']:>6d}  [Correctly Identified Fake]")
    print(f"    - True Negatives (TN):    {metrics['true_negatives']:>6d}  [Correctly Identified Real]")
    print(f"    - False Positives (FP):   {metrics['false_positives']:>6d}  [Real Misclassified as Fake]")
    print(f"    - False Negatives (FN):   {metrics['false_negatives']:>6d}  [Fake Misclassified as Real]")
    print("-" * 70)
    print(" 3. MODALITY ATTENTION & SYNC SCORES:")
    print(f"    - Average Visual Weight (alpha_v):  {avg_a_v:.4f}")
    print(f"    - Average Audio Weight (alpha_a):   {avg_a_a:.4f}")
    print(f"    - Average Sync Weight (alpha_s):    {avg_a_s:.4f}")
    print(f"    - Audio-Visual Sync Confidence:     {avg_sync * 100:.2f}%")
    print(sep + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Deepfake Detector on Test Dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multimodal_best.pt", help="Path to trained model checkpoint")
    parser.add_argument("--test-manifest", type=str, default="Datasets/metadata/test.csv", help="Test CSV manifest path")
    parser.add_argument("--data-root", type=str, default="Datasets/raw/faceforensicspp/c23", help="Root folder for video files")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum test samples to evaluate (optional)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    checkpoint_path = project_root / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    test_manifest_path = project_root / args.test_manifest if not Path(args.test_manifest).is_absolute() else Path(args.test_manifest)
    data_root_path = project_root / args.data_root if not Path(args.data_root).is_absolute() else Path(args.data_root)

    evaluate_test_dataset(
        checkpoint_path=checkpoint_path,
        test_manifest_path=test_manifest_path,
        data_root_path=data_root_path,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
