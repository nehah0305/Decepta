"""
Comprehensive FF++ Test Set Evaluation Script for Multimodal Deepfake Detector.

Evaluates trained checkpoint (multimodal_best.pt) on FaceForensics++ test set:
- Breakdown by each manipulation type (Original, Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures)
- Overall Real vs. Fake accuracy
- Confusion Matrix (TP, TN, FP, FN, False Positive Rate, False Negative Rate)
- ROC-AUC and PR-AUC scores
- Performance across Quality / Compression tiers (High Quality vs Low Quality)
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
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure model root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG, VisualPipelineConfig
from models.multimodal_detector import MultimodalDeepfakeDetector
from training.multimodal_dataset import MultimodalVideoDataset, collate_multimodal_batch
from training.dataset import VideoSampleItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_first_id(video_path: str) -> int:
    filename = Path(video_path).stem
    if "_" in filename:
        parts = filename.split("_")
        try:
            return int(parts[0])
        except ValueError:
            return -1
    else:
        try:
            return int(filename)
        except ValueError:
            return -1


def load_ffpp_test_samples(
    csv_path: Path,
    data_root: Path,
    max_samples: Optional[int] = None
) -> Tuple[List[VideoSampleItem], pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if "first_id" not in df.columns:
        df["first_id"] = df["video_path"].apply(extract_first_id)

    # Filter standard test split (first_id >= 860) and exclude DFD
    test_df = df[(df["first_id"] >= 860) & (df["manipulation"] != "DeepFakeDetection")].copy()

    if len(test_df) == 0:
        logger.warning("No samples matching first_id >= 860 found. Using split == 'test' or all non-DFD samples.")
        if "split" in df.columns and (df["split"] == "test").any():
            test_df = df[(df["split"] == "test") & (df["manipulation"] != "DeepFakeDetection")].copy()
        else:
            test_df = df[df["manipulation"] != "DeepFakeDetection"].copy()

    if max_samples and max_samples > 0:
        per_manip = max(1, max_samples // len(test_df["manipulation"].unique()))
        sampled_dfs = [group.head(per_manip) for _, group in test_df.groupby("manipulation")]
        test_df = pd.concat(sampled_dfs, ignore_index=True)

    items = []
    for _, row in test_df.iterrows():
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
    return items, test_df


def run_ffpp_evaluation(
    checkpoint_path: Path,
    manifest_path: Path,
    data_root: Path,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    config: VisualPipelineConfig = DEFAULT_CONFIG
) -> Dict:
    device = torch.device(config.DEVICE)
    logger.info(f"Using compute device: {device}")

    # 1. Load Samples
    test_samples, test_df = load_ffpp_test_samples(manifest_path, data_root, max_samples=max_samples)
    logger.info(f"Loaded {len(test_samples)} FF++ test set video samples across {len(test_df['manipulation'].unique())} manipulation types.")

    if len(test_samples) == 0:
        raise ValueError("No test samples were loaded!")

    # 2. Build Dataset & DataLoader
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

    # 3. Model Setup
    model = MultimodalDeepfakeDetector(
        visual_dim=config.TRANSFORMER_DIM,
        audio_dim=config.AUDIO_FEATURE_DIM,
        sync_dim=config.SYNC_FEATURE_DIM,
        fusion_dim=config.FUSION_DIM,
        mode=config.MODEL_MODE,
        dropout=config.TRANSFORMER_DROPOUT,
        frame_chunk_size=config.FRAME_BATCH_SIZE
    ).to(device)

    chk = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = chk.get("model_state_dict", chk)
    model.load_state_dict(state_dict, strict=False)
    logger.info(f"Successfully loaded checkpoint: {checkpoint_path}")
    model.eval()

    # 4. Inference Loop
    all_probs = []
    all_preds = []
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

            use_amp = False
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

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

            logger.info(f"Batch [{step+1:03d}/{len(test_loader):03d}] | Processed {len(all_probs)} videos")

    elapsed_time = time.time() - start_time

    # Attach predictions to dataframe
    test_df = test_df.iloc[:len(all_probs)].copy()
    test_df["prob"] = all_probs
    test_df["pred"] = all_preds

    # 5. Global Metrics & Confusion Matrix
    y_true = test_df["label"].values
    y_prob = np.nan_to_num(test_df["prob"].values, nan=0.5)
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 1.0

    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec_arr, prec_arr)

    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # 6. Per-Manipulation Analysis
    manip_results = {}
    for manip, group in test_df.groupby("manipulation"):
        m_true = group["label"].values
        m_pred = group["pred"].values
        m_prob = group["prob"].values
        m_acc = accuracy_score(m_true, m_pred)
        m_fp = int(np.sum((m_true == 0) & (m_pred == 1)))
        m_fn = int(np.sum((m_true == 1) & (m_pred == 0)))
        m_count = len(group)
        manip_results[manip] = {
            "count": m_count,
            "accuracy": float(m_acc),
            "false_positives": m_fp,
            "false_negatives": m_fn,
            "mean_prob": float(np.mean(m_prob))
        }

    # 7. Quality / Compression Tier Analysis (HQ vs LQ by file size median)
    median_size = test_df["file_size_mb"].median()
    hq_df = test_df[test_df["file_size_mb"] >= median_size]
    lq_df = test_df[test_df["file_size_mb"] < median_size]

    def get_tier_metrics(df_tier):
        if len(df_tier) == 0:
            return {}
        t_true = df_tier["label"].values
        t_prob = np.nan_to_num(df_tier["prob"].values, nan=0.5)
        t_pred = (t_prob >= 0.5).astype(int)
        t_tn, t_fp, t_fn, t_tp = confusion_matrix(t_true, t_pred, labels=[0, 1]).ravel()
        t_auc = roc_auc_score(t_true, t_prob) if len(np.unique(t_true)) > 1 else 1.0
        return {
            "count": len(df_tier),
            "accuracy": float(accuracy_score(t_true, t_pred)),
            "roc_auc": float(t_auc),
            "true_positives": int(t_tp),
            "true_negatives": int(t_tn),
            "false_positives": int(t_fp),
            "false_negatives": int(t_fn),
        }

    hq_metrics = get_tier_metrics(hq_df)
    lq_metrics = get_tier_metrics(lq_df)

    # 8. Formatted Summary Report Printing
    line = "=" * 78
    subline = "-" * 78
    print("\n" + line)
    print("      FACEFORENSICS++ (FF++) TEST SET EVALUATION REPORT")
    print(line)
    print(f" Checkpoint Evaluated:        {checkpoint_path.name}")
    print(f" Dataset Manifest:           {manifest_path.name}")
    print(f" Total Test Samples:          {len(test_df)}")
    print(f" Inference Time:             {elapsed_time:.2f} seconds ({elapsed_time/len(test_df):.2f}s / video)")
    print(subline)
    print(" 1. OVERALL MODEL PERFORMANCE:")
    bal_acc = (real_acc + fake_acc) / 2.0
    print(f"    - Overall Test Accuracy:  {acc * 100:>6.2f}%")
    print(f"    - Balanced Accuracy:      {bal_acc * 100:>6.2f}%")
    print(f"    - Real Specificity (TN):  {real_acc * 100:>6.2f}%  ({tn}/{tn+fp})")
    print(f"    - Fake Recall (TP):       {fake_acc * 100:>6.2f}%  ({tp}/{tp+fn})")
    print(f"    - ROC-AUC Score:          {roc_auc * 100:>6.2f}%")
    print(f"    - PR-AUC Score:           {pr_auc * 100:>6.2f}%")
    print(f"    - Precision:              {prec * 100:>6.2f}%")
    print(f"    - Recall (Sensitivity):   {rec * 100:>6.2f}%")
    print(f"    - Specificity:            {real_acc * 100:>6.2f}%")
    print(f"    - F1-Score:               {f1 * 100:>6.2f}%")
    print(subline)
    print(" 2. CONFUSION MATRIX & ERROR BREAKDOWN:")
    print(f"    - True Positives (TP):    {tp:>5d}  [Correctly Detected Fakes]")
    print(f"    - True Negatives (TN):    {tn:>5d}  [Correctly Identified Reals]")
    print(f"    - False Positives (FP):   {fp:>5d}  [Real Videos Misclassified as Fake]  (FPR: {fpr_rate*100:.2f}%)")
    print(f"    - False Negatives (FN):   {fn:>5d}  [Fake Videos Misclassified as Real]  (FNR: {fnr_rate*100:.2f}%)")
    print(subline)
    print(" 3. PER-MANIPULATION ACCURACY BREAKDOWN:")
    print(f"    {'Manipulation Type':<22} | {'Count':<6} | {'Accuracy':<10} | {'FP':<5} | {'FN':<5} | {'Mean Prob':<10}")
    print(subline)
    for manip, res in sorted(manip_results.items()):
        print(f"    {manip:<22} | {res['count']:<6d} | {res['accuracy']*100:>8.2f}% | {res['false_positives']:<5d} | {res['false_negatives']:<5d} | {res['mean_prob']:>9.4f}")
    print(subline)
    print(" 4. COMPRESSION / QUALITY TIER ANALYSIS:")
    print(f"    - High Quality Tier (>= {median_size:.2f} MB):")
    print(f"        * Accuracy: {hq_metrics.get('accuracy', 0)*100:.2f}% | ROC-AUC: {hq_metrics.get('roc_auc', 0)*100:.2f}% | FP: {hq_metrics.get('false_positives', 0)} | FN: {hq_metrics.get('false_negatives', 0)}")
    print(f"    - Low Quality Tier (< {median_size:.2f} MB):")
    print(f"        * Accuracy: {lq_metrics.get('accuracy', 0)*100:.2f}% | ROC-AUC: {lq_metrics.get('roc_auc', 0)*100:.2f}% | FP: {lq_metrics.get('false_positives', 0)} | FN: {lq_metrics.get('false_negatives', 0)}")
    print(line + "\n")

    return {
        "overall_accuracy": float(acc),
        "real_accuracy": float(real_acc),
        "fake_accuracy": float(fake_acc),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "false_positive_rate": float(fpr_rate),
        "false_negative_rate": float(fnr_rate),
        "per_manipulation": manip_results,
        "quality_tiers": {"high_quality": hq_metrics, "low_quality": lq_metrics}
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal_best.pt on FF++ Test Set")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoints/multimodal_best.pt")
    parser.add_argument("--manifest", type=str, default="Datasets/metadata/faceforensicspp.csv")
    parser.add_argument("--data-root", type=str, default="Datasets/raw/faceforensicspp")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=180)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    chk_path = project_root / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    man_path = project_root / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    root_path = project_root / args.data_root if not Path(args.data_root).is_absolute() else Path(args.data_root)

    run_ffpp_evaluation(
        checkpoint_path=chk_path,
        manifest_path=man_path,
        data_root=root_path,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
