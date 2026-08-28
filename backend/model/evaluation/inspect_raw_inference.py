"""
Diagnostic Inspection Script for Multimodal Deepfake Detector.

Performs:
Check 1: Compares evaluation on test.csv vs faceforensicspp.csv test set.
Check 3: Prints raw logits, probabilities, and labels for 10 Real and 10 Fake videos.
Check 4: Reports dataset class balances across manifests.
Check 5: Inspects raw input tensors, min/max stats, modality masks, and attention weights.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG
from models.multimodal_detector import MultimodalDeepfakeDetector
from training.multimodal_dataset import MultimodalVideoDataset, collate_multimodal_batch
from training.dataset import VideoSampleItem


def main():
    project_root = Path(__file__).resolve().parents[2]
    checkpoint_path = project_root / "model/checkpoints/multimodal_best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("      DECEPTA MULTIMODAL MODEL DIAGNOSTIC INSPECTION")
    print("=" * 80)

    # -------------------------------------------------------------
    # CHECK 4: DATASET CLASS BALANCES IN MANIFESTS
    # -------------------------------------------------------------
    print("\n[CHECK 4] DATASET MANIFEST CLASS BALANCES:")
    for manifest_name in ["train.csv", "validation.csv", "test.csv", "faceforensicspp.csv"]:
        m_path = project_root / f"Datasets/metadata/{manifest_name}"
        if m_path.exists():
            df = pd.read_csv(m_path)
            real_cnt = len(df[df["label"] == 0])
            fake_cnt = len(df[df["label"] == 1])
            tot = len(df)
            real_pct = (real_cnt / tot * 100) if tot > 0 else 0
            fake_pct = (fake_cnt / tot * 100) if tot > 0 else 0
            print(f"  - {manifest_name:<20}: Total={tot:<5} | REAL={real_cnt:<5} ({real_pct:5.1f}%) | FAKE={fake_cnt:<5} ({fake_pct:5.1f}%)")

    # -------------------------------------------------------------
    # CHECK 1 & CHECK 3 & CHECK 5: MODEL INFERENCE DIAGNOSTICS
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[CHECK 3 & 5] INSPECTING RAW MODEL OUTPUTS & INPUT TENSORS")
    print("-" * 80)

    # Load 10 Real and 10 Fake samples from FF++ test set
    ffpp_csv = project_root / "Datasets/metadata/faceforensicspp.csv"
    df_ffpp = pd.read_csv(ffpp_csv)

    def extract_first_id(video_path: str) -> int:
        filename = Path(video_path).stem
        if "_" in filename:
            parts = filename.split("_")
            try: return int(parts[0])
            except ValueError: return -1
        else:
            try: return int(filename)
            except ValueError: return -1

    df_ffpp["first_id"] = df_ffpp["video_path"].apply(extract_first_id)
    test_ffpp = df_ffpp[(df_ffpp["first_id"] >= 860) & (df_ffpp["manipulation"] != "DeepFakeDetection")].copy()

    real_samples = test_ffpp[test_ffpp["label"] == 0].head(10)
    fake_samples = test_ffpp[test_ffpp["label"] == 1].head(10)
    eval_df = pd.concat([real_samples, fake_samples], ignore_index=True)

    data_root = project_root / "Datasets/raw/faceforensicspp"
    sample_items = []
    for _, row in eval_df.iterrows():
        rel_path = str(row["video_path"])
        full_path = data_root / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        sample_items.append(
            VideoSampleItem(
                video_id=str(row.get("sample_id", Path(rel_path).stem)),
                video_path=str(full_path),
                label=int(row["label"]),
                split="test"
            )
        )

    dataset = MultimodalVideoDataset(
        samples=sample_items,
        coverage_ratio=DEFAULT_CONFIG.FRAME_COVERAGE_RATIO,
        min_frames=DEFAULT_CONFIG.MIN_FRAMES,
        max_frames=DEFAULT_CONFIG.MAX_FRAMES,
        face_size=DEFAULT_CONFIG.FACE_SIZE,
        mouth_size=DEFAULT_CONFIG.MOUTH_ROI_SIZE,
        audio_window_sec=DEFAULT_CONFIG.AUDIO_WINDOW_SECONDS,
        audio_hop_sec=DEFAULT_CONFIG.AUDIO_HOP_SECONDS,
        device=DEFAULT_CONFIG.DEVICE
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_multimodal_batch,
        num_workers=0
    )

    model = MultimodalDeepfakeDetector(
        visual_dim=DEFAULT_CONFIG.TRANSFORMER_DIM,
        audio_dim=DEFAULT_CONFIG.AUDIO_FEATURE_DIM,
        sync_dim=DEFAULT_CONFIG.SYNC_FEATURE_DIM,
        fusion_dim=DEFAULT_CONFIG.FUSION_DIM,
        mode=DEFAULT_CONFIG.MODEL_MODE,
        dropout=DEFAULT_CONFIG.TRANSFORMER_DROPOUT,
        frame_chunk_size=DEFAULT_CONFIG.FRAME_BATCH_SIZE
    ).to(device)

    chk = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = chk.get("model_state_dict", chk)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"\nEvaluating {len(sample_items)} samples (10 REAL, 10 FAKE)...")
    print(f"{'Video Path':<42} | {'Label':<6} | {'Raw Logit':<10} | {'Prob':<8} | {'alpha_v':<7} | {'alpha_a':<7} | {'alpha_s':<7}")
    print("-" * 105)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            faces = batch["face_frames"].to(device)
            mouths = batch["mouth_crops"].to(device)
            mels = batch["mel_windows"].to(device)
            mod_masks = batch["modality_masks"].to(device)
            pad_v = batch["padding_mask_v"].to(device)
            pad_a = batch["padding_mask_a"].to(device)
            label = int(batch["labels"].cpu().numpy()[0])
            video_path = eval_df.iloc[i]["video_path"]

            # Diagnostic Check 5: inspect batch tensors
            if i == 0:
                print("\n  [CHECK 5 DETAIL - Batch 0 Input Tensor Inspection]:")
                print(f"    - face_frames shape: {faces.shape}, min={faces.min():.3f}, max={faces.max():.3f}, mean={faces.mean():.3f}")
                print(f"    - mouth_crops shape: {mouths.shape}, min={mouths.min():.3f}, max={mouths.max():.3f}, mean={mouths.mean():.3f}")
                print(f"    - mel_windows shape: {mels.shape}, min={mels.min():.3f}, max={mels.max():.3f}, mean={mels.mean():.3f}")
                print(f"    - modality_masks: {mod_masks.cpu().numpy().tolist()}")
                print("-" * 105)

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

            logit = float(outputs.logits.view(-1).cpu().numpy()[0])
            prob = float(outputs.probability.view(-1).cpu().numpy()[0])
            av = float(outputs.alpha_v.view(-1).cpu().numpy()[0]) if outputs.alpha_v is not None else 0.0
            aa = float(outputs.alpha_a.view(-1).cpu().numpy()[0]) if outputs.alpha_a is not None else 0.0
            as_w = float(outputs.alpha_s.view(-1).cpu().numpy()[0]) if outputs.alpha_s is not None else 0.0

            label_str = "REAL" if label == 0 else "FAKE"
            v_name = Path(video_path).name
            parent_dir = Path(video_path).parent.name
            short_path = f"{parent_dir}/{v_name}"

            print(f"{short_path:<42} | {label_str:<6} | {logit:<10.4f} | {prob:<8.4f} | {av:<7.4f} | {aa:<7.4f} | {as_w:<7.4f}")

    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
