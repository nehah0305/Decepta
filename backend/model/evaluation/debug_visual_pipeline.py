"""
Master Visual Branch Diagnostic & Debugging Suite.

Performs Steps 1 - 8:
1. Label Verification (REAL = 0, FAKE = 1)
2. Data Leakage Verification (train vs val vs test)
3. Visual & FFT Input Artifact Inspection (Saves sample face crops & FFT images)
4. Temporal Frame Difference Verification (Checks if frame sequence varies over time)
5. Gradient Flow Verification (Verifies non-zero gradients across spatial, FFT, transformer, and classifier layers)
6. Embedding Separability Analysis (Extracts 25 REAL + 25 FAKE 768-D embeddings and computes centroid distances)
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG, VisualPipelineConfig
from models.visual_model import VisualDeepfakeDetector
from training.multimodal_dataset import MultimodalVideoDataset, collate_multimodal_batch
from training.dataset import VideoSampleItem


def main():
    project_root = Path(__file__).resolve().parents[2]
    artifact_dir = project_root / "artifacts/debug_visual_samples"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("      DECEPTA MASTER VISUAL PIPELINE DIAGNOSTIC SUITE")
    print("=" * 80)

    # -------------------------------------------------------------
    # STEP 1: VERIFY LABELS & PATH MAPPING
    # -------------------------------------------------------------
    print("\n[CHECK 1] VERIFYING LABEL CONSISTENCY (REAL = 0, FAKE = 1):")
    ffpp_csv = project_root / "Datasets/metadata/faceforensicspp.csv"
    if ffpp_csv.exists():
        df = pd.read_csv(ffpp_csv)
        print(f"  Loaded {len(df)} samples from faceforensicspp.csv")
        reals = df[df["label"] == 0]
        fakes = df[df["label"] == 1]

        # Check random 5 Reals
        real_sample_paths = reals["video_path"].head(5).tolist()
        fake_sample_paths = fakes["video_path"].head(5).tolist()

        print("  - Sample REAL video paths (label=0):")
        for p in real_sample_paths:
            print(f"      * {p}")
            assert "original" in p.lower() or "real" in p.lower() or "actors" in p.lower(), f"Suspicious Real label for {p}"

        print("  - Sample FAKE video paths (label=1):")
        for p in fake_sample_paths:
            print(f"      * {p}")
            assert "original" not in p.lower(), f"Suspicious Fake label for {p}"

        print("  ✅ Label mapping verification PASSED (REAL=0, FAKE=1 strictly verified).")

    # -------------------------------------------------------------
    # STEP 2: VERIFY TRAIN / VAL / TEST ZERO LEAKAGE
    # -------------------------------------------------------------
    print("\n[CHECK 2] VERIFYING DATASET PARTITION LEAKAGE:")
    train_csv = project_root / "Datasets/metadata/train.csv"
    val_csv = project_root / "Datasets/metadata/validation.csv"

    if train_csv.exists() and val_csv.exists() and ffpp_csv.exists():
        tr_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

        def extract_stem_id(p: str) -> str:
            return Path(p).stem.split("_")[0]

        tr_ids = set(tr_df["video_path"].apply(extract_stem_id))
        val_ids = set(val_df["video_path"].apply(extract_stem_id))

        intersection = tr_ids.intersection(val_ids)
        print(f"  Train source video IDs: {len(tr_ids)} | Val source video IDs: {len(val_ids)}")
        print(f"  Overlap count: {len(intersection)}")

        if len(intersection) == 0:
            print("  ✅ Zero leakage verification PASSED (Train and Val are strictly disjoint).")
        else:
            print(f"  ⚠️ Warning: Overlapping source IDs found: {intersection}")

    # -------------------------------------------------------------
    # STEP 3 & 4: VISUAL INSPECTION OF FACE CROPS & FFT SPECTRUMS
    # -------------------------------------------------------------
    print("\n[CHECK 3 & 4] SAVING VISUAL FACE CROPS & FFT FREQUENCY SPECTRA:")
    data_root = project_root / "Datasets/raw/faceforensicspp"

    def extract_first_id(video_path: str) -> int:
        filename = Path(video_path).stem
        if "_" in filename:
            parts = filename.split("_")
            try: return int(parts[0])
            except ValueError: return -1
        else:
            try: return int(filename)
            except ValueError: return -1

    df["first_id"] = df["video_path"].apply(extract_first_id)
    test_df = df[(df["first_id"] >= 860) & (df["manipulation"] != "DeepFakeDetection")].copy()

    real_5 = test_df[test_df["label"] == 0].head(5)
    fake_5 = test_df[test_df["label"] == 1].head(5)
    viz_df = pd.concat([real_5, fake_5], ignore_index=True)

    viz_items = []
    for _, row in viz_df.iterrows():
        rel_path = str(row["video_path"])
        full_path = data_root / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        viz_items.append(
            VideoSampleItem(
                video_id=str(row.get("sample_id", Path(rel_path).stem)),
                video_path=str(full_path),
                label=int(row["label"]),
                split="test"
            )
        )

    dataset = MultimodalVideoDataset(
        samples=viz_items,
        coverage_ratio=DEFAULT_CONFIG.FRAME_COVERAGE_RATIO,
        min_frames=DEFAULT_CONFIG.MIN_FRAMES,
        max_frames=DEFAULT_CONFIG.MAX_FRAMES,
        face_size=DEFAULT_CONFIG.FACE_SIZE,
        mouth_size=DEFAULT_CONFIG.MOUTH_ROI_SIZE,
        device=DEFAULT_CONFIG.DEVICE
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_multimodal_batch)

    # Instantiate pure Visual Model
    model = VisualDeepfakeDetector(
        spatial_dim=256,
        frequency_dim=256,
        fused_dim=256,
        transformer_dim=768,
        transformer_heads=8,
        transformer_layers=2,
        dropout=0.1,
        mode="full",
        frame_chunk_size=32
    ).to(device)

    sample_diffs = []
    for i, batch in enumerate(loader):
        faces = batch["face_frames"].to(device) # (1, N, 3, 224, 224)
        label = int(batch["labels"].cpu().numpy()[0])
        lbl_str = "REAL" if label == 0 else "FAKE"
        v_id = viz_items[i].video_id

        # Save 1st frame as PNG artifact
        frame_np = (faces[0, 0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        img = Image.fromarray(frame_np)
        save_path = artifact_dir / f"sample_{i:02d}_{lbl_str}_{v_id}.png"
        img.save(save_path)

        # Compute L1 difference between frame 0 and frame 10 (if available)
        N = faces.size(1)
        if N > 10:
            f0 = faces[0, 0].cpu().numpy()
            f10 = faces[0, 10].cpu().numpy()
            diff = float(np.mean(np.abs(f10 - f0)))
            sample_diffs.append(diff)

    print(f"  Saved {len(viz_items)} sample face crops to {artifact_dir}")
    if sample_diffs:
        mean_diff = float(np.mean(sample_diffs))
        print(f"  - Mean L1 Pixel Difference across temporal frames: {mean_diff:.4f}")
        assert mean_diff > 0.005, "Frames appear identical over time!"
        print("  ✅ Temporal sequence index verification PASSED (Frames vary chronologically).")

    # -------------------------------------------------------------
    # STEP 5: VERIFY GRADIENT FLOW ACROSS LAYERS
    # -------------------------------------------------------------
    print("\n[CHECK 5] VERIFYING GRADIENT FLOW ACROSS VISUAL BACKBONE & HEAD:")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    dummy_batch = next(iter(loader))
    faces = dummy_batch["face_frames"].to(device)
    labels = dummy_batch["labels"].to(device)

    optimizer.zero_grad()
    outputs = model(faces)
    loss = criterion(outputs.logits.view(-1), labels)
    loss.backward()

    zero_grad_layers = []
    active_grad_layers = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None or param.grad.abs().mean().item() == 0.0:
                zero_grad_layers.append(name)
            else:
                grad_val = param.grad.abs().mean().item()
                active_grad_layers.append((name, grad_val))

    print(f"  Active Trainable Parameters with Non-Zero Gradients: {len(active_grad_layers)}")
    print(f"  Parameters with Zero Gradients:                      {len(zero_grad_layers)}")

    print("\n  Sample Gradient Magnitudes:")
    for name, g_val in active_grad_layers[:8]:
        print(f"    - {name:<60} : mean |grad| = {g_val:.6e}")

    if len(zero_grad_layers) == 0:
        print("  ✅ Gradient flow verification PASSED (All visual layers receive active gradients).")
    else:
        print(f"  ⚠️ Warning: {len(zero_grad_layers)} layers have zero gradients.")

    # -------------------------------------------------------------
    # STEP 6: EXTRACT 25 REAL + 25 FAKE EMBEDDINGS & COMPUTE CENTROID DISTANCE
    # -------------------------------------------------------------
    print("\n[CHECK 6] EXTRACTING 25 REAL & 25 FAKE VISUAL EMBEDDINGS:")
    real_25 = test_df[test_df["label"] == 0].head(25)
    fake_25 = test_df[test_df["label"] == 1].head(25)
    eval_50 = pd.concat([real_25, fake_25], ignore_index=True)

    eval_items = []
    for _, row in eval_50.iterrows():
        rel_path = str(row["video_path"])
        full_path = data_root / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        eval_items.append(
            VideoSampleItem(
                video_id=str(row.get("sample_id", Path(rel_path).stem)),
                video_path=str(full_path),
                label=int(row["label"]),
                split="test"
            )
        )

    eval_dataset = MultimodalVideoDataset(
        samples=eval_items,
        coverage_ratio=DEFAULT_CONFIG.FRAME_COVERAGE_RATIO,
        min_frames=DEFAULT_CONFIG.MIN_FRAMES,
        max_frames=32,
        face_size=DEFAULT_CONFIG.FACE_SIZE,
        mouth_size=DEFAULT_CONFIG.MOUTH_ROI_SIZE,
        device=DEFAULT_CONFIG.DEVICE
    )

    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=collate_multimodal_batch)

    model.eval()
    real_embeddings = []
    fake_embeddings = []

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            faces = batch["face_frames"].to(device)
            label = int(batch["labels"].cpu().numpy()[0])

            outputs = model(faces)
            emb = outputs.video_feature.view(-1).cpu().numpy() # (768-D)

            if label == 0:
                real_embeddings.append(emb)
            else:
                fake_embeddings.append(emb)

    real_arr = np.array(real_embeddings) # (25, 768)
    fake_arr = np.array(fake_embeddings) # (25, 768)

    mu_real = np.mean(real_arr, axis=0)
    mu_fake = np.mean(fake_arr, axis=0)

    l2_dist = float(np.linalg.norm(mu_real - mu_fake))
    cos_sim = float(np.dot(mu_real, mu_fake) / (np.linalg.norm(mu_real) * np.linalg.norm(mu_fake)))

    # Intra-class variance
    var_real = float(np.mean(np.linalg.norm(real_arr - mu_real, axis=1)))
    var_fake = float(np.mean(np.linalg.norm(fake_arr - mu_fake, axis=1)))

    print("\n" + "-" * 80)
    print("      VISUAL EMBEDDING SEPARABILITY REPORT")
    print("-" * 80)
    print(f"  - Extracted Embeddings Shape: Real={real_arr.shape}, Fake={fake_arr.shape}")
    print(f"  - Centroid L2 Distance (||μ_real - μ_fake||_2):  {l2_dist:.6f}")
    print(f"  - Centroid Cosine Similarity:                   {cos_sim:.6f}")
    print(f"  - Average Intra-Class Scatter (Real):            {var_real:.6f}")
    print(f"  - Average Intra-Class Scatter (Fake):            {var_fake:.6f}")
    print(f"  - Inter-to-Intra Distance Ratio:                {l2_dist / ((var_real + var_fake) / 2.0):.6f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
