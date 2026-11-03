# What: This file implements the preprocess dfd faces module in backend/model/training/preprocess_dfd_faces.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

"""
GPU-Accelerated Video Face Preprocessing for Decepta.

Extracts canonical 224x224 aligned faces from video frames using CUDA MTCNN
and stores them as compact uint8 tensors (.pt) on NVMe SSD.

Features:
- Pipelined multithreaded video reading + CUDA batched MTCNN inference
- Exact 2D affine similarity alignment compatible with FaceAlignmentPipeline
- Fully resumable (skips already preprocessed videos)
- Real-time throughput (vids/sec) and ETA reporting

Usage:
    python training/preprocess_dfd_faces.py --config training/train_config_dfd.yaml
"""

import argparse
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import yaml

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.face_alignment import FaceAlignmentPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def sample_video_frames(
    video_path: Path,
    num_frames: int = 16,
    coverage_ratio: float = 0.70
) -> List[np.ndarray]:
    """
    Sequentially reads and samples `num_frames` from a video across `coverage_ratio`.
    Sequential reads are ~5x faster than random keyframe seeking.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Calculate span over coverage_ratio
    usable_frames = max(num_frames, int(total_frames * coverage_ratio))
    start_offset = max(0, (total_frames - usable_frames) // 2)
    step = max(1, usable_frames // num_frames)

    frames = []
    current_idx = 0
    cap_opened = True

    # Seek to start offset if significant
    if start_offset > 10:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_offset)
        current_idx = start_offset

    while len(frames) < num_frames and cap_opened:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        if (current_idx - start_offset) % step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        current_idx += 1
        if current_idx >= start_offset + usable_frames:
            break

    cap.release()
    return frames


def extract_faces_from_video(
    pipeline: FaceAlignmentPipeline,
    frames_rgb: List[np.ndarray],
    target_size: int = 224
) -> torch.Tensor:
    """
    Runs batched MTCNN on CUDA and applies affine alignment to target_size.
    Returns tensor of shape (N, 3, target_size, target_size) in uint8 [0..255].
    """
    if not frames_rgb:
        return torch.zeros(1, 3, target_size, target_size, dtype=torch.uint8)

    # Convert to PIL for MTCNN
    pil_imgs = [Image.fromarray(f) for f in frames_rgb]

    try:
        boxes_batch, probs_batch, lm_batch = pipeline.mtcnn.detect(pil_imgs, landmarks=True)
    except Exception as e:
        logger.warning(f"MTCNN batch detect failed: {e}")
        return torch.zeros(1, 3, target_size, target_size, dtype=torch.uint8)

    valid_faces = []
    pipeline.reset_tracking()

    for idx, (f_rgb, boxes, probs, landmarks) in enumerate(zip(frames_rgb, boxes_batch, probs_batch, lm_batch)):
        if boxes is None or len(boxes) == 0:
            continue

        try:
            box, conf, lm_dict = pipeline._select_primary_face(boxes, probs, landmarks, f_rgb.shape)
            left_eye = tuple(lm_dict["left_eye"])
            right_eye = tuple(lm_dict["right_eye"])
            M = pipeline.compute_similarity_transform(left_eye, right_eye)

            aligned = cv2.warpAffine(
                f_rgb,
                M,
                (target_size, target_size),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101
            )
            # Permute to (3, H, W)
            face_t = torch.from_numpy(aligned).permute(2, 0, 1)  # uint8
            valid_faces.append(face_t)
        except Exception:
            continue

    if not valid_faces:
        return torch.zeros(1, 3, target_size, target_size, dtype=torch.uint8)

    return torch.stack(valid_faces, dim=0)


def run_preprocessing(
    manifest_paths: List[Path],
    data_root: Path,
    output_dir: Path,
    num_frames: int = 16,
    coverage_ratio: float = 0.70,
    face_size: int = 224,
    device: str = "cuda",
    num_reader_threads: int = 4
):
    """
    Main orchestration loop for pipelined face extraction.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all video records
    records = []
    for mpath in manifest_paths:
        if not mpath.exists():
            continue
        df = pd.read_csv(mpath)
        for _, row in df.iterrows():
            sample_id = str(row["sample_id"])
            video_rel = str(row["video_path"])
            video_full = (data_root / video_rel).resolve()
            records.append((sample_id, video_full))

    # Deduplicate by sample_id
    seen = set()
    unique_records = []
    for sid, vpath in records:
        if sid not in seen:
            seen.add(sid)
            unique_records.append((sid, vpath))

    total_videos = len(unique_records)
    logger.info(f"Total unique videos to preprocess: {total_videos}")
    if total_videos == 0:
        logger.error(f"No video records found in manifests: {[str(m) for m in manifest_paths]}")
        return

    # Check already processed
    pending_records = [
        (sid, vpath) for sid, vpath in unique_records
        if not (output_dir / f"{sid}.pt").exists()
    ]
    already_done = total_videos - len(pending_records)
    logger.info(f"Already preprocessed: {already_done}/{total_videos} ({already_done/total_videos*100:.1f}%)")
    logger.info(f"Remaining to process: {len(pending_records)}")

    if not pending_records:
        logger.info("All videos already preprocessed! Exiting.")
        return

    # Initialize GPU MTCNN
    pipeline = FaceAlignmentPipeline(target_size=face_size, device=device)

    # Producer-consumer queue
    video_queue = queue.Queue(maxsize=16)

    def reader_worker(worker_id: int, sub_records: List[Tuple[str, Path]]):
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: worker_id, sub_records.
        Output: Side effects performed by the function; no value is explicitly returned.
        """
        for sid, vpath in sub_records:
            if not vpath.exists():
                video_queue.put((sid, []))
                continue
            frames = sample_video_frames(vpath, num_frames=num_frames, coverage_ratio=coverage_ratio)
            video_queue.put((sid, frames))

    # Split records among reader threads
    threads = []
    chunk_size = (len(pending_records) + num_reader_threads - 1) // num_reader_threads
    for i in range(num_reader_threads):
        chunk = pending_records[i * chunk_size : (i + 1) * chunk_size]
        if chunk:
            t = threading.Thread(target=reader_worker, args=(i, chunk), daemon=True)
            t.start()
            threads.append(t)

    # Consumer (Main thread on GPU)
    start_time = time.time()
    processed_count = 0
    total_to_process = len(pending_records)

    logger.info("=" * 80)
    logger.info(f"STARTING GPU FACE EXTRACTION ON {device.upper()}")
    logger.info(f"  Target size: {face_size}x{face_size} | Sampled frames/video: {num_frames}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 80)

    for _ in range(total_to_process):
        sid, frames = video_queue.get()
        out_path = output_dir / f"{sid}.pt"

        if frames:
            faces_tensor = extract_faces_from_video(pipeline, frames, target_size=face_size)
        else:
            faces_tensor = torch.zeros(1, 3, face_size, face_size, dtype=torch.uint8)

        # Save to disk
        torch.save(faces_tensor, out_path)
        processed_count += 1

        if processed_count % 50 == 0 or processed_count == total_to_process:
            elapsed = time.time() - start_time
            vids_per_sec = processed_count / max(1e-5, elapsed)
            remaining_vids = total_to_process - processed_count
            eta_mins = (remaining_vids / max(1e-5, vids_per_sec)) / 60.0
            overall_pct = (already_done + processed_count) / total_videos * 100.0

            logger.info(
                f"[{already_done + processed_count}/{total_videos}] ({overall_pct:.1f}%) | "
                f"Batch: {processed_count}/{total_to_process} | "
                f"Speed: {vids_per_sec:.2f} vids/s | "
                f"ETA: {eta_mins:.1f} min"
            )

    logger.info("=" * 80)
    logger.info(f"FACE EXTRACTION COMPLETE! Total processed: {processed_count} in {(time.time()-start_time)/60:.1f} min")
    logger.info(f"All face tensors saved in: {output_dir}")
    logger.info("=" * 80)


def main():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    parser = argparse.ArgumentParser(description="Preprocess video faces for Decepta")
    parser.add_argument("--config", type=str, default="training/train_config_dfd.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default="Datasets/preprocessed/dfd_faces", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device for MTCNN (cuda/cpu)")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU video reader threads")
    args = parser.parse_args()

    model_dir = Path(__file__).resolve().parent.parent
    backend_dir = model_dir.parent

    # Load config (try relative to model_dir or backend_dir)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        if (model_dir / cfg_path).exists():
            cfg_path = model_dir / cfg_path
        elif (backend_dir / cfg_path).exists():
            cfg_path = backend_dir / cfg_path

    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    dset_cfg = cfg.get("dataset", {})
    samp_cfg = cfg.get("sampling", {})

    data_root = backend_dir / dset_cfg.get("data_root", "Datasets")
    manifests = [
        backend_dir / dset_cfg.get("train_manifest", "Datasets/metadata/train_dfd.csv"),
        backend_dir / dset_cfg.get("val_manifest", "Datasets/metadata/val_dfd.csv"),
        backend_dir / dset_cfg.get("test_manifest", "Datasets/metadata/test_dfd.csv"),
    ]

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = backend_dir / out_dir
    num_frames = samp_cfg.get("min_frames", 16)
    coverage = samp_cfg.get("coverage_ratio", 0.70)
    face_size = samp_cfg.get("face_size", 224)

    run_preprocessing(
        manifest_paths=manifests,
        data_root=data_root,
        output_dir=out_dir,
        num_frames=num_frames,
        coverage_ratio=coverage,
        face_size=face_size,
        device=args.device,
        num_reader_threads=args.workers
    )


if __name__ == "__main__":
    main()
