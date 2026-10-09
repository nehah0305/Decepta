import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import DATA_ROOT, FRAME_SIZE, FRAMES_PER_VIDEO, METADATA_ROOT, CACHE_ROOT


def read_frames(video_path: Path, frames_per_video: int, frame_size: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    indices = np.linspace(0, frame_count - 1, frames_per_video).astype(int)
    frames = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if not success:
            capture.release()
            raise RuntimeError(f"Unable to read frame {frame_index}: {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame, (frame_size, frame_size)))
    capture.release()
    return np.stack(frames).transpose(0, 3, 1, 2)


def prepare_split(manifest: pd.DataFrame, split: str, output_dir: Path, frames_per_video: int, frame_size: int) -> None:
    samples = manifest[manifest["split"] == split].reset_index(drop=True)
    frames_path = output_dir / f"{split}_frames.npy"
    labels_path = output_dir / f"{split}_labels.npy"
    frames = np.lib.format.open_memmap(
        frames_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(samples), frames_per_video, 3, frame_size, frame_size),
    )
    labels = np.empty(len(samples), dtype=np.int64)
    jobs = [
        (index, DATA_ROOT / str(row["video_path"]), int(row["label"]), frames_per_video, frame_size)
        for index, row in samples.iterrows()
    ]
    with ProcessPoolExecutor(max_workers=prepare_split.workers) as executor:
        for index, sample_frames, label in executor.map(_read_sample, jobs, chunksize=1):
            frames[index] = sample_frames
            labels[index] = label
            if (index + 1) % 50 == 0 or index + 1 == len(samples):
                print(f"{split}: {index + 1}/{len(samples)}", flush=True)
    frames.flush()
    np.save(labels_path, labels)


def _read_sample(job: tuple[int, Path, int, int, int]) -> tuple[int, np.ndarray, int]:
    index, video_path, label, frames_per_video, frame_size = job
    return index, read_frames(video_path, frames_per_video, frame_size), label


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a disk-backed DFD frame cache for fast GPU training.")
    parser.add_argument("--manifest", type=Path, default=METADATA_ROOT / "dfd.csv")
    parser.add_argument("--output-dir", type=Path, default=CACHE_ROOT)
    parser.add_argument("--frames", type=int, default=FRAMES_PER_VIDEO)
    parser.add_argument("--frame-size", type=int, default=FRAME_SIZE)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_split.workers = args.workers
    for split in args.splits:
        prepare_split(manifest, split, args.output_dir, args.frames, args.frame_size)
    print(f"Cache ready: {args.output_dir}")


if __name__ == "__main__":
    main()