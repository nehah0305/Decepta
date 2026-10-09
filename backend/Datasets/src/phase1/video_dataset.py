from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, manifest_path: Path, data_root: Path, split: str, frames_per_video: int = 8, frame_size: int = 224, cache_dir: Path | None = None) -> None:
        self.data_root = Path(data_root)
        self.frames_per_video = frames_per_video
        self.frame_size = frame_size
        manifest = pd.read_csv(manifest_path)
        self.samples = manifest[manifest["split"] == split].reset_index(drop=True)

        if self.samples.empty:
            raise ValueError(f"No samples found for split: {split}")
        missing = {"video_path", "label"} - set(self.samples.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        self.cached_frames = None
        self.cached_labels = None
        if cache_dir is not None:
            frames_path = Path(cache_dir) / f"{split}_frames.npy"
            labels_path = Path(cache_dir) / f"{split}_labels.npy"
            if frames_path.exists() and labels_path.exists():
                self.cached_frames = np.load(frames_path, mmap_mode="r")
                self.cached_labels = np.load(labels_path, mmap_mode="r")
                expected_shape = (len(self.samples), frames_per_video, 3, frame_size, frame_size)
                if self.cached_frames.shape != expected_shape or len(self.cached_labels) != len(self.samples):
                    raise ValueError(f"Cache shape does not match {split} manifest: {self.cached_frames.shape}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cached_frames is not None:
            video = torch.from_numpy(np.array(self.cached_frames[index], copy=True)).float() / 255.0
            label = torch.tensor(int(self.cached_labels[index]), dtype=torch.long)
            return video, label

        row = self.samples.iloc[index]
        video_path = self.data_root / str(row["video_path"])
        frames = self._read_frames(video_path)
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return video, label

    def _read_frames(self, video_path: Path) -> np.ndarray:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            capture.release()
            raise RuntimeError(f"Video has no readable frames: {video_path}")

        indices = np.linspace(0, frame_count - 1, self.frames_per_video).astype(int)
        frames = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success:
                capture.release()
                raise RuntimeError(f"Unable to read frame {frame_index}: {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(frame, (self.frame_size, self.frame_size)))
        capture.release()
        return np.stack(frames)