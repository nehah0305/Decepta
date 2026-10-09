from pathlib import Path
import re

import cv2
import pandas as pd

from ..config import (
    DATASETS_DIR,
    MASTER_MANIFEST,
    METADATA_DIR,
    REPORTS_DIR,
    TEST_MANIFEST,
    TRAIN_MANIFEST,
    VALIDATION_MANIFEST,
)


DATASET_ROOT = DATASETS_DIR / "raw" / "DFD"
ORIGINAL_ROOT = DATASET_ROOT / "DFD_original sequences"
MANIPULATED_ROOT = DATASET_ROOT / "DFD_manipulated_sequences" / "DFD_manipulated_sequences"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ORIGINAL_PATTERN = re.compile(r"^(?P<source>\d+)__(?P<action>.+)$")
MANIPULATED_PATTERN = re.compile(r"^(?P<source>\d+)_(?P<target>\d+)__(?P<action>.+)__(?P<hash>[A-Za-z0-9]+)$")


def _metadata(video_path: Path) -> tuple[int, int, int, str]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec_number = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((codec_number >> (8 * index)) & 0xFF) for index in range(4)).strip()
    capture.release()
    return frame_count, width, height, codec or "unknown"


def _row(video_path: Path, relative_path: str, sample_id: str, label: int, source_id: str, manipulation: str, generator: str) -> dict:
    frame_count, width, height, codec = _metadata(video_path)
    return {
        "sample_id": sample_id,
        "dataset": "DeepFakeDetection (DFD)",
        "video_path": relative_path,
        "label": label,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "codec": codec,
        "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
        "manipulation": manipulation,
        "generator": generator,
        "subject_id": source_id,
        "source_id": source_id,
        "original_video_id": None,
        "split_group_id": source_id,
        "audio_path": None,
        "split": None,
    }


def create_manifest() -> pd.DataFrame:
    if not ORIGINAL_ROOT.exists() or not MANIPULATED_ROOT.exists():
        raise FileNotFoundError(f"Expected DFD folders not found under {DATASET_ROOT}")

    original_paths = sorted(path for path in ORIGINAL_ROOT.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    manipulated_paths = sorted(path for path in MANIPULATED_ROOT.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    if not original_paths or not manipulated_paths:
        raise FileNotFoundError("DFD must contain both original and manipulated videos")

    rows = []
    for index, video_path in enumerate(original_paths):
        match = ORIGINAL_PATTERN.match(video_path.stem)
        if not match:
            raise ValueError(f"Unexpected DFD original filename: {video_path.name}")
        source_id = f"DFD_SOURCE_{int(match['source']):02d}"
        rows.append(_row(video_path, video_path.relative_to(DATASETS_DIR).as_posix(), f"DFD_{index:07d}", 0, source_id, "Original", "None"))

    offset = len(rows)
    for index, video_path in enumerate(manipulated_paths, offset):
        match = MANIPULATED_PATTERN.match(video_path.stem)
        if not match:
            raise ValueError(f"Unexpected DFD manipulated filename: {video_path.name}")
        source_id = f"DFD_SOURCE_{int(match['source']):02d}"
        rows.append(_row(video_path, video_path.relative_to(DATASETS_DIR).as_posix(), f"DFD_{index:07d}", 1, source_id, "DeepFake", "DeepFakeDetection"))

    return pd.DataFrame(rows)


def assign_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    from sklearn.model_selection import GroupShuffleSplit

    manifest = manifest.copy()
    first_splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_indices, temporary_indices = next(first_splitter.split(manifest, groups=manifest["split_group_id"]))
    manifest["split"] = "temporary"
    manifest.loc[train_indices, "split"] = "train"

    temporary = manifest.loc[temporary_indices]
    second_splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    validation_indices, test_indices = next(second_splitter.split(temporary, groups=temporary["split_group_id"]))
    manifest.loc[temporary.index[validation_indices], "split"] = "validation"
    manifest.loc[temporary.index[test_indices], "split"] = "test"
    return manifest


def main() -> None:
    manifest = assign_splits(create_manifest())
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(METADATA_DIR / "dfd.csv", index=False)
    manifest.to_csv(MASTER_MANIFEST, index=False)
    for split, path in (("train", TRAIN_MANIFEST), ("validation", VALIDATION_MANIFEST), ("test", TEST_MANIFEST)):
        manifest[manifest["split"] == split].to_csv(path, index=False)
    pd.DataFrame([{
        "dataset": "DeepFakeDetection (DFD)",
        "total_samples": len(manifest),
        "real_samples": int((manifest["label"] == 0).sum()),
        "fake_samples": int((manifest["label"] == 1).sum()),
        "train_samples": int((manifest["split"] == "train").sum()),
        "validation_samples": int((manifest["split"] == "validation").sum()),
        "test_samples": int((manifest["split"] == "test").sum()),
    }]).to_csv(REPORTS_DIR / "dataset_statistics.csv", index=False)
    print(manifest["split"].value_counts().sort_index())
    print(manifest.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()