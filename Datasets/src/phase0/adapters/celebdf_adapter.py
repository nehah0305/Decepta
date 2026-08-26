from pathlib import Path
import re

import cv2
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ..config import (
    DATASETS_DIR,
    MASTER_MANIFEST,
    METADATA_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TEST_MANIFEST,
    TRAIN_MANIFEST,
    VALIDATION_MANIFEST,
)


DATASET_ROOT = DATASETS_DIR / "raw" / "celeb df (v2)"
TEST_LIST = DATASET_ROOT / "List_of_testing_videos.txt"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _normalise_path(value: str) -> str:
    return value.strip().replace("\\", "/")


def _testing_videos() -> dict[str, int]:
    if not TEST_LIST.exists():
        raise FileNotFoundError(f"Celeb-DF testing list not found: {TEST_LIST}")

    testing = {}
    for line_number, line in enumerate(TEST_LIST.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or parts[0] not in {"0", "1"}:
            raise ValueError(f"Invalid testing-list entry on line {line_number}: {line!r}")
        relative_path = _normalise_path(parts[1])
        # The official list uses 1 for real and 0 for fake; our convention is 0/1.
        label = 0 if parts[0] == "1" else 1
        if relative_path in testing:
            raise ValueError(f"Duplicate testing-list path: {relative_path}")
        testing[relative_path] = label
    return testing


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


def _group_id(relative_path: str) -> tuple[str, str, str]:
    path = Path(relative_path)
    stem = path.stem
    directory = path.parts[0]
    if directory == "Celeb-synthesis":
        match = re.fullmatch(r"(id\d+)_(id\d+)_\d+", stem)
        if not match:
            raise ValueError(f"Unexpected Celeb-synthesis filename: {relative_path}")
        first_id, second_id = match.groups()
        pair = "_".join(sorted((first_id, second_id)))
        return f"CELEB_PAIR_{pair}", first_id, second_id
    if directory == "Celeb-real":
        match = re.match(r"(id\d+)_", stem)
        subject = match.group(1) if match else stem
        return f"CELEB_SUBJECT_{subject}", subject, subject
    if directory == "YouTube-real":
        return f"CELEB_YOUTUBE_{stem}", stem, stem
    raise ValueError(f"Unexpected Celeb-DF directory: {directory}")


def create_manifest() -> pd.DataFrame:
    testing = _testing_videos()
    video_paths = sorted(
        path for path in DATASET_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_paths:
        raise FileNotFoundError(f"No videos found under {DATASET_ROOT}")

    rows = []
    seen_paths = set()
    for index, video_path in enumerate(video_paths):
        relative_path = _normalise_path(str(video_path.relative_to(DATASET_ROOT)))
        seen_paths.add(relative_path)
        directory = Path(relative_path).parts[0]
        label = 0 if directory.endswith("real") else 1
        testing_label = testing.get(relative_path)
        if testing_label is not None and testing_label != label:
            raise ValueError(f"Testing-list label disagrees with folder for {relative_path}")
        split_group_id, subject_id, source_id = _group_id(relative_path)
        frame_count, width, height, codec = _metadata(video_path)
        manipulation = "Original" if label == 0 else "DeepFake"
        rows.append({
            "sample_id": f"CELEBDF_{index:07d}",
            "dataset": "Celeb-DF v2",
            "video_path": relative_path,
            "label": label,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "codec": codec,
            "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
            "manipulation": manipulation,
            "generator": "Celeb-synthesis" if label else "None",
            "subject_id": subject_id,
            "source_id": source_id,
            "original_video_id": None,
            "split_group_id": split_group_id,
            "audio_path": None,
            "split": "test" if relative_path in testing else None,
        })

    missing_testing_paths = sorted(set(testing) - seen_paths)
    if missing_testing_paths:
        raise FileNotFoundError(f"Testing list contains missing videos: {missing_testing_paths[:10]}")
    return pd.DataFrame(rows)


def assign_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()
    remaining = manifest[manifest["split"].isna()]
    validation_fraction = 0.15 / (0.70 + 0.15)
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=RANDOM_SEED)
    train_indices, validation_indices = next(
        splitter.split(remaining, groups=remaining["split_group_id"])
    )
    manifest.loc[remaining.index[train_indices], "split"] = "train"
    manifest.loc[remaining.index[validation_indices], "split"] = "validation"
    return manifest


def main() -> None:
    manifest = assign_splits(create_manifest())
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(METADATA_DIR / "celebdf.csv", index=False)
    manifest.to_csv(MASTER_MANIFEST, index=False)
    for split, path in (("train", TRAIN_MANIFEST), ("validation", VALIDATION_MANIFEST), ("test", TEST_MANIFEST)):
        manifest[manifest["split"] == split].to_csv(path, index=False)
    pd.DataFrame([{
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