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


def _identity_group_ids(relative_paths: list[str]) -> dict[str, str]:
    parent = {}

    def find(identity: str) -> str:
        parent.setdefault(identity, identity)
        if parent[identity] != identity:
            parent[identity] = find(parent[identity])
        return parent[identity]

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for relative_path in relative_paths:
        directory = Path(relative_path).parts[0]
        if directory == "Celeb-synthesis":
            identities = re.findall(r"id\d+", Path(relative_path).stem)
            union(identities[0], identities[1])
        elif directory == "Celeb-real":
            find(re.match(r"(id\d+)_", Path(relative_path).stem).group(1))

    component_names = {}
    groups = {}
    for relative_path in relative_paths:
        directory = Path(relative_path).parts[0]
        if directory == "YouTube-real":
            groups[relative_path] = f"CELEB_YOUTUBE_{Path(relative_path).stem}"
            continue
        identity = re.findall(r"id\d+", Path(relative_path).stem)[0]
        root_identity = find(identity)
        component_names.setdefault(root_identity, f"CELEB_COMPONENT_{len(component_names):03d}")
        groups[relative_path] = component_names[root_identity]
    return groups


def _group_id(relative_path: str, identity_groups: dict[str, str]) -> tuple[str, str, str]:
    path = Path(relative_path)
    stem = path.stem
    directory = path.parts[0]
    if directory == "Celeb-synthesis":
        match = re.fullmatch(r"(id\d+)_(id\d+)_\d+", stem)
        if not match:
            raise ValueError(f"Unexpected Celeb-synthesis filename: {relative_path}")
        first_id, second_id = match.groups()
        return identity_groups[relative_path], first_id, second_id
    if directory == "Celeb-real":
        match = re.match(r"(id\d+)_", stem)
        subject = match.group(1) if match else stem
        return identity_groups[relative_path], subject, subject
    if directory == "YouTube-real":
        return identity_groups[relative_path], stem, stem
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
    relative_paths = [
        _normalise_path(str(video_path.relative_to(DATASET_ROOT)))
        for video_path in video_paths
    ]
    identity_groups = _identity_group_ids(relative_paths)
    seen_paths = set()
    for index, video_path in enumerate(video_paths):
        relative_path = relative_paths[index]
        seen_paths.add(relative_path)
        directory = Path(relative_path).parts[0]
        label = 0 if directory.endswith("real") else 1
        testing_label = testing.get(relative_path)
        if testing_label is not None and testing_label != label:
            raise ValueError(f"Testing-list label disagrees with folder for {relative_path}")
        split_group_id, subject_id, source_id = _group_id(relative_path, identity_groups)
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
    manifest["split"] = None
    group_counts = manifest.groupby("split_group_id")["label"].agg(["size", "sum"])
    identity_groups = group_counts[group_counts["sum"] > 0].sort_values("size", ascending=False).index.tolist()
    if len(identity_groups) != 3:
        raise ValueError(f"Expected three mixed Celeb-DF identity components, found {len(identity_groups)}")

    split_by_group = {
        identity_groups[0]: "train",
        identity_groups[1]: "validation",
        identity_groups[2]: "test",
    }
    youtube_groups = sorted(
        group for group in group_counts.index
        if group.startswith("CELEB_YOUTUBE_")
    )
    youtube_midpoint = len(youtube_groups) // 2
    split_by_group.update({group: "validation" for group in youtube_groups[:youtube_midpoint]})
    split_by_group.update({group: "test" for group in youtube_groups[youtube_midpoint:]})
    manifest["split"] = manifest["split_group_id"].map(split_by_group).fillna("train")
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