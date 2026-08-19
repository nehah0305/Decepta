from pathlib import Path
from typing import List

from tqdm import tqdm


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


def find_video_files(dataset_directory: Path) -> List[Path]:
    """
    Recursively find all video files in a dataset directory.
    """

    if not dataset_directory.exists():

        raise FileNotFoundError(
            f"Dataset directory does not exist:\n"
            f"{dataset_directory}"
        )

    videos = []

    print(
        f"\nScanning dataset:\n"
        f"{dataset_directory}"
    )

    for path in tqdm(
        dataset_directory.rglob("*"),
        desc="Scanning files"
    ):

        if path.is_file():

            if path.suffix.lower() in VIDEO_EXTENSIONS:

                videos.append(path)

    print(
        f"\nFound {len(videos)} video files."
    )

    return videos


def check_duplicate_paths(video_paths: List[Path]) -> List[Path]:
    """
    Find duplicate file paths.
    """

    seen = set()

    duplicates = []

    for path in video_paths:

        resolved = path.resolve()

        if resolved in seen:

            duplicates.append(path)

        else:

            seen.add(resolved)

    return duplicates