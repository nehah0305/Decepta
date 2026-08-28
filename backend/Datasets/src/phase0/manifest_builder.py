from pathlib import Path
from typing import Optional

import pandas as pd


def create_empty_manifest() -> pd.DataFrame:
    """
    Create an empty manifest with the correct schema.
    """

    columns = [
        "sample_id",
        "dataset",
        "video_path",
        "audio_path",
        "subject_id",
        "source_id",
        "manipulation",
        "generator",
        "label",
    ]

    return pd.DataFrame(columns=columns)


def add_sample(
    manifest: pd.DataFrame,
    sample_id: str,
    dataset: str,
    video_path: str,
    subject_id: str,
    source_id: str,
    manipulation: str,
    generator: str,
    label: int,
    audio_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add one sample to the manifest.
    """

    new_row = {
        "sample_id": sample_id,
        "dataset": dataset,
        "video_path": video_path,
        "audio_path": audio_path,
        "subject_id": subject_id,
        "source_id": source_id,
        "manipulation": manipulation,
        "generator": generator,
        "label": label,
    }

    return pd.concat(
        [
            manifest,
            pd.DataFrame([new_row])
        ],
        ignore_index=True,
    )


def save_manifest(
    manifest: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save manifest as CSV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nManifest saved to:\n"
        f"{output_path}"
    )