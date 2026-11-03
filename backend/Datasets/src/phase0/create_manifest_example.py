# What: This file implements the create manifest example module in backend/Datasets/src/phase0/create_manifest_example.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

from pathlib import Path

import pandas as pd

from .config import MASTER_MANIFEST


def create_example_manifest():

    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    data = [

        {
            "sample_id": "S001",
            "dataset": "Example",
            "video_path": "example/video_001.mp4",
            "audio_path": "",
            "subject_id": "P001",
            "source_id": "SRC001",
            "manipulation": "Original",
            "generator": "None",
            "label": 0,
        },

        {
            "sample_id": "S002",
            "dataset": "Example",
            "video_path": "example/video_002.mp4",
            "audio_path": "",
            "subject_id": "P002",
            "source_id": "SRC002",
            "manipulation": "DeepFake",
            "generator": "ExampleGenerator",
            "label": 1,
        },

        {
            "sample_id": "S003",
            "dataset": "Example",
            "video_path": "example/video_003.mp4",
            "audio_path": "",
            "subject_id": "P003",
            "source_id": "SRC003",
            "manipulation": "Original",
            "generator": "None",
            "label": 0,
        },

        {
            "sample_id": "S004",
            "dataset": "Example",
            "video_path": "example/video_004.mp4",
            "audio_path": "",
            "subject_id": "P004",
            "source_id": "SRC004",
            "manipulation": "DeepFake",
            "generator": "ExampleGenerator",
            "label": 1,
        },
    ]

    df = pd.DataFrame(data)

    MASTER_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        MASTER_MANIFEST,
        index=False
    )

    print(
        f"Example manifest created at:\n"
        f"{MASTER_MANIFEST}"
    )


if __name__ == "__main__":

    create_example_manifest()