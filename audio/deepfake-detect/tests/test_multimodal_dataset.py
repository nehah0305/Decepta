# What: This file implements the test multimodal dataset module in audio/deepfake-detect/tests/test_multimodal_dataset.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.preprocessing.dataset import MultimodalDataset


def _write_video(path: Path, value: int, num_frames=4, size=(48, 48)) -> bool:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: path, value, num_frames, size.
    Output: The returned value, or side effects performed by the function.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, size)
    if not writer.isOpened():
        return False
    for _ in range(num_frames):
        writer.write(np.full((size[1], size[0], 3), value, dtype=np.uint8))
    writer.release()
    return True


def _write_tone(path: Path, freq: float, duration_s=0.3, sample_rate=16000):
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: path, freq, duration_s, sample_rate.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sample_rate)


def _make_manifest(tmp_path: Path) -> Path:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: tmp_path.
    Output: The returned value, or side effects performed by the function.
    """
    video_path = tmp_path / "clip.mp4"
    audio_path = tmp_path / "clip.wav"
    if not _write_video(video_path, value=100):
        pytest.skip("mp4v codec unavailable in this environment")
    _write_tone(audio_path, freq=220)

    manifest = pd.DataFrame(
        [{"video_path": str(video_path), "audio_path": str(audio_path), "label": "fake",
          "identity_id": "id0"}]
    )
    csv_path = tmp_path / "manifest.csv"
    manifest.to_csv(csv_path, index=False)
    return csv_path


def test_multimodal_dataset_returns_paired_tensors(tmp_path: Path):
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: tmp_path.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    csv_path = _make_manifest(tmp_path)
    dataset = MultimodalDataset(
        csv_path,
        split="val",
        visual_kwargs={"frame_rate": 8, "frame_size": 32, "num_frames": 4},
        acoustic_kwargs={"n_mfcc": 13, "num_frames": 20},
    )

    frames, v_mask, features, a_mask, label = dataset[0]

    assert frames.shape == (4, 3, 32, 32)
    assert v_mask.shape == (4,)
    assert features.shape == (20, 47)
    assert a_mask.shape == (20,)
    assert label.item() == 1.0


def test_multimodal_dataset_length_matches_manifest(tmp_path: Path):
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: tmp_path.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    csv_path = _make_manifest(tmp_path)
    dataset = MultimodalDataset(
        csv_path, split="val",
        visual_kwargs={"frame_rate": 8, "frame_size": 32, "num_frames": 4},
        acoustic_kwargs={"n_mfcc": 13, "num_frames": 20},
    )
    assert len(dataset) == 1
