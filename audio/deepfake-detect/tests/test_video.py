# What: This file implements the test video module in audio/deepfake-detect/tests/test_video.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preprocessing.video import align_face, detect_face_bbox, preprocess_video, sample_frames


def _write_synthetic_video(path: Path, num_frames: int = 10, fps: int = 10, size=(64, 64)) -> bool:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: path, num_frames, fps, size.
    Output: The returned value, or side effects performed by the function.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        return False

    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), fill_value=i * 20 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return True


def test_sample_frames_from_synthetic_video(tmp_path: Path):
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: tmp_path.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    video_path = tmp_path / "clip.mp4"
    if not _write_synthetic_video(video_path):
        pytest.skip("mp4v codec unavailable in this environment")

    frames = sample_frames(str(video_path), frame_rate=5)
    assert len(frames) > 0
    assert frames[0].shape[-1] == 3  # RGB


def test_sample_frames_missing_file_raises(tmp_path: Path):
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: tmp_path.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    with pytest.raises(FileNotFoundError):
        sample_frames(str(tmp_path / "missing.mp4"), frame_rate=5)


def test_align_face_no_face_falls_back_to_center_crop():
    # Flat color frame -> Haar cascade will find nothing, exercising the
    # center-crop fallback path.
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    frame = np.full((100, 200, 3), 128, dtype=np.uint8)
    aligned = align_face(frame, size=224)
    assert aligned.shape == (224, 224, 3)
    assert detect_face_bbox(frame) is None


def test_preprocess_video_returns_stacked_tensor(tmp_path: Path):
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: tmp_path.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    video_path = tmp_path / "clip.mp4"
    if not _write_synthetic_video(video_path):
        pytest.skip("mp4v codec unavailable in this environment")

    tensor = preprocess_video(str(video_path), frame_rate=5, size=64)
    assert tensor.ndim == 4
    assert tensor.shape[1:] == (64, 64, 3)
