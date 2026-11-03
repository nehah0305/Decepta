# What: This file implements the test augmentations module in audio/deepfake-detect/tests/test_augmentations.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import numpy as np

from src.preprocessing.augmentations import (
    audio_additive_noise,
    audio_codec_simulation,
    gaussian_noise,
    horizontal_flip,
    jpeg_compression,
    random_resized_crop,
)


def _frame(size=64):
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: size.
    Output: The returned value, or side effects performed by the function.
    """
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def test_jpeg_compression_preserves_shape():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    frame = _frame()
    out = jpeg_compression(frame, quality=40)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_gaussian_noise_changes_pixels_but_keeps_shape():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    frame = _frame()
    out = gaussian_noise(frame, std=10.0, rng=np.random.default_rng(1))
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)


def test_horizontal_flip_reverses_columns():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    frame = _frame()
    out = horizontal_flip(frame)
    assert np.array_equal(out, frame[:, ::-1, :])


def test_random_resized_crop_output_size():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    frame = _frame(size=100)
    out = random_resized_crop(frame, output_size=64, rng=np.random.default_rng(2))
    assert out.shape == (64, 64, 3)


def test_audio_additive_noise_preserves_length():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    signal = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
    out = audio_additive_noise(signal, snr_db=15, rng=np.random.default_rng(3))
    assert out.shape == signal.shape
    assert not np.array_equal(out, signal)


def test_audio_codec_simulation_quantizes_within_range():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    signal = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
    out = audio_codec_simulation(signal, bit_depth=4)
    assert out.shape == signal.shape
    assert out.min() >= -1.01 and out.max() <= 1.01
