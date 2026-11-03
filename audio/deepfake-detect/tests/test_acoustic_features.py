# What: This file implements the test acoustic features module in audio/deepfake-detect/tests/test_acoustic_features.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import numpy as np

from src.models.acoustic.features import extract_acoustic_features


def _tone(freq: float, duration_s: float = 0.5, sample_rate: int = 16000) -> np.ndarray:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: freq, duration_s, sample_rate.
    Output: The returned value, or side effects performed by the function.
    """
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_feature_count_matches_names():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    signal = _tone(220)
    features, names = extract_acoustic_features(signal, sample_rate=16000, n_mfcc=13)

    assert features.shape[1] == len(names)
    assert features.ndim == 2
    assert features.shape[0] > 1


def test_feature_names_ordering_and_uniqueness():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    _, names = extract_acoustic_features(_tone(220), n_mfcc=13)

    assert len(names) == len(set(names))
    assert names[0] == "mfcc_0"
    assert names[13] == "mfcc_delta_0"
    assert names[26] == "mfcc_delta2_0"
    assert names[39:] == [
        "f0",
        "voicing_confidence",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_flatness",
        "zero_crossing_rate",
        "short_time_energy",
    ]


def test_no_nans_in_output():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    features, _ = extract_acoustic_features(_tone(440), n_mfcc=13)
    assert not np.isnan(features).any()


def test_short_signal_is_handled():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    tiny_signal = np.zeros(50, dtype=np.float32)
    features, names = extract_acoustic_features(tiny_signal, n_mfcc=13)
    assert features.shape[1] == len(names)
    assert features.shape[0] >= 1


def test_different_tones_give_different_spectral_centroid():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    low, names = extract_acoustic_features(_tone(110), n_mfcc=13)
    high, _ = extract_acoustic_features(_tone(4000), n_mfcc=13)

    centroid_idx = names.index("spectral_centroid")
    assert high[:, centroid_idx].mean() > low[:, centroid_idx].mean()
