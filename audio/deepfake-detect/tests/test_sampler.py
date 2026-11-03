# What: This file implements the test sampler module in audio/deepfake-detect/tests/test_sampler.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import torch

from src.preprocessing.sampler import class_balanced_loss_weights, make_weighted_sampler


def test_weighted_sampler_upsamples_minority_class():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    labels = ["real"] * 4 + ["fake"] * 160  # ~40:1 imbalance
    sampler = make_weighted_sampler(labels, enabled=True)

    drawn = [labels[i] for i in sampler]
    real_fraction = drawn.count("real") / len(drawn)

    # Should be pulled well above the raw ~2.4% real fraction.
    assert real_fraction > 0.25


def test_weighted_sampler_disabled_returns_none():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    labels = ["real"] * 4 + ["fake"] * 160
    assert make_weighted_sampler(labels, enabled=False) is None


def test_class_balanced_loss_weights_favor_minority():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    labels = ["real"] * 4 + ["fake"] * 160
    weights = class_balanced_loss_weights(labels, enabled=True)

    weight_real, weight_fake = weights.tolist()
    assert weight_real > weight_fake


def test_class_balanced_loss_weights_disabled_is_uniform():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    labels = ["real"] * 4 + ["fake"] * 160
    weights = class_balanced_loss_weights(labels, enabled=False)
    assert torch.allclose(weights, torch.tensor([1.0, 1.0]))
