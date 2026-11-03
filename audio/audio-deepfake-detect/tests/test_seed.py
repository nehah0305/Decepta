# What: This file implements the test seed module in audio/audio-deepfake-detect/tests/test_seed.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import random

import numpy as np
import torch

from src.utils.seed import set_seed


def test_set_seed_reproducible():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    set_seed(123)
    a = (random.random(), np.random.rand(), torch.rand(1).item())

    set_seed(123)
    b = (random.random(), np.random.rand(), torch.rand(1).item())

    assert a == b
