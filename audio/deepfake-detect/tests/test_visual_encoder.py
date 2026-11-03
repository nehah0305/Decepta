# What: This file implements the test visual encoder module in audio/deepfake-detect/tests/test_visual_encoder.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import torch

from src.models.common import masked_mean_pool
from src.models.visual.encoder import VisualClassifier, VisualEncoder


def _make_encoder(embed_dim=32):
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: embed_dim.
    Output: The returned value, or side effects performed by the function.
    """
    return VisualEncoder(
        backbone="efficientnet_b0",
        pretrained=False,
        embed_dim=embed_dim,
        transformer_depth=2,
        transformer_heads=4,
        transformer_ff_dim=64,
        dropout=0.0,
    )


def test_visual_encoder_output_shape():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    encoder = _make_encoder(embed_dim=32)
    frames = torch.rand(2, 5, 3, 64, 64)

    hv = encoder(frames)

    assert hv.shape == (2, 5, 32)


def test_visual_encoder_respects_padding_mask():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    encoder = _make_encoder(embed_dim=32)
    frames = torch.rand(2, 5, 3, 64, 64)
    padding_mask = torch.zeros(2, 5, dtype=torch.bool)
    padding_mask[0, 3:] = True  # last two frames of sample 0 are padding

    hv = encoder(frames, padding_mask=padding_mask)
    assert hv.shape == (2, 5, 32)


def test_masked_mean_pool_ignores_padded_positions():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    x = torch.zeros(1, 4, 3)
    x[0, 0] = 1.0
    x[0, 1] = 3.0
    x[0, 2] = 100.0  # padded, should be excluded
    x[0, 3] = 100.0  # padded, should be excluded
    mask = torch.tensor([[False, False, True, True]])

    pooled = masked_mean_pool(x, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0, 2.0]]))


def test_visual_classifier_output_shape():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    encoder = _make_encoder(embed_dim=32)
    classifier = VisualClassifier(encoder, embed_dim=32)
    frames = torch.rand(3, 5, 3, 64, 64)

    logits = classifier(frames)
    assert logits.shape == (3,)
