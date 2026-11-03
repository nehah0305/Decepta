# What: This file implements the test late fusion module in audio/deepfake-detect/tests/test_late_fusion.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import torch
from torch.utils.data import Dataset

from src.models.fusion.late_fusion import average_probabilities, predict_branch_probabilities


def test_average_probabilities_plain_average():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    visual = [0.9, 0.1, 0.8]
    acoustic = [0.7, 0.3, 0.2]
    fused = average_probabilities(visual, acoustic, visual_weight=0.5)
    assert fused == [0.8, 0.2, 0.5]


def test_average_probabilities_weighted():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    visual = [1.0, 0.0]
    acoustic = [0.0, 1.0]
    fused = average_probabilities(visual, acoustic, visual_weight=0.75)
    assert fused == [0.75, 0.25]


def test_average_probabilities_length_mismatch_raises():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    try:
        average_probabilities([0.1, 0.2], [0.5])
        assert False, "expected AssertionError"
    except AssertionError:
        pass


class _DummyDataset(Dataset):
    """
    What: This class provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Constructor arguments and class attributes supplied by callers.
    Output: Instances exposing the class attributes and methods.
    """
    def __init__(self, values):
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self, values.
        Output: Side effects performed by the function; no value is explicitly returned.
        """
        self.values = values

    def __len__(self):
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self.
        Output: The returned value, or side effects performed by the function.
        """
        return len(self.values)

    def __getitem__(self, idx):
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self, idx.
        Output: The returned value, or side effects performed by the function.
        """
        x = torch.full((3,), self.values[idx])
        mask = torch.zeros(3, dtype=torch.bool)
        label = torch.tensor(0.0)
        return x, mask, label


class _SumModel(torch.nn.Module):
    """
    What: This class provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Constructor arguments and class attributes supplied by callers.
    Output: Instances exposing the class attributes and methods.
    """
    def forward(self, x, padding_mask=None):
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self, x, padding_mask.
        Output: The returned value, or side effects performed by the function.
        """
        return x.sum(dim=-1)


def test_predict_branch_probabilities_matches_manual_sigmoid():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    dataset = _DummyDataset([0.0, 10.0, -10.0])
    model = _SumModel()

    probs = predict_branch_probabilities(model, dataset, torch.device("cpu"), batch_size=2)

    expected = [torch.sigmoid(torch.tensor(v * 3)).item() for v in [0.0, 10.0, -10.0]]
    assert len(probs) == 3
    for got, want in zip(probs, expected):
        assert abs(got - want) < 1e-5
