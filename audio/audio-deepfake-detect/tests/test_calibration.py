# What: This file implements the test calibration module in audio/audio-deepfake-detect/tests/test_calibration.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

from pathlib import Path

import numpy as np
import torch

from src.evaluation.calibration import (
    apply_temperature,
    compute_reliability_diagram,
    fit_temperature,
    get_acoustic_logits,
    plot_reliability_diagram,
)


def test_apply_temperature_matches_manual_sigmoid():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    logits = np.array([-2.0, 0.0, 2.0])
    temperature = 2.0
    expected = 1.0 - 1.0 / (1.0 + np.exp(-logits / temperature))
    assert np.allclose(apply_temperature(logits, temperature), expected)


def test_fit_temperature_corrects_overconfident_logits():
    # Genuine overconfidence (not just noise): the model claims near-100%
    # confidence via large-magnitude logits, but is only actually correct
    # 75% of the time -- the classic case temperature scaling fixes.
    # (Perfectly-separable logits, by contrast, are already NLL-optimal at
    # low T; softening them with T>1 would only increase loss.)
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=1000)
    is_correct = rng.random(1000) < 0.75
    predicted_label = np.where(is_correct, labels, 1 - labels)
    overconfident_logits = np.where(predicted_label == 1, 8.0, -8.0) + rng.normal(0, 0.1, size=1000)

    temperature = fit_temperature(overconfident_logits, labels)

    assert temperature > 3.0  # should scale confidence down substantially

    criterion = torch.nn.BCEWithLogitsLoss()
    logits_t = torch.tensor(overconfident_logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    nll_before = criterion(logits_t, labels_t).item()
    nll_after = criterion(logits_t / temperature, labels_t).item()
    assert nll_after < nll_before


def test_reliability_diagram_near_zero_ece_when_well_calibrated():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    rng = np.random.default_rng(1)
    probs = rng.uniform(0, 1, size=2000)
    labels = (rng.uniform(0, 1, size=2000) < probs).astype(int)

    diagram = compute_reliability_diagram(probs, labels, n_bins=10)
    assert diagram["ece"] < 0.1


def test_reliability_diagram_high_ece_when_miscalibrated():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    rng = np.random.default_rng(2)
    probs = np.full(1000, 0.9)  # always confident...
    labels = rng.integers(0, 2, size=1000)  # ...but actually 50/50

    diagram = compute_reliability_diagram(probs, labels, n_bins=10)
    assert diagram["ece"] > 0.3


def test_plot_reliability_diagram_writes_file(tmp_path: Path):
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: tmp_path.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    labels = np.array([1] * 50 + [0] * 50)
    before = compute_reliability_diagram(np.array([0.9] * 50 + [0.1] * 50), labels)
    after = compute_reliability_diagram(np.array([0.6] * 50 + [0.4] * 50), labels)

    output_path = tmp_path / "reliability.png"
    plot_reliability_diagram(before, after, str(output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 0


class _DummyAcousticModel(torch.nn.Module):
    """
    What: This class provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Constructor arguments and class attributes supplied by callers.
    Output: Instances exposing the class attributes and methods.
    """
    def forward(self, features, padding_mask=None):
        # Just needs to be a deterministic function of the input.
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self, features, padding_mask.
        Output: The returned value, or side effects performed by the function.
        """
        return features.mean(dim=(1, 2)) * 10


class _DummyAcousticDataset(torch.utils.data.Dataset):
    """
    What: This class provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Constructor arguments and class attributes supplied by callers.
    Output: Instances exposing the class attributes and methods.
    """
    def __init__(self, labels):
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self, labels.
        Output: Side effects performed by the function; no value is explicitly returned.
        """
        self.labels = labels

    def __len__(self):
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self.
        Output: The returned value, or side effects performed by the function.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self, idx.
        Output: The returned value, or side effects performed by the function.
        """
        value = 1.0 if self.labels[idx] == 1 else 0.0
        features = torch.full((3, 5), value)
        mask = torch.zeros(3, dtype=torch.bool)
        return features, mask, torch.tensor(float(self.labels[idx]))


def test_get_acoustic_logits_preserves_order_and_labels():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    labels = [0, 1, 0, 1, 1]
    dataset = _DummyAcousticDataset(labels)
    model = _DummyAcousticModel()

    logits, returned_labels = get_acoustic_logits(
        model, dataset, torch.device("cpu"), batch_size=2
    )

    assert len(logits) == 5
    assert list(returned_labels) == labels
    # Order preserved: label 1 -> all-ones features -> logit 10, label 0 -> 0.
    assert list(logits) == [0.0, 10.0, 0.0, 10.0, 10.0]
