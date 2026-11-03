# What: This file implements the loader module in audio/audio-deepfake-detect/src/inference/loader.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

"""Loads the exported acoustic artefact (models/acoustic_model.joblib).

The artefact is self-contained -- weights, the architecture config needed to
rebuild the graph, the data config it was trained under, and the test
metrics of the run that produced it -- so nothing here reads the Hydra
configs. That is what lets this package be copied somewhere else and still
run.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import torch
from omegaconf import OmegaConf

from src.models.acoustic.encoder import AcousticClassifier, AcousticEncoder

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "models" / "acoustic_model.joblib")


def acoustic_input_dim(n_mfcc: int) -> int:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: n_mfcc.
    Output: The returned value, or side effects performed by the function.
    """
    return 3 * n_mfcc + 8


def build_acoustic_model(model_cfg) -> AcousticClassifier:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: model_cfg.
    Output: The returned value, or side effects performed by the function.
    """
    encoder = AcousticEncoder(
        input_dim=acoustic_input_dim(model_cfg.n_mfcc),
        embed_dim=model_cfg.embed_dim,
        transformer_depth=model_cfg.transformer.depth,
        transformer_heads=model_cfg.transformer.heads,
        transformer_ff_dim=model_cfg.transformer.ff_dim,
        dropout=model_cfg.transformer.dropout,
    )
    return AcousticClassifier(encoder, embed_dim=model_cfg.embed_dim,
                              pooling=model_cfg.get("pooling", "mean"))


@dataclass
class LoadedAcousticModel:
    """
    What: This class provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Constructor arguments and class attributes supplied by callers.
    Output: Instances exposing the class attributes and methods.
    """
    model: AcousticClassifier
    model_config: dict
    data_config: dict
    metadata: dict

    @property
    def n_mfcc(self) -> int:
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self.
        Output: The returned value, or side effects performed by the function.
        """
        return int(self.model_config["n_mfcc"])

    @property
    def input_dim(self) -> int:
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self.
        Output: The returned value, or side effects performed by the function.
        """
        return int(self.model.encoder.input_norm.num_features)

    @property
    def test_metrics(self) -> Dict[str, float]:
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self.
        Output: The returned value, or side effects performed by the function.
        """
        return self.metadata.get("test_metrics", {})


def load_acoustic_model(
    artefact_path: str = DEFAULT_MODEL_PATH, device: torch.device = None
) -> LoadedAcousticModel:
    """Rebuilds the acoustic classifier from its exported artefact and loads
    the trained weights strictly (a config/weights mismatch fails here, not
    silently at prediction time)."""
    artefact = joblib.load(artefact_path)

    if artefact.get("name") != "acoustic":
        raise ValueError(
            f"{artefact_path} holds a {artefact.get('name')!r} model, not the acoustic "
            "one. This package only ships and loads the audio branch."
        )

    model_cfg = artefact["config"]
    model = build_acoustic_model(OmegaConf.create(model_cfg))
    model.load_state_dict(artefact["weights"], strict=True)
    model.eval()
    if device is not None:
        model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    return LoadedAcousticModel(
        model=model,
        model_config=model_cfg,
        data_config=artefact.get("data_config", {}),
        metadata=artefact.get("metadata", {}),
    )


def audio_settings_from(data_config: dict) -> Dict[str, float]:
    """The framing parameters a clip must be preprocessed with to match how
    the model was trained. Getting these wrong is silent: the model still
    produces a number, it is just a number about differently-shaped input.
    """
    return {
        "sample_rate": int(data_config.get("audio_sample_rate", 16000)),
        "frame_ms": float(data_config.get("audio_frame_ms", 25.0)),
        "hop_ms": float(data_config.get("audio_hop_ms", 20.0)),
        "num_audio_frames": int(data_config.get("num_audio_frames", 400)),
        "pitch_tracker": str(data_config.get("pitch_tracker", "yin")),
    }


def describe(loaded: LoadedAcousticModel) -> List[str]:
    """Human-readable provenance lines, for CLI output and logs."""
    meta, metrics = loaded.metadata, loaded.test_metrics
    lines = [
        f"dataset:        {meta.get('dataset', 'unknown')}",
        f"exported at:    {meta.get('exported_at', 'unknown')}",
        f"parameters:     {meta.get('num_parameters', 'unknown'):,}"
        if isinstance(meta.get("num_parameters"), int)
        else "parameters:     unknown",
        f"input dim:      {loaded.input_dim} named features (n_mfcc={loaded.n_mfcc})",
    ]
    if metrics:
        lines.append(
            "test metrics:   "
            + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
        )
    return lines
