# What: This file implements the test fusion checkpoint init module in audio/deepfake-detect/tests/test_fusion_checkpoint_init.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

from pathlib import Path

import torch

from src.models.fusion.cross_attention import CrossModalFusion, load_standalone_checkpoint_into
from src.training.train_acoustic import build_acoustic_model
from src.training.train_visual import build_visual_model


def _visual_model_cfg():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: The returned value, or side effects performed by the function.
    """
    return {
        "backbone": "efficientnet_b0",
        "pretrained": False,
        "embed_dim": 32,
        "transformer": {"depth": 2, "heads": 4, "ff_dim": 64, "dropout": 0.0},
    }


def _acoustic_model_cfg():
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: The returned value, or side effects performed by the function.
    """
    return {
        "n_mfcc": 13,
        "embed_dim": 32,
        "transformer": {"depth": 2, "heads": 4, "ff_dim": 64, "dropout": 0.0},
    }


def test_visual_checkpoint_loads_into_fusion_submodules(tmp_path: Path):
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: tmp_path.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    from omegaconf import OmegaConf

    standalone = build_visual_model(OmegaConf.create(_visual_model_cfg()))
    checkpoint_path = tmp_path / "visual_best.pt"
    torch.save(standalone.state_dict(), checkpoint_path)

    fusion = CrossModalFusion(
        visual_encoder=build_visual_model(OmegaConf.create(_visual_model_cfg())).encoder,
        acoustic_encoder=build_acoustic_model(OmegaConf.create(_acoustic_model_cfg())).encoder,
        embed_dim=32, cross_attention_heads=4, cross_attention_dropout=0.0, gate_hidden_dim=16,
    )

    load_standalone_checkpoint_into(fusion.visual_encoder, fusion.visual_aux_head, checkpoint_path)

    for (name, standalone_param), (_, fusion_param) in zip(
        standalone.encoder.state_dict().items(), fusion.visual_encoder.state_dict().items()
    ):
        assert torch.allclose(standalone_param, fusion_param), f"mismatch in {name}"
    for (name, standalone_param), (_, fusion_param) in zip(
        standalone.head.state_dict().items(), fusion.visual_aux_head.state_dict().items()
    ):
        assert torch.allclose(standalone_param, fusion_param), f"mismatch in {name}"


def test_acoustic_checkpoint_loads_into_fusion_submodules(tmp_path: Path):
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: tmp_path.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    from omegaconf import OmegaConf

    standalone = build_acoustic_model(OmegaConf.create(_acoustic_model_cfg()))
    checkpoint_path = tmp_path / "acoustic_best.pt"
    torch.save(standalone.state_dict(), checkpoint_path)

    fusion = CrossModalFusion(
        visual_encoder=build_visual_model(OmegaConf.create(_visual_model_cfg())).encoder,
        acoustic_encoder=build_acoustic_model(OmegaConf.create(_acoustic_model_cfg())).encoder,
        embed_dim=32, cross_attention_heads=4, cross_attention_dropout=0.0, gate_hidden_dim=16,
    )

    load_standalone_checkpoint_into(
        fusion.acoustic_encoder, fusion.acoustic_aux_head, checkpoint_path
    )

    for (name, standalone_param), (_, fusion_param) in zip(
        standalone.encoder.state_dict().items(), fusion.acoustic_encoder.state_dict().items()
    ):
        assert torch.allclose(standalone_param, fusion_param), f"mismatch in {name}"
