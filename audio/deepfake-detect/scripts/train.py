# What: This file implements the train module in audio/deepfake-detect/scripts/train.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

"""Training entry point. Dispatches to the branch-specific trainer based on
cfg.model.name (set by `model=visual|acoustic|fusion` on the CLI).
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import get_logger
from src.utils.seed import set_seed

TRAINERS = {
    "visual_branch": "src.training.train_visual:train_visual",
    "acoustic_branch": "src.training.train_acoustic:train_acoustic",
    "fusion": "src.training.train_fusion:train_fusion",
}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: cfg.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    logger = get_logger(cfg.experiment_name, log_dir=Path(cfg.output_dir))
    logger.info(f"Resolved config:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.model.name not in TRAINERS:
        logger.info(
            f"No trainer implemented yet for model={cfg.model.name!r} -- see the "
            "build plan for the relevant stage (train_acoustic.py / train_fusion.py)."
        )
        return

    module_path, func_name = TRAINERS[cfg.model.name].split(":")
    import importlib

    train_fn = getattr(importlib.import_module(module_path), func_name)
    train_fn(cfg, logger=logger)


if __name__ == "__main__":
    main()
