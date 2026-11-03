# What: This file implements the eval module in audio/deepfake-detect/scripts/eval.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

"""Evaluation entry point (placeholder). See Stage 8 for the full eval suite."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import get_logger
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: cfg.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    logger = get_logger(f"{cfg.experiment_name}_eval", log_dir=Path(cfg.output_dir))
    logger.info("Evaluation pipeline not implemented yet -- see Stage 8.")


if __name__ == "__main__":
    main()
