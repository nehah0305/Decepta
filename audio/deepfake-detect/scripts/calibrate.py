# What: This file implements the calibrate module in audio/deepfake-detect/scripts/calibrate.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

"""Stage 6 entry point: temperature calibration + three-way decision
policy selection, using the frozen Stage 5 fusion checkpoint.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.calibrate_and_select_policy import run_calibration_and_policy
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="calibration")
def main(cfg: DictConfig) -> None:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: cfg.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    run_calibration_and_policy(cfg)


if __name__ == "__main__":
    main()
