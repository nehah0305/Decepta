# What: This file implements the evaluate late fusion module in audio/deepfake-detect/scripts/evaluate_late_fusion.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

"""Stage 4 entry point: evaluate the late-fusion (probability-averaging)
baseline using the frozen Stage 2/3 checkpoints.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.late_fusion_eval import evaluate_late_fusion
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="late_fusion")
def main(cfg: DictConfig) -> None:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: cfg.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    evaluate_late_fusion(cfg)


if __name__ == "__main__":
    main()
