# What: This file implements the explain module in audio/deepfake-detect/scripts/explain.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

"""Stage 7 entry point: SHAP + Grad-CAM explanations and the attribution
agreement rate, using the frozen Stage 5 fusion checkpoint and the Stage 6
policy.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.run_explanations import run_explanations
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="explain")
def main(cfg: DictConfig) -> None:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: cfg.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    run_explanations(cfg)


if __name__ == "__main__":
    main()
