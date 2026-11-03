# What: This file implements the config module in backend/Datasets/src/phase1/config.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT
METADATA_ROOT = PROJECT_ROOT / "metadata"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
CACHE_ROOT = PROJECT_ROOT / "phase1_cache"

FRAME_SIZE = 224
FRAMES_PER_VIDEO = 8
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-4