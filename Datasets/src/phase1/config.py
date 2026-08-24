from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "raw" / "faceforensicspp" / "c23"
METADATA_ROOT = PROJECT_ROOT / "metadata"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"

FRAME_SIZE = 224
FRAMES_PER_VIDEO = 8
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-4