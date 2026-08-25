from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = PROJECT_ROOT
RAW_DATA_DIR = DATASETS_DIR / "raw"
METADATA_DIR = DATASETS_DIR / "metadata"

REPORTS_DIR = PROJECT_ROOT / "reports" / "phase0"


# Create required directories
METADATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MANIFEST FILES
# ============================================================

MASTER_MANIFEST = METADATA_DIR / "master.csv"

TRAIN_MANIFEST = METADATA_DIR / "train.csv"

VALIDATION_MANIFEST = METADATA_DIR / "validation.csv"

TEST_MANIFEST = METADATA_DIR / "test.csv"


# ============================================================
# DATASET SPLIT RATIOS
# ============================================================

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42


# ============================================================
# LABELS
# ============================================================

REAL_LABEL = 0

FAKE_LABEL = 1


LABEL_MAP = {
    "real": REAL_LABEL,
    "original": REAL_LABEL,
    "fake": FAKE_LABEL,
}


# ============================================================
# VALID SPLITS
# ============================================================

VALID_SPLITS = {
    "train",
    "validation",
    "test",
}


# ============================================================
# REQUIRED MANIFEST COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "sample_id",
    "dataset",
    "video_path",
    "subject_id",
    "source_id",
    "split_group_id",
    "manipulation",
    "generator",
    "label",
]