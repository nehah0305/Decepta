"""
Central Configuration Module for Visual Deepfake Detection System.

Configures:
- High-coverage frame sampling (default 70%, 60-80% target)
- Custom Spatial & Frequency CNNs, Gated Fusion, Temporal Transformer
- Frame quality thresholds, MTCNN settings, training & inference parameters
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import torch


@dataclass
class VisualPipelineConfig:
    """Master configuration for the visual deepfake detection pipeline."""

    # -------------------------------------------------------------------------
    # 1. High-Coverage Frame Sampling
    # -------------------------------------------------------------------------
    FRAME_COVERAGE_RATIO: float = 0.70  # Analyze ~70% of usable video timeline
    MIN_FRAMES: int = 32                # Minimum candidate frames to sample
    MAX_FRAMES: Optional[int] = None    # Optional hard cap on sampled frames (None = no cap)
    FRAME_BATCH_SIZE: int = 32          # GPU/RAM chunk size for CNN processing (does NOT drop frames)
    ALLOW_RANDOM_SAMPLING: bool = False # Deterministic equidistant sampling by default

    # -------------------------------------------------------------------------
    # 2. Frame Quality Filtering
    # -------------------------------------------------------------------------
    QUALITY_MIN_BRIGHTNESS: float = 5.0    # Lower bound for average pixel intensity
    QUALITY_MAX_BRIGHTNESS: float = 250.0  # Upper bound for average pixel intensity
    QUALITY_BLUR_THRESHOLD: float = 15.0   # Laplacian variance threshold (non-aggressive)
    NON_AGGRESSIVE_FILTERING: bool = True  # Keep slightly degraded frames as they carry artifacts

    # -------------------------------------------------------------------------
    # 3. Face Detection & Alignment (MTCNN)
    # -------------------------------------------------------------------------
    FACE_SIZE: int = 224                   # Canonical aligned face resolution (224x224)
    MTCNN_MIN_FACE_SIZE: int = 40          # Minimum face size in pixels for detection
    MTCNN_THRESHOLDS: Tuple[float, float, float] = (0.6, 0.7, 0.7)
    PRIMARY_FACE_SELECTION: str = "tracking" # "tracking" (continuity) or "largest_confidence"

    # -------------------------------------------------------------------------
    # 4. Neural Network Architectural Dimensions
    # -------------------------------------------------------------------------
    SPATIAL_FEATURE_DIM: int = 256         # Custom Spatial CNN output dimension
    FREQUENCY_FEATURE_DIM: int = 256       # Custom Frequency CNN output dimension
    FUSION_HIDDEN_DIM: int = 128           # Gated fusion MLP hidden dimension
    FUSED_FEATURE_DIM: int = 256           # Fused frame representation dimension
    TRANSFORMER_DIM: int = 768             # Temporal Transformer model dimension (d_model)
    TRANSFORMER_HEADS: int = 8             # Number of self-attention heads
    TRANSFORMER_LAYERS: int = 2            # Number of TransformerEncoder layers
    TRANSFORMER_FEEDFORWARD_DIM: int = 2048 # Feedforward dimension in transformer
    TRANSFORMER_DROPOUT: float = 0.1       # Transformer dropout rate
    NUM_CLASSES: int = 1                   # Binary classification (Real=0, Fake=1)

    # -------------------------------------------------------------------------
    # 5. Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE: int = 4                    # Video-level batch size for training
    LEARNING_RATE: float = 1e-4            # Initial learning rate
    WEIGHT_DECAY: float = 1e-4             # L2 regularization factor
    NUM_EPOCHS: int = 20                   # Total training epochs
    GRADIENT_CLIP_VAL: float = 1.0         # Max gradient norm clipping
    USE_AMP: bool = True                   # Mixed precision training where available
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # 6. Ablation Experiment Settings
    # -------------------------------------------------------------------------
    # Options: "full", "spatial_only", "frequency_only", "no_gate", "frame_average"
    MODEL_MODE: str = "full"

    # -------------------------------------------------------------------------
    # 7. File Paths and Output Directories
    # -------------------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).resolve().parent
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    METADATA_DIR: Path = OUTPUT_DIR / "frame_metadata"
    FEATURES_DIR: Path = OUTPUT_DIR / "features"
    PREDICTIONS_DIR: Path = OUTPUT_DIR / "predictions"
    CHECKPOINTS_DIR: Path = OUTPUT_DIR / "checkpoints"

    def __post_init__(self):
        """Ensure necessary output directories exist."""
        for d in [self.OUTPUT_DIR, self.METADATA_DIR, self.FEATURES_DIR, self.PREDICTIONS_DIR, self.CHECKPOINTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# Global default configuration instance
DEFAULT_CONFIG = VisualPipelineConfig()
