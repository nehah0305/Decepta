"""
Master Configuration Module for Multimodal Deepfake Detection System.

Configures:
1. Visual Branch (High-coverage frame sampling ~70%, custom Spatial & Frequency CNNs, Gated Fusion, Temporal Transformer)
2. Audio Authenticity Branch (16 kHz mono, 4s overlapping windows, Log-Mel, custom 2D Audio CNN, Transformer self-attention)
3. Audio-Visual Synchronization Branch (112x112 mouth ROI encoder, temporal alignment, cosine similarity, learnable sync module, InfoNCE loss)
4. Adaptive Multimodal Fusion (Learned sample-specific modality attention weights alpha_v, alpha_a, alpha_s, missing modality masks)
5. Training, Losses, Evaluation, and File Paths
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import torch


@dataclass
class VisualPipelineConfig:
    """Master configuration for the multimodal deepfake detection pipeline."""

    # -------------------------------------------------------------------------
    # 1. Visual Branch: High-Coverage Frame Sampling & Architecture
    # -------------------------------------------------------------------------
    FRAME_COVERAGE_RATIO: float = 0.70  # Analyze ~70% of usable video timeline
    MIN_FRAMES: int = 32                # Minimum candidate frames to sample
    MAX_FRAMES: Optional[int] = 64      # Cap on sampled keyframes per video for memory safety
    FRAME_BATCH_SIZE: int = 16          # GPU/RAM chunk size for CNN processing (does NOT drop frames)
    ALLOW_RANDOM_SAMPLING: bool = False # Deterministic equidistant sampling by default

    # Frame Quality Filtering
    QUALITY_MIN_BRIGHTNESS: float = 5.0    # Lower bound for average pixel intensity
    QUALITY_MAX_BRIGHTNESS: float = 250.0  # Upper bound for average pixel intensity
    QUALITY_BLUR_THRESHOLD: float = 15.0   # Laplacian variance threshold (non-aggressive)
    NON_AGGRESSIVE_FILTERING: bool = True  # Keep slightly degraded frames as they carry artifacts

    # MTCNN Face Detection & Canonical 224x224 Alignment
    FACE_SIZE: int = 224                   # Canonical aligned face resolution (224x224)
    MTCNN_MIN_FACE_SIZE: int = 40          # Minimum face size in pixels for detection
    MTCNN_THRESHOLDS: Tuple[float, float, float] = (0.6, 0.7, 0.7)
    PRIMARY_FACE_SELECTION: str = "tracking" # "tracking" (continuity) or "largest_confidence"

    # Visual Neural Network Dimensions
    SPATIAL_FEATURE_DIM: int = 256         # Custom Spatial CNN output dimension
    FREQUENCY_FEATURE_DIM: int = 256       # Custom Frequency CNN output dimension
    FUSION_HIDDEN_DIM: int = 128           # Gated fusion MLP hidden dimension
    FUSED_FEATURE_DIM: int = 256           # Fused frame representation dimension
    TRANSFORMER_DIM: int = 768             # Temporal Transformer model dimension (d_model)
    TRANSFORMER_HEADS: int = 8             # Number of self-attention heads
    TRANSFORMER_LAYERS: int = 2            # Number of TransformerEncoder layers
    TRANSFORMER_FEEDFORWARD_DIM: int = 2048 # Feedforward dimension in transformer
    TRANSFORMER_DROPOUT: float = 0.1       # Transformer dropout rate

    # -------------------------------------------------------------------------
    # 2. Audio Authenticity Branch (Branch A)
    # -------------------------------------------------------------------------
    AUDIO_SAMPLE_RATE: int = 16000         # Standardized audio sample rate (16 kHz)
    AUDIO_WINDOW_SECONDS: float = 4.0      # 4.0 seconds (64,000 samples per window)
    AUDIO_HOP_SECONDS: float = 2.0         # 2.0 seconds hop for overlapping windows
    AUDIO_N_MELS: int = 128                # Mel filter bank frequency channels
    AUDIO_N_FFT: int = 1024                # STFT FFT size
    AUDIO_HOP_LENGTH: int = 256            # STFT hop length
    AUDIO_WIN_LENGTH: int = 1024           # STFT window length
    AUDIO_F_MIN: float = 0.0               # Minimum frequency in Hz
    AUDIO_F_MAX: float = 8000.0            # Maximum frequency in Hz
    AUDIO_TOP_DB: float = 80.0             # Dynamic range for dB scale
    AUDIO_CNN_TYPE: str = "2d"             # Custom 2D Spectrogram CNN
    AUDIO_CNN_CHANNELS: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256]
    )
    AUDIO_FEATURE_DIM: int = 768           # Global audio authenticity feature dimension
    AUDIO_TRANSFORMER_HEADS: int = 8       # Self-attention heads in audio branch
    AUDIO_TRANSFORMER_LAYERS: int = 2      # Transformer encoder layers in audio branch
    AUDIO_TRANSFORMER_DROPOUT: float = 0.1 # Dropout in audio transformer

    # -------------------------------------------------------------------------
    # 3. Audio-Visual Synchronization Branch (Branch B)
    # -------------------------------------------------------------------------
    MOUTH_ROI_SIZE: int = 112              # Mouth ROI resolution (112x112)
    MOUTH_EMBEDDING_DIM: int = 256         # Mouth encoder output embedding dimension
    AUDIO_SYNC_EMBEDDING_DIM: int = 256    # Audio sync token projection dimension
    SYNC_FEATURE_DIM: int = 256            # Sync branch output feature dimension (F_sync)
    SYNC_LOSS_TYPE: str = "infonce"        # Canonical InfoNCE loss
    SYNC_TEMPERATURE: float = 0.07         # Temperature tau for InfoNCE
    SYNC_LOSS_WEIGHT: float = 0.5          # lambda_sync multiplier in total loss
    SYNC_SHIFT_OFFSETS: List[float] = field(
        default_factory=lambda: [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]
    )

    # -------------------------------------------------------------------------
    # 4. Adaptive Multimodal Fusion
    # -------------------------------------------------------------------------
    FUSION_DIM: int = 768                  # Common projection space dimension
    FUSION_HIDDEN_DIM_ATTN: int = 256      # Hidden dimension in modality attention network
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
    # Modal options: "full", "visual_only", "audio_only", "sync_only",
    #                "visual_audio", "visual_sync", "audio_sync", "concat_fusion"
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
    SYNC_EVAL_DIR: Path = OUTPUT_DIR / "sync_eval"

    def __post_init__(self):
        """Ensure necessary output directories exist."""
        for d in [
            self.OUTPUT_DIR,
            self.METADATA_DIR,
            self.FEATURES_DIR,
            self.PREDICTIONS_DIR,
            self.CHECKPOINTS_DIR,
            self.SYNC_EVAL_DIR
        ]:
            d.mkdir(parents=True, exist_ok=True)


# Global default configuration instance (aliased for backward compatibility)
DEFAULT_CONFIG = VisualPipelineConfig()
MultimodalPipelineConfig = VisualPipelineConfig
