"""
Model Architecture Module for Visual Deepfake Detection System.
"""

from .spatial_cnn import SpatialCNN
from .fft_module import FFT2DModule
from .frequency_cnn import FrequencyCNN
from .gated_fusion import GatedFusion
from .temporal_transformer import TemporalTransformer, PositionalEncoding, AttentionPooling
from .visual_model import VisualDeepfakeDetector, VisualModelOutput

__all__ = [
    "SpatialCNN",
    "FFT2DModule",
    "FrequencyCNN",
    "GatedFusion",
    "TemporalTransformer",
    "PositionalEncoding",
    "AttentionPooling",
    "VisualDeepfakeDetector",
    "VisualModelOutput",
]
