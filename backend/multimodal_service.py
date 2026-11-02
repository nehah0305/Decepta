"""
Decepta Multimodal Deepfake Detection API Service
Combines:
1. Visual Spatial Deepfake Detector (ResNet-50 Stage B + 2D Canonical Landmark Alignment + Robust Quantile Aggregation)
2. Acoustic Transformer Deepfake Detector (68 Acoustic Descriptors + KernelSHAP)
3. Multimodal Decision Engine
"""
import sys
from pathlib import Path

# Ensure backend directory is in path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from audio_service import app, main, run_video_detection, run_audio_detection

if __name__ == "__main__":
    main()

