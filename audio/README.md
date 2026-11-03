# Decepta — Audio Forensic Detection Subsystems

This directory houses the acoustic deepfake detection engines integrated into the Decepta platform.

## Subdirectories

### 1. `audio-deepfake-detect/` (Active Audio Engine)
- **Model Checkpoint**: `models/acoustic_model.joblib` (286,410 parameters, 1.02 MB)
- **Architecture**: Acoustic Transformer with attention pooling over 68 frame-level acoustic descriptors:
  - 20 Mel-Frequency Cepstral Coefficients (MFCCs)
  - 20 First-order deltas ($\Delta$)
  - 20 Second-order delta-deltas ($\Delta^2$)
  - Fundamental Frequency ($F_0$) pitch & Voicing Confidence
  - 4 Spectral Statistics (Centroid, Bandwidth, Rolloff, Flatness)
  - Short-Time Energy & Zero-Crossing Rate (ZCR)
- **Benchmark Performance**: Validated on LAV-DF with **97.23% ROC-AUC**, **97.25% Precision**, and **91.42% Accuracy**.
- **Explainability**: Integrated **KernelSHAP** feature attribution explaining model verdicts across all 68 descriptors.
- **Service Integration**: Served via `backend/deepfake_detection_service.py` to Decepta's React frontend.

### 2. `deepfake-detect/` (Multimodal Reference Pipeline)
- **Model Checkpoint**: `models/fusion_model.joblib` (4.9M parameters)
- **Architecture**: Cross-Modal Attention Fusion combining spatial visual representations with acoustic descriptors.
- **Benchmark Performance**: Validated on LAV-DF Multimodal with **99.32% ROC-AUC**.

### 3. `ai-voice-detect/` (Research Baseline)
- Experimental speech detector codebase for WavLM-Base+ representations.

## Quick Usage

```powershell
# Start the FastAPI Audio Detection Service (port 8000)
python backend/deepfake_detection_service.py --serve --port 8000

# Run direct CLI inference on an audio file
python backend/deepfake_detection_service.py --file frontend/public/samples/audio/LAVDF_Synthetic_Voice_042.wav
```
