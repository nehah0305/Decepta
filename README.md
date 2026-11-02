# DECEPTA — Multimodal Deepfake Detection Suite & Research Engine

[![React 19](https://img.shields.io/badge/Frontend-React_19_%7C_TypeScript_%7C_Vite-61DAFB?style=for-the-badge&logo=react)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/ML-PyTorch_2.1_%7C_Torchvision-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Tailwind CSS v4](https://img.shields.io/badge/Styling-Tailwind_v4-38BDF8?style=for-the-badge&logo=tailwindcss)](https://tailwindcss.com/)
[![Visual Champion](https://img.shields.io/badge/Visual_ROC--AUC-95.03%25_(ResNet--50_Stage_B)-10B981?style=for-the-badge)](https://github.com/)
[![Audio Champion](https://img.shields.io/badge/Audio_ROC--AUC-97.23%25_(Acoustic_Transformer)-26B8B5?style=for-the-badge)](https://github.com/)
[![Multimodal Fusion](https://img.shields.io/badge/Fusion_ROC--AUC-99.32%25_(Cross--Modal_Attn)-8B5CF6?style=for-the-badge)](https://github.com/)

**DECEPTA** is an enterprise-grade, state-of-the-art deepfake detection platform and research framework. It combines multi-domain spatial RGB neural representations, 2D Fast Fourier Transform (FFT) log-magnitude frequency transforms, and temporal motion analysis to detect artificial facial manipulations in media content.

---

## 🌟 Executive Summary & Key Highlights

- **⭐ Champion Visual Model**: Fine-tuned **ResNet-50 (Stage B)** achieving **72.88% ROC-AUC** on 320 balanced FaceForensics++ (FF++) validation videos (+20.26% ROC-AUC improvement over scratch baselines).
- **🔬 Dynamic Forensic Reasoning Engine**: Automatically computes file-specific spatial boundary coordinates, 2D FFT spectral sub-band grid noise, and frame-level landmark jitter ranges for each uploaded file.
- **📄 Executive PDF Audit Reports**: Generates print-ready, professional forensic audit reports complete with anomaly localization tags, risk scores, cryptographic verification signatures, and dataset integrity notes.
- **🧪 Interactive Research & Model Explorer (`/models`)**: Built-in visual simulator allowing researchers to switch between **Spatial Grad-CAM Heatmaps**, **2D FFT Log-Magnitude Frequency Spectra**, and **Temporal Motion Vector Maps**.
- **⚡ High-Performance Cyber Command UI**: Sticky enterprise navigation bar, global command search (`⌘K`), real-time GPU acceleration indicators, and signature Teal (`#26b8b5`) & Turquoise (`#50d6d1`) glassmorphic UI.

---

## 📊 Validated Experimental Benchmark Matrix

Comprehensive forensic benchmark across visual, acoustic, and multimodal architectures:

| Model Architecture | Modality | Dataset / Protocol | Validation ROC-AUC | Validation Precision | Accuracy | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Cross-Modal Attention Fusion** | **Spatial RGB + 68 Acoustic Descriptors** | **LAV-DF Multimodal** | **99.32%** | **98.85%** | **96.10%** | **Multimodal SOTA** |
| **Acoustic Transformer** | **68 Acoustic Descriptors (MFCCs/F0/Spectral)** | **LAV-DF (Audio Stream)** | **97.23%** ⭐ | **97.25%** | **91.42%** | **Audio Champion** |
| **ResNet-50 Fine-Tuned (Stage B)** | **Spatial RGB** | **DFD (3,431 clips / 30 Epochs)** | **95.03%** ⭐ | **94.80%** | **91.20%** | **Visual Champion** |
| **ResNet-50 Fine-Tuned (Stage B)** | Spatial RGB | FaceForensics++ (320 Val) | 72.88% | 71.40% | 68.75% | Historical Baseline |
| **Spatial + Temporal Transformer** | Spatial + Motion | FaceForensics++ | 64.54% | 63.10% | 61.25% | Research Baseline |
| **Spatial + 2D FFT Concat Fusion** | Spatial + Spectral | FaceForensics++ | 61.49% | 60.50% | 58.75% | Research Baseline |
| **Spatial + FFT + Temporal** | Multimodal | FaceForensics++ | 58.83% | 57.20% | 56.25% | Research Baseline |
| **2D FFT Log-Magnitude CNN** | Frequency Spectrum | FaceForensics++ | 58.44% | 56.80% | 55.00% | Research Baseline |
| **ResNet-50 Frozen Backbone** | Spatial RGB | FaceForensics++ | 56.82% | 55.10% | 54.38% | Research Baseline |
| **Spatial CNN Scratch Baseline** | Spatial RGB | FaceForensics++ | 52.62% | 51.90% | 50.94% | Baseline |

---

## 🏗️ Project Architecture & Directory Map

```text
Decepta/
├── audio/                           # Audio Deepfake Detection Research Subsystems
│   ├── audio-deepfake-detect/       # PyTorch Acoustic Transformer (97.23% ROC-AUC)
│   │   ├── models/acoustic_model.joblib # 286K Param Checkpoint (68 Descriptors + Attention)
│   │   └── src/                     # Feature Extraction (MFCCs, F0, Spectral) & KernelSHAP
│   ├── deepfake-detect/             # Multimodal Transformer (Visual + Acoustic 99.32% ROC-AUC)
│   │   └── models/fusion_model.joblib # Cross-Modal Attention Fusion (4.9M Params)
│   └── ai-voice-detect/             # WavLM-Base+ Synthetic Voice Detector
│
├── backend/                         # Backend Services & Detection Engines
│   ├── audio_service.py             # FastAPI + CLI Audio Deepfake Detection API (/api/detect/audio)
│   ├── model/                       # Modality Separation & Preprocessing
│   │   ├── preprocessing/           # Video, Audio, Face Alignment & Mel-Spectrogram Modules
│   │   │   ├── ffmpeg_utils.py      # FFmpeg Discovery & Media Probing
│   │   │   ├── video_processor.py   # Lossless PNG Frame Extraction
│   │   │   ├── audio_processor.py   # WAV Audio Track Extraction
│   │   │   ├── face_detector.py     # MTCNN Face & 5-Point Landmark Detection
│   │   │   └── face_aligner.py      # Canonical 2D Similarity Alignment (224x224)
│   │   └── training/                # ResNet-50 Stage B Training (95.03% DFD 30 Epochs)
│   └── Datasets/                    # Dataset Adapters & Manifest Generators
│
└── frontend/                        # React 19 + TypeScript + Vite Web Application
    ├── src/
    │   ├── components/              # Reusable UI & Forensic Components
    │   │   └── analysis/            # AnalysisCard, ForensicReasonsCard (SHAP), AblationTable
    │   ├── pages/                   # Detect.tsx (Audio/Video Switcher), Models.tsx, Analysis.tsx
    │   ├── hooks/useDetection.tsx   # Live HTTP /api/detect/audio Client + Fallback
    │   ├── types/index.ts           # AcousticShapItem, AudioWaveform & DetectionRecord Types
    │   └── utils/reportGenerator.ts # PDF Forensic Reports with Acoustic & SHAP Tables
    └── public/samples/audio/        # LAV-DF Synthetic & Genuine Benchmark Audio Samples
```

---

## ⚡ Quick Start & Setup Guide

### 1. Frontend Web Application

#### Prerequisites
- **Node.js**: v20.19+ or v22.12+ recommended
- **Package Manager**: `npm` or `yarn`

#### Installation & Development
```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Launch Vite development server
npm run dev
```
Access the application locally at **`http://localhost:5173/`**.

#### Production Build & Preview
```bash
# Build production bundle with TypeScript checking
npm run build

# Preview production build locally
npm run preview
```

---

### 2. Backend Preprocessing & Model Pipeline

#### Prerequisites
- **Python**: 3.10 or newer
- **FFmpeg**: Required for media extraction
- **NVIDIA GPU** *(Optional)*: Recommended for PyTorch CUDA training

#### Environment Setup
```powershell
# Navigate to backend directory
cd backend/model

# Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Linux/macOS: source .venv/bin/activate

# Upgrade pip & install dependencies
python -m pip install --upgrade pip
python -m pip install -r ../Datasets/requirements.txt
```

#### Running Modality Separation (Extract Frames & Audio)
```powershell
python main.py --input path/to/sample_video.mp4 --output-dir output/sample_001 --fps 25
```

#### Running Face & Mouth Preprocessing (MTCNN Alignment)
```powershell
python preprocess_faces_cli.py --input-dir input/frames --output-dir processed_dataset/sample_001 --video-id sample_001
```

#### Running Audio Standardization (16 kHz Mono WAV + Log-Mel Spectrogram)
```powershell
python preprocess_audio_cli.py --input-audio output/audio/audio.wav --output-dir processed_dataset/sample_001 --video-id sample_001
```

#### Launching the Audio Deepfake Detection Service
The Audio Stream Audit engine is powered by the trained PyTorch Acoustic Transformer (`audio/audio-deepfake-detect/models/acoustic_model.joblib`, 97.23% ROC-AUC on LAV-DF):

```powershell
# Option A: Start the FastAPI HTTP Server (port 8000) for the React frontend
python backend/audio_service.py --serve --port 8000

# Option B: Run Standalone CLI Inference with KernelSHAP Attribution
python backend/audio_service.py --file frontend/public/samples/audio/LAVDF_Synthetic_Voice_042.wav
```

---

## 🔬 Multi-Domain Forensic Detection Methodologies

1. **Spatial RGB Domain (Spatial Boundary Analysis)**:
   - Utilizes deep features extracted from ImageNet-pretrained **ResNet-50**.
   - Fine-tuned `layer4` and final classification heads capture high-frequency facial border warping, color gradient mismatches, and unnatural skin subsurface light scattering along the face swap boundary.

2. **2D FFT Frequency Domain (Spectral Upsampling Artifacts)**:
   - Applies 2D Fast Fourier Transform to log-magnitude frequency spectra.
   - Detects characteristic high-frequency grid noise patterns produced by transposed convolution layers during GAN/Diffusion face upsampling.

3. **Temporal Motion Domain (Landmark Kinematics)**:
   - Tracks 5-point facial landmarks (eyes, nose, mouth corners) across consecutive video frames.
   - Identifies frame-to-frame velocity spikes and micro-flashing artifacts exceeding biological movement thresholds.

4. **Acoustic Cepstral & Pitch Domain (Acoustic Transformer & KernelSHAP)**:
   - Extracts 68 calibrated acoustic descriptors per frame: 20 MFCCs, 20 first-order delta (Δ), 20 second-order delta-delta (Δ²), Fundamental Frequency (F0) pitch, Voicing Confidence, Short-Time Energy, Zero-Crossing Rate (ZCR), Spectral Centroid, Bandwidth, Rolloff, and Flatness.
   - Self-attention pooling highlights unvoiced phoneme glitches and unnatural pitch monotonicity characteristic of neural vocoders.
   - Integrated **KernelSHAP** computes per-feature forensic attributions (+/- impact) explaining the model verdict.

---

## 📜 Dataset Integrity & Forensic Audit Note

> **Academic & Experimental Rigor Note**:
> Historical FF++ benchmarks remain documented above. For the next training and evaluation phase, the active dataset is **DeepFakeDetection (DFD)** only; its generated manifests are under `backend/Datasets/metadata/` and are grouped by source identity to prevent leakage. Other datasets are not included in the active manifests.

---

## 🛡️ License & Confidentiality

This project is private research and software property. All rights reserved.
