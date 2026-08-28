# DECEPTA — Multimodal Deepfake Detection Suite & Research Engine

[![React 19](https://img.shields.io/badge/Frontend-React_19_%7C_TypeScript_%7C_Vite-61DAFB?style=for-the-badge&logo=react)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/ML-PyTorch_2.1_%7C_Torchvision-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Tailwind CSS v4](https://img.shields.io/badge/Styling-Tailwind_v4-38BDF8?style=for-the-badge&logo=tailwindcss)](https://tailwindcss.com/)
[![ROC-AUC Champion](https://img.shields.io/badge/Best_ROC--AUC-72.88%25_(ResNet--50_Stage_B)-10B981?style=for-the-badge)](https://github.com/)

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

All visual models were evaluated on 320 balanced FaceForensics++ validation videos under identical evaluation conditions:

| Model Architecture | Input Modality | Fine-Tuning Strategy | Validation ROC-AUC | Status |
| :--- | :--- | :--- | :---: | :---: |
| **ResNet-50 Fine-Tuned (Stage B)** | **Spatial RGB** | **Unfrozen Layer4 & FC** | **72.88%** ⭐ | **Production Champion** |
| **Spatial + Temporal Transformer** | Spatial + Motion | Multi-Head Temporal Attn | **64.54%** | Research Baseline |
| **Spatial + 2D FFT Concat Fusion** | Spatial + Spectral | Late Feature Concat | **61.49%** | Research Baseline |
| **Spatial + FFT + Temporal** | Multimodal | Multi-Branch Fusion | **58.83%** | Research Baseline |
| **2D FFT Log-Magnitude CNN** | Frequency Spectrum | From Scratch | **58.44%** | Research Baseline |
| **ResNet-50 Frozen Backbone** | Spatial RGB | Frozen ImageNet Conv | **56.82%** | Research Baseline |
| **Spatial CNN Scratch Baseline** | Spatial RGB | From Scratch | **52.62%** | Baseline |

---

## 🏗️ Project Architecture & Directory Map

```text
Decepta/
├── frontend/                        # React 19 + TypeScript + Vite Web Application
│   ├── src/
│   │   ├── components/              # Reusable UI & Forensic Components
│   │   │   ├── analysis/            # AnalysisCard, ForensicReasonsCard, ConfidenceScore, AblationTable
│   │   │   ├── layout/              # Sidebar, MobileNav, DashboardLayout
│   │   │   └── ui/                  # Button, Card, Badge, Toggle, Input, Progress, FileUploader
│   │   ├── pages/                   # Application Pages
│   │   │   ├── Landing.tsx          # Command Center Landing Page
│   │   │   ├── Detect.tsx           # Interactive Deepfake Analyzer & Quick Demo
│   │   │   ├── Analysis.tsx         # Detailed Forensic Report View
│   │   │   ├── History.tsx          # Forensic Audit History & Log Explorer
│   │   │   ├── Models.tsx           # Research & Model Architecture Explorer
│   │   │   ├── Settings.tsx         # System Configuration & Preferences
│   │   │   └── User.tsx             # Researcher Profile & Account Management
│   │   ├── hooks/                   # Custom Hooks (useDetection, useAuth, useToast)
│   │   ├── utils/                   # Forensic PDF Report Generator (reportGenerator.ts)
│   │   ├── types/                   # TypeScript Interfaces & Definitions
│   │   ├── data/                    # Mock Data & LocalStorage Keys
│   │   ├── App.tsx                  # Client-Side Routing Configuration
│   │   └── index.css                # Glassmorphic Design System & Tailwind v4 Custom Theme
│   ├── package.json                 # Node.js Dependencies & Build Scripts
│   └── vite.config.ts               # Vite Server & Transpilation Config
│
└── backend/                         # Deepfake Detection Python ML Pipelines
    ├── model/                       # Modality Separation & Preprocessing
    │   ├── preprocessing/           # Video, Audio, Face Alignment & Mel-Spectrogram Modules
    │   │   ├── ffmpeg_utils.py      # FFmpeg Discovery & Media Probing
    │   │   ├── video_processor.py   # Lossless PNG Frame Extraction
    │   │   ├── audio_processor.py   # WAV Audio Track Extraction
    │   │   ├── face_detector.py     # MTCNN Face & 5-Point Landmark Detection
    │   │   ├── face_aligner.py      # Canonical 2D Similarity Alignment (224x224)
    │   │   └── audio_preprocessor.py# Audio Resampling (16 kHz) & Mel-Spectrogram Matrix
    │   ├── training/                # PyTorch Training & Evaluation Scripts
    │   │   ├── train_resnet50.py    # ResNet-50 Fine-Tuning Pipeline (Stage B)
    │   │   ├── train_fft_scratch.py # 2D FFT Spectral CNN Training
    │   │   └── train_spatial_fft_temporal.py # Multi-Branch Fusion Models
    │   └── main.py                  # Modality Separation CLI
    │
    └── Datasets/                    # Dataset Adapters & Manifest Generators
        ├── src/
        │   ├── phase0/              # Dataset Adapters (FF++, Celeb-DF v2) & Leakage Checkers
        │   └── phase1/              # PyTorch Dataset Loaders & Baseline Trainer
        └── metadata/                # Train, Validation & Test Manifest CSV Files
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

---

## 📜 Dataset Integrity & Forensic Audit Note

> **Academic & Experimental Rigor Note**:
> During our dataset inspection of standard FaceForensics++ (FF++) releases, an exhaustive audit established that all 7,000 FF++ video samples contain silent audio tracks (0% audio coverage). Consequently, our active production model relies on the validated **ResNet-50 Fine-Tuned Spatial Pipeline (72.88% ROC-AUC)**. The **FakeAVCeleb** dataset (21,566 videos with aligned audio-visual tracks) has been designated for future multimodal expansion once individual audio branches reach validation stability (>0.60 ROC-AUC).

---

## 🛡️ License & Confidentiality

This project is private research and software property. All rights reserved.
