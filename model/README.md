# Decepta — Multimodal Deepfake Detection Preprocessing Pipeline

This repository implements the end-to-end preprocessing pipeline for a multimodal deepfake detection system.

The system is structured in three modular, reproducible stages:
1. **Multimodal Modality Separation**: Ingests raw videos, separating video frames (lossless PNG) and audio (PCM 16-bit WAV) with 1:1 temporal alignment.
2. **Video Preprocessing (Face & Mouth ROI)**: Ingests sampled PNG frames, detects faces & 5 facial landmarks via MTCNN, applies canonical 2D similarity alignment, extracts temporal mouth sequences with coordinate smoothing, and saves `landmarks.json` and `metadata.json`.
3. **Audio Preprocessing (Waveform & Mel-Spectrogram)**: Ingests extracted audio, converts to mono, resamples to exactly 16 kHz, standardizes to 4.0 seconds (64,000 samples), computes log-power Mel-spectrograms (.npy), and saves `audio_metadata.json`.

---

## Architecture Overview

```
                      [Input Video / Media]
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
       [Video Frame Extractor]          [Audio Extractor]
                │                               │
                ▼                               ▼
          input/frames/                   output/audio/
       (Sampled PNG Frames)                 (Raw WAV)
                │                               │
                ▼                               ▼
     [MTCNN Face Detection]             [Audio Preprocessor]
                │                               │
                ▼                               ├─────────────────────────────┐
        [Face Alignment]                        ▼                             ▼
                │                          audio_16k_4s.wav                mel.npy
        ┌───────┴────────┐             (16kHz Mono 4s / 64k samples)   (128x251 Mel Matrix)
        ▼                ▼                      │                             │
  aligned_faces/    mouth_rois/                 └──────────────┬──────────────┘
  (Spatial Branch) (Temporal Branch)                           ▼
        │                │                            audio_metadata.json
        └───────┬────────┘
                ▼
          metadata.json
```

---

## Directory Structure

```text
model/
├── preprocessing/
│   ├── __init__.py             # Package exports
│   ├── ffmpeg_utils.py         # FFmpeg auto-discovery, probing, execution
│   ├── video_processor.py      # Lossless PNG frame extraction & timestamping
│   ├── audio_processor.py      # Independent raw WAV extraction from video
│   ├── pipeline.py             # Multimodal separation orchestrator
│   ├── face_detector.py        # MTCNN face & 5-point landmark detection
│   ├── face_aligner.py         # Canonical 2D similarity face alignment
│   ├── mouth_extractor.py      # Mouth ROI extraction & temporal smoothing
│   ├── dataset_preprocessor.py # Video face dataset orchestrator
│   └── audio_preprocessor.py   # Audio 16 kHz, 4s, Mel-Spectrogram module
├── tests/
│   ├── __init__.py
│   ├── generate_test_media.py    # Synthetic video/audio test generator
│   ├── test_preprocessing.py     # Unit tests for video/audio separation
│   ├── test_face_preprocessing.py # Unit tests for face/mouth pipeline
│   └── test_audio_stage.py       # Unit tests for audio standardization & mel
├── main.py                     # CLI for Video & Audio Modality Separation
├── preprocess_faces_cli.py     # CLI for Video Face & Mouth ROI Preprocessing
├── preprocess_audio_cli.py     # CLI for Audio Preprocessing (16 kHz 4s + Mel)
└── README.md
```

---

## Output Dataset Schema

Processing a video sample produces the complete canonical dataset directory:

```text
processed_dataset/
└── video_001/
    ├── frames/                 # Chronological input frames (frame_01.png, ...)
    ├── aligned_faces/          # Canonical aligned faces (224x224 PNG) (face_01.png, ...)
    ├── mouth_rois/             # Temporal mouth sequence (96x96 PNG) (mouth_01.png, ...)
    ├── landmarks/
    │   └── landmarks.json      # Cached 5-point coordinates & detection confidences
    ├── metadata.json           # Complete video & face forensic metadata
    ├── audio/
    │   └── audio_16k_4s.wav    # Standardized 16 kHz Mono 4-second WAV (64,000 samples)
    ├── spectrogram/
    │   └── mel.npy             # 128x251 numerical log-Mel spectrogram matrix
    └── audio_metadata.json     # Complete audio stream & mel configuration metadata
```

---

### `audio_metadata.json` Schema

```json
{
  "video_id": "video_001",
  "input_audio": "audio.wav",
  "original_sample_rate": 44100,
  "original_channels": 2,
  "processed_sample_rate": 16000,
  "processed_channels": 1,
  "original_duration": 8.52,
  "processed_duration": 4.0,
  "num_samples": 64000,
  "mel_parameters": {
    "n_mels": 128,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024
  },
  "processing_status": "success"
}
```

---

## Usage

### 1. Audio Preprocessing Stage (CLI)

```powershell
python preprocess_audio_cli.py --input-audio output/audio/audio.wav --output-dir processed_dataset/video_001 --video-id video_001
```

#### CLI Options:
| Option | Short | Default | Description |
|---|---|---|---|
| `--input-audio` | `-i` | *(required)* | Path to the extracted audio file |
| `--output-dir` | `-o` | `processed_dataset/video_001` | Destination directory |
| `--video-id` | | `video_001` | Identifier for the sample/video |
| `--sample-rate` | | `16000` | Target sampling rate in Hz |
| `--duration` | | `4.0` | Target duration in seconds |
| `--n-mels` | | `128` | Number of Mel filter bands |
| `--n-fft` | | `1024` | FFT window size |
| `--hop-length` | | `256` | STFT hop length |
| `--win-length` | | `1024` | STFT window length |

---

### 2. Audio Preprocessing (Python API)

```python
from preprocessing import preprocess_audio_file, AudioPreprocessorConfig

result = preprocess_audio_file(
    input_audio_path="output/audio/audio.wav",
    output_dir="processed_dataset/video_001",
    video_id="video_001",
    config=AudioPreprocessorConfig(
        target_sample_rate=16000,
        target_duration_seconds=4.0,
        n_mels=128
    )
)

print(f"Standardized WAV saved: {result.waveform_path} (64k samples)")
print(f"Mel-Spectrogram saved: {result.spectrogram_path} (Shape: {result.mel_shape})")
```

---

### 3. Video Face & Mouth Preprocessing (CLI)

```powershell
python preprocess_faces_cli.py --input-dir input/frames --output-dir processed_dataset/video_001 --video-id video_001 --label real
```

---

### 4. Video & Audio Extraction from Raw Video (CLI)

```powershell
python main.py --input path/to/video.mp4 --output-dir output --fps 25
```

---

## Running Automated Tests

Run the complete test suite (23 unit & integration tests):

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
