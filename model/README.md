# Decepta — Multimodal Deepfake Detection Preprocessing Pipeline

This repository implements the end-to-end preprocessing pipeline for a multimodal deepfake detection system.

The system is structured in two complementary stages:
1. **Multimodal Modality Separation**: Ingests raw videos, separating video frames (lossless PNG) and audio (PCM 16-bit WAV) with 1:1 temporal alignment.
2. **Video Preprocessing (Face & Mouth ROI)**: Ingests sampled PNG frames, detects faces & 5 facial landmarks via MTCNN, applies canonical 2D similarity alignment, extracts temporal mouth sequences with coordinate smoothing, and produces a structured dataset with forensic metadata.

---

## Architecture

```
                      [Input Video]
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      [Video Frame Extractor]    [Audio Extractor]
              │                           │
              ▼                           ▼
        input/frames/             output/audio/audio.wav
     (Sampled PNG Frames)
              │
              ▼
   [MTCNN Face Detection] ──► [landmarks/landmarks.json]
              │
              ▼
      [Face Alignment] (2D Affine / Similarity Transform)
              │
              ├─────────────────────────────┐
              ▼                             ▼
    aligned_faces/face_01.png      [Mouth Localization & EMA Smoothing]
    (Spatial/Frequency Branch)              │
                                            ▼
                                  mouth_rois/mouth_01.png
                                  (Temporal/Lip-Sync Branch)
              │                             │
              └──────────────┬──────────────┘
                             ▼
                       metadata.json
```

---

## Directory Structure

```text
model/
├── preprocessing/
│   ├── __init__.py           # Package exports
│   ├── ffmpeg_utils.py       # FFmpeg auto-discovery, probing, execution
│   ├── video_processor.py    # Lossless PNG frame extraction & timestamping
│   ├── audio_processor.py    # Independent WAV audio extraction
│   ├── pipeline.py           # Multimodal pipeline orchestrator
│   ├── face_detector.py      # MTCNN face & 5-point landmark detection
│   ├── face_aligner.py       # Canonical 2D similarity face alignment
│   ├── mouth_extractor.py    # Mouth ROI extraction & temporal smoothing
│   └── dataset_preprocessor.py # Dataset preprocessor orchestrator
├── tests/
│   ├── __init__.py
│   ├── generate_test_media.py    # Synthetic video/audio test generator
│   ├── test_preprocessing.py     # Unit tests for video/audio separation
│   └── test_face_preprocessing.py # Unit tests for face/mouth pipeline
├── main.py                   # CLI for Video & Audio Modality Separation
├── preprocess_faces_cli.py   # CLI for Face & Mouth ROI Preprocessing
└── README.md
```

---

## Output Dataset Schema

Processing a video sequence creates the following structure:

```text
processed_dataset/
└── video_001/
    ├── frames/               # Chronological input frames (frame_01.png, ...)
    ├── aligned_faces/        # Canonical aligned faces (224x224 PNG) (face_01.png, ...)
    ├── mouth_rois/           # Temporal mouth sequence (96x96 PNG) (mouth_01.png, ...)
    ├── landmarks/
    │   └── landmarks.json    # Cached 5-point coordinates & detection confidences
    └── metadata.json         # Complete structured metadata
```

### `metadata.json` Schema

```json
{
  "video_id": "video_001",
  "label": "real",
  "num_frames": 16,
  "frame_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  "face_detected": [true, true, ...],
  "face_bbox": [[x1, y1, x2, y2], ...],
  "face_confidence": [0.999, ...],
  "facial_landmarks": [
    {
      "left_eye": [x, y],
      "right_eye": [x, y],
      "nose": [x, y],
      "mouth_left": [x, y],
      "mouth_right": [x, y]
    },
    ...
  ],
  "alignment_status": ["success", ...],
  "mouth_roi_coordinates": [[x1, y1, x2, y2], ...],
  "mouth_roi_status": ["success", ...]
}
```

---

## Usage

### 1. Face Alignment & Mouth ROI Extraction (CLI)

```powershell
python preprocess_faces_cli.py --input-dir input/frames --output-dir processed_dataset/video_001 --video-id video_001 --label real
```

#### CLI Options:
| Option | Short | Default | Description |
|---|---|---|---|
| `--input-dir` | `-i` | *(required)* | Path to directory containing sampled PNG frames |
| `--output-dir` | `-o` | `processed_dataset/video_001` | Destination directory |
| `--video-id` | | `video_001` | Identifier for the video sequence |
| `--label` | | `unknown` | Ground truth label (`real`, `fake`, etc.) |
| `--face-size` | | `224` | Aligned face resolution ($224 \times 224$) |
| `--mouth-size` | | `96` | Mouth ROI resolution ($96 \times 96$) |
| `--device` | | `None` (auto) | PyTorch device (`cuda` or `cpu`) |

---

### 2. Python API

```python
from preprocessing import preprocess_frame_sequence

result = preprocess_frame_sequence(
    frames_dir="input/frames",
    output_dir="processed_dataset/video_001",
    video_id="video_001",
    label="real",
    face_size=(224, 224),
    mouth_size=(96, 96)
)

print(f"Processed {result.num_frames} frames.")
print(f"Faces Detected: {sum(result.face_detected)} / {result.num_frames}")
print(f"Metadata saved at: {result.metadata_file_path}")
```

---

### 3. Video & Audio Modality Separation (FFmpeg CLI)

```powershell
python main.py --input path/to/video.mp4 --output-dir output --fps 25
```

---

## Running Tests

Run the complete test suite (15 unit & integration tests):

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
