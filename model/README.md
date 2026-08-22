# Decepta — Multimodal Deepfake Detection Preprocessing Pipeline

This module implements the preprocessing stage for a multimodal deepfake detection system. It ingests video files containing both video and audio streams, and separates them into independent modalities while preserving temporal synchronization and forensic artifact integrity.

---

## Features

### 1. Video Processing — Frame Extraction
- **Lossless PNG Frame Output**: Extracts frames strictly as `.png` files to avoid compression artifacts, sharpening, or quality loss that could alter forensic cues.
- **Configurable Frame Rate**: Extract frames at 5–10 FPS (or custom rates).
- **Sequential Naming**: Saves frames sequentially as `frame_000001.png`, `frame_000002.png`, etc., in `output/frames/`.
- **Chronological Timestamps**: Calculates and tracks per-frame timestamps (in seconds).
- **Full Forensic Metadata**: Records original FPS, extracted FPS, resolution, duration, total frames, and per-frame timestamps.

### 2. Audio Processing — Audio Extraction
- **Independent Stream Extraction**: Extracts the raw audio stream into `output/audio/audio.wav`.
- **Lossless PCM 16-bit WAV**: High-fidelity uncompressed linear PCM encoding (`pcm_s16le`).
- **Strict Temporal Alignment**: Starts at exact timestamp `00:00:00` aligned 1:1 with video frames.
- **Configurable Resampling**: Optional target sample rates (e.g., 16 kHz for speech models, 44.1 kHz, or native).

### 3. Consolidated Forensic Metadata
- Automatically generates `output/metadata.json` with all stream parameters, frame timestamps, and execution summary.

---

## Directory Structure

```
model/
├── preprocessing/
│   ├── __init__.py           # Module exports
│   ├── ffmpeg_utils.py       # FFmpeg auto-discovery, probing, execution
│   ├── video_processor.py    # Lossless PNG frame extraction & timestamping
│   ├── audio_processor.py    # Independent WAV audio extraction
│   └── pipeline.py           # Multimodal pipeline orchestrator & metadata JSON
├── tests/
│   ├── __init__.py
│   ├── generate_test_media.py # Synthetic video/audio test generator
│   └── test_preprocessing.py # Comprehensive unit and integration test suite
├── main.py                   # Command-line interface (CLI)
├── README.md
└── output/                   # Default output folder
    ├── frames/               # Extracted PNG frames (frame_000001.png, ...)
    ├── audio/                # Extracted WAV audio (audio.wav)
    └── metadata.json         # Complete forensic metadata
```

---

## Usage

### 1. Command-Line Interface (CLI)

Run preprocessing on an input video:

```powershell
python main.py --input path/to/video.mp4 --output-dir output --fps 5
```

#### CLI Options:
| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-i` | *(required)* | Path to the input video file |
| `--output-dir` | `-o` | `output` | Directory where `frames/`, `audio/`, and `metadata.json` are saved |
| `--fps` | `-f` | `5.0` | Target frame rate for frame extraction (e.g., 5.0 or 10.0 FPS) |
| `--sample-rate`| `-sr`| `None` (native) | Target audio sample rate in Hz (e.g. 16000 for speech models) |
| `--channels` | `-c` | `None` (native) | Target audio channels (e.g. 1 for mono, 2 for stereo) |
| `--no-metadata`| | `False` | Disable writing `metadata.json` |

---

### 2. Python API

```python
from preprocessing import preprocess_video

result = preprocess_video(
    video_path="path/to/video.mp4",
    output_dir="output",
    fps=5.0,
    audio_sample_rate=16000  # optional
)

print(f"Status: {result.status}")
print(f"Total Extracted Frames: {result.total_extracted_frames}")
print(f"Audio Path: {result.audio_file_path}")
print(f"Frame 1 Timestamp: {result.frame_timestamps['frame_000001.png']}s")
```

Or using the class-based interface:

```python
from preprocessing import MultimodalPreprocessor

preprocessor = MultimodalPreprocessor(extracted_fps=10.0)
result = preprocessor.process("path/to/video.mp4", output_dir="output")
```

---

## Running Tests

Execute the automated test suite:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
