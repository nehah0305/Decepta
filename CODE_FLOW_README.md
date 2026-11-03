# Decepta Code Flow

This document explains how an uploaded audio or video file moves through Decepta, which file owns each step, and what each step receives and returns.

## 1. System overview

```text
User selects media
        |
        v
frontend/src/pages/Detect.tsx
        |
        v
frontend/src/hooks/useDetection.tsx
        |
        | POST multipart file
        v
backend/deepfake_detection_service.py
        |
        +--> Audio input: run_audio_detection()
        |       |
        |       +--> audio/audio-deepfake-detect/src/inference/pipeline.py
        |       |       |
        |       |       +--> acoustic features and Acoustic Transformer
        |       |       +--> KernelSHAP explanations and waveform data
        |       |
        |       +--> Video input: run_video_detection()
        |               |
        |               +--> FFmpeg extracts a 16 kHz mono WAV stream
        |               +--> audio detection evaluates the embedded voice
        |               +--> VideoReader samples video frames
        |               +--> FaceAlignmentPipeline detects and aligns faces
        |               +--> ResNet-50 Stage B evaluates sampled faces
        |               +--> visual and audio scores are combined
        |
        v
DetectionRecord JSON response
        |
        v
frontend/src/pages/Analysis.tsx
and frontend/src/components/analysis/
        |
        v
Verdict, confidence, forensic reasons, timeline, waveform, and report
```

## 2. Input entry points

### Browser upload

**File:** [`frontend/src/pages/Detect.tsx`](./frontend/src/pages/Detect.tsx)

**Input:** A browser `File` selected as either audio or video.

**Processing:** The page stores the selected file and calls `runDetection(file, type)` from the detection context.

**Output:** The selected file is sent to the detection hook, and the UI displays progress and the completed result.

### Detection request coordinator

**File:** [`frontend/src/hooks/useDetection.tsx`](./frontend/src/hooks/useDetection.tsx)

**Main operation:** `runDetection(file, type)`

**Input:**

- `file`: browser `File` object.
- `type`: `'audio'` or `'video'`.

**Processing:**

1. Creates a `FormData` object with the field name `file`.
2. Sends audio to `/api/detect/audio` or video to `/api/detect/video`.
3. Waits for the backend JSON response.
4. Assigns a frontend ID and stores the result in React state and `localStorage`.
5. If the backend is unreachable or times out, creates a deterministic client fallback using `createResult()`.
6. Sets detection status to `completed` and progress to `100`.

**Output:** A `DetectionRecord` used by the analysis page, history page, and report generator.

> The fallback is a UI/demo fallback. It does not replace the real model inference when the backend responds successfully.

## 3. Backend service flow

### Service configuration and model loading

**File:** [`backend/deepfake_detection_service.py`](./backend/deepfake_detection_service.py)

Important locations:

- `ROOT_DIR`, `AUDIO_PKG_DIR`, and `MODEL_DIR` define the repository and model locations.
- `get_pipeline()` lazily loads the acoustic model and returns an `AudioDeepfakePipeline`.
- `get_visual_detector()` lazily loads the ResNet-50 detectors and face alignment pipeline.

Lazy loading means model files are loaded only when the first request needs them, rather than during module import.

### Health endpoint

**Function:** `health_check()`

**Route:** `GET /` and `GET /api/health`

**Input:** No request body.

**Processing:** Reports service readiness, model names, and available routes.

**Output:** JSON containing `status`, model descriptions, and route names.

## 4. Audio flow

### API upload

**Function:** `detect_audio(file: UploadFile)`

**File:** [`backend/deepfake_detection_service.py`](./backend/deepfake_detection_service.py)

**Input:** Multipart upload containing one audio file.

**Processing:**

1. Uses the uploaded filename extension for a temporary file suffix.
2. Reads the upload into a temporary file.
3. Calls `run_audio_detection()`.
4. Deletes the temporary file in the cleanup block.

**Output:** A JSON `DetectionRecord` with the audio verdict and analysis data.

### Audio inference orchestration

**Function:** `run_audio_detection(file_path, file_name=None)`

**File:** [`backend/deepfake_detection_service.py`](./backend/deepfake_detection_service.py)

**Input:**

- `file_path`: temporary or local path to an audio file.
- `file_name`: optional original filename used in the response.

**Processing:**

1. Gets the cached acoustic pipeline through `get_pipeline()`.
2. Calls `AudioDeepfakePipeline.infer()`.
3. Reads the returned fake probability (`pFake`) and decision.
4. Converts the probability into a `DEEPFAKE` or `GENUINE` verdict.
5. Converts KernelSHAP items into readable forensic reasons with `_shap_to_forensic_reasons()`.
6. Preserves waveform, frame count, raw probabilities, and model metadata.

**Output:** A `DetectionRecord` containing:

- `verdict` and `result`
- `confidence`
- `reasons`
- `acousticShap`
- `waveform`
- raw model values such as `pFake`, `cScore`, `decision`, and `logit`

### Acoustic model pipeline

**File:** [`audio/audio-deepfake-detect/src/inference/pipeline.py`](./audio/audio-deepfake-detect/src/inference/pipeline.py)

**Input:** An audio path and inference configuration.

**Processing:** The pipeline loads audio, extracts the configured acoustic descriptors, runs the Acoustic Transformer/checkpoint, and optionally calculates KernelSHAP attributions.

**Output:** A dictionary consumed by `run_audio_detection()`, including fake probability, decision metadata, acoustic explanations, analyzed frame count, and waveform information.

## 5. Video flow

### API upload

**Function:** `detect_video(file: UploadFile)`

**File:** [`backend/deepfake_detection_service.py`](./backend/deepfake_detection_service.py)

**Input:** Multipart upload containing a video file.

**Processing:** Stores the upload temporarily, calls `run_video_detection()`, logs the result, returns JSON, and removes the temporary file.

**Output:** A multimodal `DetectionRecord`.

### Multimodal inference orchestration

**Function:** `run_video_detection(file_path, file_name=None)`

**File:** [`backend/deepfake_detection_service.py`](./backend/deepfake_detection_service.py)

**Input:**

- `file_path`: path to the uploaded video.
- `file_name`: optional original filename.

**Processing:**

1. **Extract audio:** FFmpeg converts the embedded audio to 16 kHz, mono, PCM WAV.
2. **Check speech energy:** RMS and peak thresholds prevent quiet background noise from being treated as speech.
3. **Analyze audio:** If sufficient voice energy exists, `run_audio_detection()` evaluates the extracted WAV.
4. **Read video metadata:** `VideoReader` provides frame count, dimensions, and duration.
5. **Detect and align faces:** `FaceAlignmentPipeline` tracks faces and produces canonical aligned face images.
6. **Sample frames:** The detector evaluates evenly distributed frames and uses a sequential fallback if the first sampling pass finds no faces.
7. **Run visual models:** The primary ResNet-50 model provides spatial fake probabilities; the latent model provides additional face-swap evidence.
8. **Calculate temporal evidence:** Blur, inter-frame mean squared error, frame fractions, medians, upper quantiles, and maxima are calculated.
9. **Apply visual rules:** Persistent, focal, latent, and temporal manipulation patterns are evaluated.
10. **Fuse modalities:** A visual fake or an audio fake is sufficient to produce a deepfake verdict. If both are real, the real confidence values are combined.
11. **Build explanations:** Human-readable forensic reasons are created from visual and acoustic evidence.

**Output:** A multimodal `DetectionRecord` containing:

- final `verdict`, `result`, and `confidence`
- number of sampled face segments
- visual and acoustic forensic reasons
- optional SHAP and waveform data
- raw visual, audio, temporal, resolution, bitrate, and compression metadata

### Video reader and face alignment

**Files:**

- [`backend/model/preprocessing/video_reader.py`](./backend/model/preprocessing/video_reader.py)
- [`backend/model/preprocessing/face_alignment.py`](./backend/model/preprocessing/face_alignment.py)

**Input:** Video path and sampled frame indices.

**Processing:** `VideoReader` decodes frames and metadata. `FaceAlignmentPipeline` detects faces, tracks them across frames, and maps detected landmarks to a canonical face size.

**Output:** RGB frames, timestamps, face detection status, aligned face tensors, and landmark information for visual inference.

## 6. API routes

| Route | Method | Input | Processing function | Output |
|---|---|---|---|---|
| `/api/health` | `GET` | None | `health_check()` | Service and model status JSON |
| `/api/detect/audio` | `POST` | Multipart audio file | `detect_audio()` → `run_audio_detection()` | Audio `DetectionRecord` |
| `/api/detect/video` | `POST` | Multipart video file | `detect_video()` → `run_video_detection()` | Multimodal `DetectionRecord` |
| `/api/infer` | `POST` | Multipart audio or video file | `infer_generic()` | Dispatches by file extension |

The routes are defined in [`backend/deepfake_detection_service.py`](./backend/deepfake_detection_service.py). The frontend calls the audio and video detection routes from [`frontend/src/hooks/useDetection.tsx`](./frontend/src/hooks/useDetection.tsx).

## 7. Result display and persistence

### Analysis page

**File:** [`frontend/src/pages/Analysis.tsx`](./frontend/src/pages/Analysis.tsx)

**Input:** A detection ID from the route.

**Processing:** Looks up the matching `DetectionRecord` through the detection context.

**Output:** Renders the detailed analysis view.

### Analysis components

**Directory:** [`frontend/src/components/analysis/`](./frontend/src/components/analysis/)

These components consume the `DetectionRecord` and render:

- verdict and confidence
- metrics and model version
- timeline of inference stages
- waveform bars for audio
- SHAP/acoustic forensic reasons
- ablation and benchmark information

### Persistence and reports

- [`frontend/src/hooks/useDetection.tsx`](./frontend/src/hooks/useDetection.tsx) persists detection history in `localStorage`.
- [`frontend/src/utils/reportGenerator.ts`](./frontend/src/utils/reportGenerator.ts) converts a result into a downloadable forensic report.
- [`frontend/src/types/index.ts`](./frontend/src/types/index.ts) defines the shared `DetectionRecord`, waveform, SHAP, and forensic reason shapes.

## 8. Command-line flow

**File:** [`backend/deepfake_detection_service.py`](./backend/deepfake_detection_service.py)

The `main()` function supports two command-line modes:

```text
python backend/deepfake_detection_service.py --file path/to/audio.wav
python backend/deepfake_detection_service.py --file path/to/video.mp4
python backend/deepfake_detection_service.py --serve --host 0.0.0.0 --port 8000
```

**Input:** CLI arguments `--file`, `--serve`, `--host`, and `--port`.

**Processing:** File extensions select audio or video inference. With `--serve`, Uvicorn starts the FastAPI application.

**Output:** Either formatted JSON detection results or a running HTTP service.

## 9. Important implementation note

The backend is the source of model-based truth when it is available. The frontend intentionally contains a deterministic fallback in `createResult()` so the interface remains usable while the backend is offline. For production verification, start the backend, confirm `GET /api/health`, and then run an upload through the API-backed path.
