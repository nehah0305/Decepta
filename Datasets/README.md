# Deepfake Detection Dataset

This project prepares FaceForensics++ C23 videos for binary deepfake detection
and provides a baseline video-classification model.

## Dataset

The downloaded videos are stored under:

```text
raw/faceforensicspp/c23/
```

The dataset contains 7,000 videos:

- 1,000 original videos
- 6,000 manipulated videos
- DeepFakeDetection, Deepfakes, Face2Face, FaceShifter, FaceSwap, and NeuralTextures

Raw videos, processed media, virtual environments, reports, caches, and model
checkpoints are excluded from Git by `.gitignore`.

## Requirements

- Python 3.10 or newer
- Windows PowerShell
- OpenCV-compatible video files
- NVIDIA GPU is optional

Create and activate the local environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current environment uses CPU PyTorch. A CUDA-enabled PyTorch build can be
installed separately when GPU training is required.

## Prepare Manifests

Generate the standard FaceForensics++ manifest:

```powershell
.\.venv\Scripts\python.exe -m src.phase0.adapters.faceforensics_adapter
```

Create leakage-safe train, validation, and test manifests:

```powershell
Copy-Item metadata\faceforensicspp.csv metadata\master.csv -Force
.\.venv\Scripts\python.exe -m src.phase0.run_phase0
```

This produces:

```text
metadata/faceforensicspp.csv
metadata/master.csv
metadata/train.csv
metadata/validation.csv
metadata/test.csv
reports/phase0/dataset_statistics.csv
```

Prepare Celeb-DF v2 using its official testing list and grouped training split:

```powershell
.\.venv\Scripts\python.exe -m src.phase0.adapters.celebdf_adapter
```

This creates `metadata/celebdf.csv` and updates the training manifests for
`raw/celeb df (v2)/`. The official test list is kept exactly; its source pairs
can also occur in the remaining videos, so this is an official dataset split
rather than a strict identity-disjoint evaluation split.

The split process groups related videos together and checks subject, source,
and split-group leakage before reporting Phase 0 as successful.

## Train the Baseline Model

Run the Phase 1 training pipeline:

```powershell
.\.venv\Scripts\python.exe -m src.phase1.train
```

The baseline model samples eight frames per video, resizes them to 224x224,
averages frame-level CNN features, and performs binary classification.

Useful options:

```powershell
.\.venv\Scripts\python.exe -m src.phase1.train --epochs 5 --batch-size 4 --frames 8
```

The best validation checkpoint is saved to:

```text
checkpoints/best_frame_cnn.pt
```

The trainer automatically uses CUDA when `torch.cuda.is_available()` is true;
otherwise it uses the CPU.

## Project Structure

```text
metadata/       Generated manifests
raw/            FaceForensics++ videos, not committed
reports/        Generated statistics, not committed
src/phase0/     Manifest creation, validation, splitting, leakage checks
src/phase1/     Dataset loader, CNN model, and training loop
requirements.txt Python dependencies
```

## Validation

Compile the source tree:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src
```

Check installed dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

Do not commit the raw videos, `.venv`, generated reports, or model checkpoints.