# Deepfake Detection Dataset

This project prepares the DeepFakeDetection (DFD) dataset for binary deepfake
detection and provides a baseline video-classification model.

## Dataset

The DFD videos are stored under:

```text
raw/DFD/DFD_original sequences/
raw/DFD/DFD_manipulated_sequences/DFD_manipulated_sequences/
```

The adapter discovers all supported video files in these folders. Original
videos are labelled `0`; manipulated videos are labelled `1`.

Raw videos, processed media, virtual environments, reports, caches, and model
checkpoints are excluded from Git by `.gitignore`.

## Requirements

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

```powershell
python -m src.phase0.adapters.dfd_adapter
```

This writes the active DFD manifests:

```text
metadata/dfd.csv
metadata/master.csv
metadata/train.csv
metadata/validation.csv
metadata/test.csv
reports/phase0/dataset_statistics.csv
```

All videos sharing a DFD source identifier are assigned to one split, preventing
source leakage between training, validation, and test data.

## Train the Baseline Model

Run the Phase 1 training pipeline on DFD (the default manifest):

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
raw/            DFD videos, not committed
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