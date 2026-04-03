# QML Brain Tumor Classification (BraTS)

## Overview

This project contains a preprocessing + feature-engineering pipeline for BraTS MRI data and multiple modeling scripts (classical and quantum-inspired) for binary tumor aggressiveness classification.

The current validated target status is:

- Best accuracy: `0.9283` (PCA features + linear SVM)
- Target threshold: `>= 0.90`
- Target met: `YES`

## Repository layout

- `preprocessing.py`: NIfTI loading, normalization, handcrafted feature extraction, PCA export.
- `baseline_ml.py`: reproducible classical baselines (SVM/LogReg), tuning, plots, and metrics export.
- `quantum_model.py`, `quantum_model1.py`, `circuit.py`, `resnet.py`: quantum-inspired experiment scripts.
- `run_pipeline.bat`, `run_pipeline.sh`: one-command setup + preprocessing + baseline training.
- `preprocessed/`: generated features/volumes.
- `results/`: metrics, classification reports, confusion matrices, ROC curves, and summaries.

## Setup

1. Install Python 3.10+.
2. Create and activate a virtual environment.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows (cmd):

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Windows:

```bat
run_pipeline.bat
```

macOS/Linux:

```bash
bash run_pipeline.sh
```

Manual run:

```bash
python preprocessing.py
python baseline_ml.py
```

## Outputs

Main generated artifacts:

- `preprocessed/features_raw.csv`
- `preprocessed/features_pca.csv`
- `results/raw_metrics.json`
- `results/pca_metrics.json`
- `results/raw_best_model.json`
- `results/pca_best_model.json`
- `results/best_model_summary.json`

## Notes

- `baseline_ml.py` uses a non-interactive plotting backend for compatibility on headless/CLI setups.
- RBF-SVM hyperparameters are tuned with GridSearchCV and logged in the metrics JSON.
