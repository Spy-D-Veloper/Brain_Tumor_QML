# Changes Log

Date: 2026-04-03

## What you did before (existing work in repo)

- Set up the project workspace and cloned the repository.
- Added and maintained core project scripts:
  - `preprocessing.py`
  - `baseline_ml.py` (initial baseline implementation)
  - quantum experiment scripts (`quantum_model.py`, `quantum_model1.py`, `circuit.py`, `resnet.py`)
- Added pipeline scripts:
  - `run_pipeline.bat`
  - `run_pipeline.sh`
- Prepared/generated processed data and experiment outputs under:
  - `preprocessed/`
  - `results/`
- Updated `.gitignore` rules for Python/venv/dataset/results/editor artifacts.
- Reached around mid-80% accuracy in one of the earlier runs and requested optimization to reach at least 90%.

## What I changed in this session

- Updated `baseline_ml.py` to improve reliability and reproducibility:
  - Set Matplotlib backend to `Agg` to avoid Tk/Tcl display errors in CLI/headless runs.
  - Added `GridSearchCV` tuning for RBF SVM (`SVM_RBF_Tuned`).
  - Kept tuned parameters and tuned CV-F1 in exported metrics.
  - Added per-feature best-model summary exports:
    - `results/raw_best_model.json`
    - `results/pca_best_model.json`
  - Added global best summary export:
    - `results/best_model_summary.json`
- Updated pipeline scripts so training runs automatically after preprocessing:
  - `run_pipeline.bat` now runs `python baseline_ml.py`
  - `run_pipeline.sh` now runs `python baseline_ml.py`
- Installed missing dependencies in the venv from `requirements.txt` to make scripts executable.
- Ran `baseline_ml.py` end-to-end and verified target achievement:
  - Best model: PCA + SVM_Linear
  - Best accuracy: `0.9283`
  - Target `>= 0.90`: met
- Updated `README.md` so docs match the current pipeline, outputs, and validated performance.

## Result summary

- Required target accuracy was achieved and verified in this environment.
- Documentation now reflects both setup/run instructions and current performance outputs.
