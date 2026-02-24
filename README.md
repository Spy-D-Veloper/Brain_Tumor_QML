# QML — BraTS Preprocessing & Feature Extraction

This repository contains a preprocessing and feature-extraction pipeline for the BraTS MRI dataset.

Quick overview

- Loads modalities: FLAIR, T1, T1CE, T2 and segmentation (NIfTI .nii.gz)
- Crops to brain region, resizes to 128×128×128, z-score normalises
- Extracts first-order + lightweight 3D GLCM texture features and tumor volume ratios
- Saves preprocessed volumes, feature CSVs and a PCA pipeline

Prerequisites

- Python 3.9+ (3.13 confirmed working here)
- Recommended packages listed in `requirements.txt`

Install

Windows / WSL / PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # or use Activate.bat on cmd
pip install -r requirements.txt
```

Run

```powershell
python preprocessing.py
```

Outputs

- `preprocessed/volumes/` — compressed `.npz` per subject with preprocessed image stack + seg
- `preprocessed/features_raw.csv` — raw extracted features per subject
- `preprocessed/features_pca.csv` — PCA-reduced features
- `preprocessed/pca_pipeline.pkl` — saved scaler + PCA model
- `preprocessed/sample_slices/` — example PNG slices

Adjusting behaviour

- Edit `preprocessing.py` to change `TARGET_SHAPE`, `PCA_VARIANCE`, or `MODALITIES`.
- For faster development test a subset of subjects by modifying the `subjects` glob in `main()`.

Notes

- The `dataset/` folder is large and intentionally ignored by `.gitignore`.
- The pipeline is intended as a starting point for experiments — adapt feature extraction to your needs.

Contact

- If you want further integrations (radiomics, augmentation, model training), tell me which direction to add next.
