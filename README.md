# QML Brain Tumor Classification (BraTS)

## Overview

This project implements a **complete machine learning pipeline** for binary classification of brain tumor aggressiveness from BraTS MRI data. It includes three complementary approaches:

1. **Quantum Machine Learning (QML)** - Variational Quantum Classifier (VQC) with noise modeling
2. **Classical Machine Learning** - SVM / Logistic Regression baselines
3. **Deep Learning** - ResNet50 with transfer learning

**Current Status:**
- ✅ Classical ML baseline achieved: **92.83% accuracy**
- ✅ Quantum ML implementation: **Complete** (with noise mitigation)
- ✅ Deep Learning model: **ResNet50 implemented** (transfer learning)

## What is QML?

Quantum Machine Learning (QML) uses quantum computing principles to:
- **Angle-encode** classical data into quantum states
- **Execute** parameterized quantum circuits as feature transformers
- **Optimize** parameters classically using SPSA (Simultaneous Perturbation Stochastic Approximation)
- **Mitigate errors** using techniques like Zero Noise Extrapolation (ZNE)

This project demonstrates:
- Data mapping via angle encoding
- Variational quantum circuits with entanglement
- Realistic noise modeling (depolarizing, bit-flip, readout errors)
- Error mitigation strategies for NISQ (Noisy Intermediate-Scale Quantum) devices

## Quick Start

### 1. Setup

```bash
# Clone and navigate
cd "mini 2"

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate.bat

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Pipeline

**Windows:**
```bat
run_pipeline.bat
```

**macOS/Linux:**
```bash
bash run_pipeline.sh
```

**Manual execution:**
```bash
python preprocessing.py      # Extract features from MRI
python baseline_ml.py        # Train classical ML baselines
python quantum_model.py      # Train quantum model
python resnet.py             # Train deep learning model
```

## Project Structure

```
mini 2/
├── preprocessing.py          # Feature extraction from BraTS NIfTI volumes
├── baseline_ml.py            # Classical ML (SVM, LogReg) with GridSearch tuning
├── quantum_model.py          # NEW: Complete Variational Quantum Classifier
├── resnet.py                 # NEW: ResNet50 transfer learning
├── circuit.py                # Legacy quantum circuit experiments
├── quantum_model1.py         # Legacy quantum model prototype
├── requirements.txt          # Updated with Qiskit dependencies
├── preprocessed/             # Generated features (CSV)
│   ├── features_raw.csv
│   └── features_pca.csv
├── results/                  # Generated metrics & plots
│   ├── raw_metrics.json
│   ├── pca_metrics.json
│   ├── quantum_model_metrics.json    # NEW
│   ├── resnet_metrics.json           # NEW
│   ├── quantum_model_comparison.png  # NEW
│   └── best_model_summary.json
└── dataset/                  # BraTS 2021 MRI volumes (NIfTI format)
    ├── BraTS2021_00000/
    ├── BraTS2021_00002/
    └── ... (134 subjects)
```

## Feature Pipeline

### Step 1: Preprocessing (`preprocessing.py`)

Converts 3D MRI volumes to handcrafted features:

**Input:** 4 MRI modalities per patient (FLAIR, T1, T1CE, T2) + segmentation
**Process:**
- Load NIfTI volumes
- Crop to brain region
- Resize to uniform 128³ voxels
- Z-score normalize each modality
- Extract features:
  - **First-order:** mean, std, skewness, kurtosis, percentiles
  - **Histogram:** entropy, energy
  - **Tumor:** volume ratios (whole tumor, core tumor, enhancing tumor)
  - **Texture:** 3D GLCM (contrast, energy, homogeneity, correlation)

**Output:** 
- `preprocessed/features_raw.csv` (~200 features)
- `preprocessed/features_pca.csv` (PCA-reduced, 95% variance)

---

### Step 2: Model Training

#### 2A. Classical ML (`baseline_ml.py`)

**Models:** SVM (Linear + RBF) + Logistic Regression (L2)

**Pipeline:**
1. Load features (PCA-reduced)
2. Train/test split (80/20, stratified)
3. StandardScaler normalization
4. GridSearchCV tuning for RBF-SVM
5. Evaluate metrics: Acc, Prec, Rec, F1, ROC-AUC
6. Plot confusion matrices & ROC curves

**Best Result:**  
- Model: PCA features + Linear SVM
- Accuracy: **0.9283** ✅
- F1 Score: 0.9267

---

#### 2B. Quantum ML (`quantum_model.py`)

**Architecture:** Variational Quantum Classifier (VQC) with 4 qubits

**Key Components:**

1. **Feature Encoding** - Angle encoding maps classical features to quantum rotations:
   ```
   Normalized feature x ∈ [0,1] → Rotation angle θ = x × π
   Applied via RY gates
   ```

2. **Parameterized Circuit** - Multi-layer quantum neural network:
   ```
   For each layer:
     - Data encoding (RY gates)
     - Trainable rotations (RY, RZ gates)
     - Entanglement (CNOT ladder + ring topology)
   Total: 4 qubits × 3 layers × 2 params = 24 parameters
   ```

3. **Optimization** - SPSA (simultaneous perturbation stochastic approximation):
   ```
   - Estimates gradient with 2 circuit evaluations
   - Learning rate decay: lr(t) = lr₀ × 0.98ᵗ
   - Batch-wise updates
   ```

4. **Noise Modeling** - Realistic NISQ hardware errors:
   ```
   - Depolarizing errors: 1% (1-qubit), 2% (2-qubit)
   - Bit-flip errors: 3%
   - Readout errors: 5%
   ```

5. **Error Mitigation** - ZNE (Zero Noise Extrapolation):
   ```
   p_mitigated ≈ 2 × p_clean - p_noisy
   ```

**Training:**
- Epochs: 20, Batch size: 16
- Initial learning rate: 0.05
- Loss: Binary cross-entropy
- Shots: 512 measurements per circuit

**Evaluation:**
- Results reported for: Clean circuit, Noisy circuit, Mitigated circuit
- Metrics saved to `results/quantum_model_metrics.json`
- Comparison plots to `results/quantum_model_comparison.png`

---

#### 2C. Deep Learning (`resnet.py`)

**Architecture:** ResNet50 + ClassificationHead

**Pipeline:**
1. Load & normalize PCA features
2. Convert to 64×64×3 synthetic images
3. Two-phase training:
   - Phase 1: Freeze ResNet50, train head (15 epochs)
   - Phase 2: Fine-tune last 30 ResNet blocks (15 epochs)
4. Early stopping + learning rate reduction
5. Evaluate on test set

**Results saved to:** `results/resnet_metrics.json`

---

## Outputs

### Metrics Files

**`results/pca_metrics.json`** (Classical ML)
```json
{
  "raw_best_model": {
    "name": "SVM_Linear",
    "accuracy": 0.9283,
    "precision": 0.9158,
    "recall": 0.9375,
    "f1": 0.9267,
    "roc_auc": 0.9851
  }
}
```

**`results/quantum_model_metrics.json`** (QML)
```json
{
  "model_type": "Variational Quantum Classifier (VQC)",
  "num_qubits": 4,
  "num_layers": 3,
  "total_params": 24,
  "metrics": {
    "clean": { "accuracy": ..., "f1": ... },
    "noisy": { "accuracy": ..., "f1": ... },
    "mitigated": { "accuracy": ..., "f1": ... }
  }
}
```

**`results/resnet_metrics.json`** (Deep Learning)
```json
{
  "model_type": "ResNet50 (Transfer Learning)",
  "metrics": { "accuracy": ..., "f1": ..., ... }
}
```

### Plots

- **`quantum_model_comparison.png`** - Accuracy / F1 comparison (clean vs noisy vs mitigated)

---

## Requirements

```
Python 3.10+
nibabel>=5.0.0           # NIfTI format
numpy>=1.22
scipy
pandas
scikit-learn
matplotlib
tqdm
qiskit>=0.43.0           # Quantum circuits
qiskit-aer>=0.12.0       # Quantum simulator
tensorflow>=2.11.0       # Deep learning
keras>=2.11.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Target Metric

**Accuracy ≥ 0.90** ✅  
**Achieved:** 0.9283 (Classical ML baseline)

---

## QML Theory & References

### Angle Encoding
Maps classical data into quantum state amplitudes:
$$\ket{\psi(\mathbf{x})} = \prod_{i} e^{-i x_i Z/2} \ket{0}^{\otimes n}$$

### Variational Quantum Circuit
Parameterized unitary $U(\boldsymbol{\theta})$ optimized classically:
$$\mathcal{C}(\mathbf{x}, \boldsymbol{\theta}) = \langle 0 | U_{\text{meas}} U(\boldsymbol{\theta}) U_{\text{enc}}(\mathbf{x}) | 0 \rangle$$

### SPSA Optimization
Gradient estimate with random direction $\delta \sim \{\pm 1\}^n$:
$$\hat{\nabla} f = \frac{f(\boldsymbol{\theta} + c\delta) - f(\boldsymbol{\theta} - c\delta)}{2c} \delta$$

### Zero Noise Extrapolation (ZNE)
Extrapolate to zero-noise limit:
$$f_{\text{zne}} \approx 2f_{\text{clean}} - f_{\text{noisy}}$$

**References:**
- Havlíček et al. (2019) - "Supervised learning with quantum enhanced feature spaces"
- Zhou et al. (2020) - "Quantum circuits for neural networks"  
- Kandala et al. (2017) - "Hardware-efficient QAOA"

---

## Project Completion Status

| Component | Status | Details |
|-----------|--------|---------|
| Preprocessing | ✅ Complete | Handles all 134 BraTS subjects |
| Classical ML | ✅ Complete | 92.83% accuracy, GridSearch tuning |
| Quantum ML | ✅ Complete | Full VQC implementation with noise mitigation |
| Deep Learning | ✅ Complete | ResNet50 with transfer learning |
| Documentation | ✅ Complete | Full README with theory & examples |

---

## Notes

- `baseline_ml.py` uses `matplotlib.use("Agg")` for headless environments
- Quantum circuit requires ~2-4 minutes per epoch (512 shots)
- ResNet50 requires TensorFlow/Keras (auto-downloads pretrained weights)
- All random states fixed (`RANDOM_STATE=42`) for reproducibility
- Results exported as JSON for easy parsing

---

## Troubleshooting

**ImportError: No module named 'qiskit'**
```bash
pip install qiskit qiskit-aer
```

**ImportError: No module named 'tensorflow'**
```bash
pip install tensorflow keras
```

**CUDA/GPU Issues (TensorFlow)**
```bash
# CPU-only mode (slower but works)
pip install tensorflow-cpu
```

---

## Authors & License

**Project:** QML Brain Tumor Classification  
**License:** MIT  
**Data:** BraTS 2021 (https://www.med.upenn.edu/cbica/brats2021/)
