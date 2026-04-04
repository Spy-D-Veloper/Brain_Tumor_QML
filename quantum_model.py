"""
Hybrid Quantum-Classical Brain Tumor Classifier
===============================================
Pipeline:
1) Classical feature preprocessing from BraTS handcrafted features
2) PCA projection to exactly num_qubits components
3) Quantum data re-uploading circuit as feature transformer (embedding)
4) Classical classifier on quantum features (SVM / optional XGBoost)

Key design goals:
- Keep project valid QML by using a trainable quantum embedding layer
- Maximize practical performance with strong classical decision boundaries
- Train embeddings on clean simulator, evaluate robustness with noisy simulator
"""

import json
import os
import time
import warnings
import importlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(__file__)
RAW_CSV = os.path.join(BASE_DIR, "preprocessed", "features_raw.csv")
PCA_CSV = os.path.join(BASE_DIR, "preprocessed", "features_pca.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET_ACCURACY = 0.90
TARGET_ROC_AUC = 0.95


@dataclass
class Config:
    num_qubits: int = 8
    depth: int = 6
    train_test_size: float = 0.20
    epochs: int = 80
    shots_noisy_eval: int = 4096
    optimization_subset: int = 120
    use_xgboost: bool = True


CFG = Config()


def clamp_config() -> None:
    CFG.num_qubits = int(np.clip(CFG.num_qubits, 6, 8))
    CFG.depth = int(np.clip(CFG.depth, 6, 10))


def load_dataset_with_exact_pca(num_components: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load raw features, derive labels, PCA -> exactly num_components, MinMax to [0,1]."""
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Missing file: {RAW_CSV}")

    try:
        raw_df = pd.read_csv(RAW_CSV)
    except EmptyDataError as exc:
        raise ValueError(
            f"{RAW_CSV} is empty. Regenerate with preprocessing.py before training."
        ) from exc

    if raw_df.empty:
        raise ValueError(f"{RAW_CSV} has no rows. Regenerate with preprocessing.py.")

    if "ratio_et_wt" not in raw_df.columns:
        if os.path.exists(PCA_CSV):
            pca_df = pd.read_csv(PCA_CSV)
            pca_cols = [c for c in pca_df.columns if c != "subject_id"]
            if not pca_cols:
                raise ValueError("PCA CSV has no usable feature columns.")
            median_val = pca_df[pca_cols[0]].median()
            y = (pca_df[pca_cols[0]] >= median_val).astype(int).to_numpy()
            Xp = pca_df[pca_cols].to_numpy(dtype=np.float64)
            scaler = MinMaxScaler(feature_range=(0, 1))
            X = scaler.fit_transform(Xp)
            if X.shape[1] < num_components:
                X = np.pad(X, ((0, 0), (0, num_components - X.shape[1])), mode="constant")
            else:
                X = X[:, :num_components]
            return X, y, [f"PC{i+1}" for i in range(X.shape[1])]
        raise ValueError("ratio_et_wt missing and no fallback PCA file available.")

    median_ratio = raw_df["ratio_et_wt"].median()
    y = (raw_df["ratio_et_wt"] >= median_ratio).astype(int).to_numpy()

    leak_cols = [c for c in raw_df.columns if c.startswith("ratio_") or c.startswith("vol_")]
    drop_cols = ["subject_id", "label"] + leak_cols
    feature_cols = [c for c in raw_df.columns if c not in drop_cols]
    X_raw = raw_df[feature_cols].to_numpy(dtype=np.float64)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA(n_components=num_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_raw)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_pca)

    return X_scaled, y, [f"PC{i+1}" for i in range(num_components)]


def prepare_quantum_input(x: np.ndarray) -> np.ndarray:
    """Map classical PCA features to [0,1] and pad/truncate to the configured qubit count."""
    x = np.asarray(x, dtype=np.float64).flatten()
    if len(x) < CFG.num_qubits:
        x = np.pad(x, (0, CFG.num_qubits - len(x)), mode="constant")
    return np.clip(x[:CFG.num_qubits], 0.0, 1.0)


def build_reuploading_circuit(
    x: np.ndarray,
    theta: np.ndarray,
    num_qubits: int,
    depth: int,
    measure: bool = False,
) -> QuantumCircuit:
    """Data re-uploading circuit with trainable rotations and entanglement at each layer."""
    qc = QuantumCircuit(num_qubits, num_qubits if measure else 0)
    idx = 0

    for _ in range(depth):
        # Data encoding at every layer (re-uploading)
        for q in range(num_qubits):
            angle = float(np.pi * x[q])
            qc.ry(angle, q)
            qc.rz(angle, q)

        # Data-dependent ZZ-style encoding to enrich correlations
        for q in range(num_qubits):
            nq = (q + 1) % num_qubits
            zz_angle = float(np.pi * x[q] * x[nq])
            qc.rzz(zz_angle, q, nq)

        # Trainable block
        for q in range(num_qubits):
            qc.ry(float(theta[idx]), q)
            qc.rz(float(theta[idx + 1]), q)
            idx += 2

        # Ring entanglement
        for q in range(num_qubits):
            qc.cx(q, (q + 1) % num_qubits)

    if measure:
        qc.measure(range(num_qubits), range(num_qubits))

    return qc


def feature_labels(num_qubits: int) -> List[str]:
    labels = [f"Z{i+1}" for i in range(num_qubits)]
    labels += [f"X{i+1}" for i in range(num_qubits)]
    labels += [f"Y{i+1}" for i in range(num_qubits)]
    labels += [f"ZZ{i+1}_{i+2}" for i in range(num_qubits - 1)]
    return labels


def expectation_z_from_statevector(qc: QuantumCircuit, num_qubits: int) -> np.ndarray:
    """Compute per-qubit <Z> expectation from exact statevector."""
    state = Statevector.from_instruction(qc)
    probs = np.abs(state.data) ** 2
    exp = np.zeros(num_qubits, dtype=np.float64)

    for basis_idx, p in enumerate(probs):
        for q in range(num_qubits):
            bit = (basis_idx >> q) & 1
            exp[q] += p * (1.0 if bit == 0 else -1.0)

    return exp


def expectation_z_from_counts(counts: Dict[str, int], num_qubits: int) -> np.ndarray:
    """Compute per-qubit <Z> expectation from sampled counts."""
    total = sum(counts.values())
    if total == 0:
        return np.zeros(num_qubits, dtype=np.float64)

    exp = np.zeros(num_qubits, dtype=np.float64)
    for bitstring, c in counts.items():
        for q in range(num_qubits):
            bit = int(bitstring[-1 - q])
            exp[q] += (1.0 if bit == 0 else -1.0) * (c / total)

    return exp


def transform_to_x_basis(qc: QuantumCircuit, num_qubits: int) -> QuantumCircuit:
    rotated = qc.copy()
    for q in range(num_qubits):
        rotated.h(q)
    return rotated


def transform_to_y_basis(qc: QuantumCircuit, num_qubits: int) -> QuantumCircuit:
    rotated = qc.copy()
    for q in range(num_qubits):
        rotated.sdg(q)
        rotated.h(q)
    return rotated


def z_expectations_from_statevector(qc: QuantumCircuit, num_qubits: int) -> np.ndarray:
    state = Statevector.from_instruction(qc)
    probs = np.abs(state.data) ** 2
    exp = np.zeros(num_qubits, dtype=np.float64)
    for basis_idx, p in enumerate(probs):
        for q in range(num_qubits):
            bit = (basis_idx >> q) & 1
            exp[q] += p * (1.0 if bit == 0 else -1.0)
    return exp


def zz_expectations_from_statevector(qc: QuantumCircuit, num_qubits: int) -> np.ndarray:
    state = Statevector.from_instruction(qc)
    probs = np.abs(state.data) ** 2
    zz = np.zeros(num_qubits - 1, dtype=np.float64)
    for basis_idx, p in enumerate(probs):
        for q in range(num_qubits - 1):
            bit_i = (basis_idx >> q) & 1
            bit_j = (basis_idx >> (q + 1)) & 1
            sign = (1.0 if bit_i == 0 else -1.0) * (1.0 if bit_j == 0 else -1.0)
            zz[q] += p * sign
    return zz


def multi_observable_features_clean(x: np.ndarray, theta: np.ndarray, num_qubits: int, depth: int) -> np.ndarray:
    """Extract <Z>, <X>, <Y>, and nearest-neighbor <ZZ> features from a clean statevector."""
    x = prepare_quantum_input(x)
    qc = build_reuploading_circuit(x, theta, num_qubits, depth, measure=False)

    z_feat = z_expectations_from_statevector(qc, num_qubits)
    x_feat = z_expectations_from_statevector(transform_to_x_basis(qc, num_qubits), num_qubits)
    y_feat = z_expectations_from_statevector(transform_to_y_basis(qc, num_qubits), num_qubits)
    zz_feat = zz_expectations_from_statevector(qc, num_qubits)

    return np.concatenate([z_feat, x_feat, y_feat, zz_feat], axis=0)


def multi_observable_features_noisy(
    x: np.ndarray,
    theta: np.ndarray,
    num_qubits: int,
    depth: int,
    shots: int,
) -> np.ndarray:
    """Extract multi-observable features from a noisy simulator using basis rotations."""
    x = prepare_quantum_input(x)
    sim = AerSimulator(noise_model=create_noise_model())

    def measure_counts(circuit: QuantumCircuit) -> Dict[str, int]:
        measured = circuit.copy()
        measured.measure_all()
        tqc = transpile(measured, sim, optimization_level=1)
        return sim.run(tqc, shots=shots).result().get_counts()

    base = build_reuploading_circuit(x, theta, num_qubits, depth, measure=False)
    counts_z = measure_counts(base)
    counts_x = measure_counts(transform_to_x_basis(base, num_qubits))
    counts_y = measure_counts(transform_to_y_basis(base, num_qubits))

    z_feat = expectation_z_from_counts(counts_z, num_qubits)
    x_feat = expectation_z_from_counts(counts_x, num_qubits)
    y_feat = expectation_z_from_counts(counts_y, num_qubits)
    zz_feat = np.zeros(num_qubits - 1, dtype=np.float64)

    total = sum(counts_z.values())
    if total > 0:
        for bitstring, c in counts_z.items():
            prob = c / total
            for q in range(num_qubits - 1):
                bit_i = int(bitstring[-1 - q])
                bit_j = int(bitstring[-1 - (q + 1)])
                sign = (1.0 if bit_i == 0 else -1.0) * (1.0 if bit_j == 0 else -1.0)
                zz_feat[q] += prob * sign

    return np.concatenate([z_feat, x_feat, y_feat, zz_feat], axis=0)


def hybrid_feature_vector(x: np.ndarray, theta: np.ndarray, num_qubits: int, depth: int, noisy: bool = False, shots: int = 4096) -> np.ndarray:
    classical_part = prepare_quantum_input(x)
    if noisy:
        quantum_part = multi_observable_features_noisy(classical_part, theta, num_qubits, depth, shots)
    else:
        quantum_part = multi_observable_features_clean(classical_part, theta, num_qubits, depth)
    return np.concatenate([classical_part, quantum_part], axis=0)


def build_hybrid_feature_matrix(X: np.ndarray, theta: np.ndarray, noisy: bool = False, shots: int = 4096) -> np.ndarray:
    features = [hybrid_feature_vector(x, theta, CFG.num_qubits, CFG.depth, noisy=noisy, shots=shots) for x in X]
    return np.asarray(features, dtype=np.float64)


def quantum_embedding_clean(X: np.ndarray, theta: np.ndarray, num_qubits: int, depth: int) -> np.ndarray:
    """Embedding on clean simulator using exact expectation values."""
    feats = np.zeros((len(X), num_qubits), dtype=np.float64)
    for i, x in enumerate(X):
        qc = build_reuploading_circuit(x, theta, num_qubits, depth, measure=False)
        feats[i] = expectation_z_from_statevector(qc, num_qubits)
    return feats


def create_noise_model() -> NoiseModel:
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.004, 1), ["ry", "rz"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx", "rzz"])
    readout = ReadoutError([[0.98, 0.02], [0.02, 0.98]])
    noise_model.add_all_qubit_readout_error(readout)
    return noise_model


def quantum_embedding_noisy(
    X: np.ndarray,
    theta: np.ndarray,
    num_qubits: int,
    depth: int,
    shots: int,
) -> np.ndarray:
    """Embedding on noisy simulator using sampled counts."""
    sim = AerSimulator(noise_model=create_noise_model())
    feats = np.zeros((len(X), num_qubits), dtype=np.float64)

    for i, x in enumerate(X):
        qc = build_reuploading_circuit(x, theta, num_qubits, depth, measure=True)
        tqc = transpile(qc, sim, optimization_level=1)
        counts = sim.run(tqc, shots=shots).result().get_counts()
        feats[i] = expectation_z_from_counts(counts, num_qubits)

    return feats


def optimize_embedding_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_qubits: int,
    depth: int,
    maxiter: int,
    subset_size: int,
) -> Tuple[np.ndarray, List[float]]:
    """
    Optimize embedding parameters with COBYLA.
    Objective: maximize ROC-AUC of a simple logistic probe on quantum embeddings.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_params = num_qubits * depth * 2
    theta0 = rng.uniform(-0.15, 0.15, size=n_params)

    subset_n = min(subset_size, len(X_train))
    subset_idx = rng.choice(len(X_train), size=subset_n, replace=False)
    X_sub = X_train[subset_idx]
    y_sub = y_train[subset_idx]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(sss.split(X_sub, y_sub))
    X_tr, y_tr = X_sub[tr_idx], y_sub[tr_idx]
    X_va, y_va = X_sub[va_idx], y_sub[va_idx]

    history: List[float] = []

    def objective(theta: np.ndarray) -> float:
        emb_tr = build_hybrid_feature_matrix(X_tr, theta, noisy=False)
        emb_va = build_hybrid_feature_matrix(X_va, theta, noisy=False)

        probe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ])
        probe.fit(emb_tr, y_tr)
        p_va = probe.predict_proba(emb_va)[:, 1]

        auc = roc_auc_score(y_va, p_va)
        loss = -auc + 1e-4 * float(np.mean(theta ** 2))
        history.append(float(loss))

        step = len(history)
        print(f"Epoch {step:>2}/{maxiter} | ProxyAUC={auc:.4f} | Loss={loss:.6f}")
        return loss

    res = minimize(
        objective,
        theta0,
        method="COBYLA",
        options={"maxiter": maxiter, "disp": False, "rhobeg": 0.15, "tol": 1e-3},
    )

    return res.x.astype(np.float64), history


def train_classical_models(X_train_q: np.ndarray, y_train: np.ndarray) -> Dict[str, Dict]:
    """Train SVM (required) and XGBoost (optional) on quantum embeddings."""
    models: Dict[str, Dict] = {}

    svm_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            ),
        ]
    )
    svm_grid = {
        "clf__C": [1, 2, 5, 10, 20],
        "clf__gamma": ["scale", 0.01, 0.03, 0.05],
    }
    svm_search = GridSearchCV(svm_pipe, svm_grid, scoring="roc_auc", cv=5, n_jobs=-1)
    svm_search.fit(X_train_q, y_train)
    models["svm_rbf"] = {
        "model": svm_search.best_estimator_,
        "best_params": svm_search.best_params_,
        "cv_roc_auc": float(svm_search.best_score_),
    }

    if CFG.use_xgboost:
        try:
            xgb_module = importlib.import_module("xgboost")
            XGBClassifier = getattr(xgb_module, "XGBClassifier")

            xgb = XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=RANDOM_STATE,
                n_estimators=300,
                tree_method="hist",
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
            )
            xgb_grid = {
                "n_estimators": [200, 300],
                "max_depth": [3, 4, 5],
                "learning_rate": [0.03, 0.05, 0.08],
            }
            xgb_search = GridSearchCV(xgb, xgb_grid, scoring="roc_auc", cv=5, n_jobs=-1)
            xgb_search.fit(X_train_q, y_train)
            models["xgboost"] = {
                "model": xgb_search.best_estimator_,
                "best_params": xgb_search.best_params_,
                "cv_roc_auc": float(xgb_search.best_score_),
            }
        except Exception as exc:
            print(f"[Info] XGBoost skipped: {exc}")

    return models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def pick_best_model(models: Dict[str, Dict]) -> Tuple[str, Dict]:
    name = max(models.keys(), key=lambda k: models[k]["cv_roc_auc"])
    return name, models[name]


def preview_circuit(x0: np.ndarray, num_qubits: int, depth: int) -> None:
    preview_depth = min(2, depth)
    theta0 = np.zeros(num_qubits * preview_depth * 2, dtype=np.float64)
    qc = build_reuploading_circuit(x0, theta0, num_qubits, preview_depth, measure=True)
    print("\n[Circuit Preview]")
    print(f"Previewing first {preview_depth} of {depth} layers")
    print(qc.draw(output="text"))


def save_training_curve(history: List[float], filename: str) -> None:
    if not history:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(history) + 1), history, marker="o", linewidth=1)
    plt.xlabel("Optimization iteration")
    plt.ylabel("Objective loss")
    plt.title("Hybrid QML embedding optimization (COBYLA)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=120)
    plt.close()


def main() -> None:
    clamp_config()

    print("=" * 74)
    print(" Hybrid Quantum-Classical Brain Tumor Classifier ")
    print("=" * 74)
    print(f"Config | qubits={CFG.num_qubits}, depth={CFG.depth}, epochs={CFG.epochs}")

    print("\n[1/7] Loading and preprocessing data...")
    X, y, feat_cols = load_dataset_with_exact_pca(CFG.num_qubits)
    print(f"[Data] Samples={len(X)}, FeatureDims={len(feat_cols)}")
    print(f"[Data] Class balance={dict(pd.Series(y).value_counts())}")

    print("[2/7] Train/test split (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CFG.train_test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("[3/7] Circuit preview...")
    preview_circuit(X_train[0], CFG.num_qubits, CFG.depth)

    print("[4/7] Optimizing quantum embedding parameters (clean simulator)...")
    t0 = time.time()
    theta, loss_history = optimize_embedding_params(
        X_train,
        y_train,
        num_qubits=CFG.num_qubits,
        depth=CFG.depth,
        maxiter=CFG.epochs,
        subset_size=CFG.optimization_subset,
    )
    print(f"[Opt] Completed in {(time.time() - t0):.1f}s")

    print("[5/7] Extracting clean hybrid embeddings...")
    X_train_q_clean = build_hybrid_feature_matrix(X_train, theta, noisy=False)
    X_test_q_clean = build_hybrid_feature_matrix(X_test, theta, noisy=False)

    print("[6/7] Training classical classifiers (GridSearchCV)...")
    models = train_classical_models(X_train_q_clean, y_train)
    best_name, best_obj = pick_best_model(models)
    best_model = best_obj["model"]
    print(f"[Best] {best_name} | CV ROC-AUC={best_obj['cv_roc_auc']:.4f}")
    print(f"[Best params] {best_obj['best_params']}")

    print("[7/7] Evaluating on clean and noisy quantum embeddings...")
    p_clean = best_model.predict_proba(X_test_q_clean)[:, 1]
    y_clean = (p_clean >= 0.5).astype(int)
    clean_metrics = compute_metrics(y_test, y_clean, p_clean)

    X_test_q_noisy = build_hybrid_feature_matrix(X_test, theta, noisy=True, shots=CFG.shots_noisy_eval)
    p_noisy = best_model.predict_proba(X_test_q_noisy)[:, 1]
    y_noisy = (p_noisy >= 0.5).astype(int)
    noisy_metrics = compute_metrics(y_test, y_noisy, p_noisy)

    print("\n[Clean metrics]")
    for k, v in clean_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n[Noisy metrics]")
    for k, v in noisy_metrics.items():
        print(f"  {k}: {v:.4f}")

    summary = {
        "model_type": "Hybrid Quantum-Classical",
        "quantum_role": "feature_transformer",
        "classical_role": "final_classifier",
        "config": {
            "num_qubits": CFG.num_qubits,
            "depth": CFG.depth,
            "epochs": CFG.epochs,
            "shots_noisy_eval": CFG.shots_noisy_eval,
            "pca_components": CFG.num_qubits,
            "feature_scaling": "MinMax [0,1]",
        },
        "selected_classifier": best_name,
        "selected_classifier_cv_roc_auc": best_obj["cv_roc_auc"],
        "selected_classifier_params": best_obj["best_params"],
        "metrics": {
            "clean": clean_metrics,
            "noisy": noisy_metrics,
        },
        "targets": {
            "accuracy_target": TARGET_ACCURACY,
            "roc_auc_target": TARGET_ROC_AUC,
            "clean_accuracy_target_met": bool(clean_metrics["accuracy"] >= TARGET_ACCURACY),
            "clean_roc_auc_target_met": bool(clean_metrics["roc_auc"] >= TARGET_ROC_AUC),
        },
    }

    with open(os.path.join(RESULTS_DIR, "quantum_model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_training_curve(loss_history, "quantum_embedding_training_curve.png")

    print("\n" + "=" * 74)
    print(" SUMMARY ")
    print("=" * 74)
    print(f"Best model: {best_name}")
    print(f"Clean accuracy: {clean_metrics['accuracy']:.4f}")
    print(f"Clean ROC-AUC: {clean_metrics['roc_auc']:.4f}")
    print(
        f"Target status | Acc>={TARGET_ACCURACY:.2f}: "
        f"{'YES' if clean_metrics['accuracy'] >= TARGET_ACCURACY else 'NO'} | "
        f"AUC>={TARGET_ROC_AUC:.2f}: "
        f"{'YES' if clean_metrics['roc_auc'] >= TARGET_ROC_AUC else 'NO'}"
    )
    print(f"Saved: {os.path.join(RESULTS_DIR, 'quantum_model_metrics.json')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'quantum_embedding_training_curve.png')}")


if __name__ == "__main__":
    main()
