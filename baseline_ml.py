"""
Classic ML Baseline — SVM & Logistic Regression
=================================================
Target:  Binary classification on tumour aggressiveness.
         Label derived from the enhancing-tumour ratio:
           1 = high ET ratio  (≥ median)   — aggressive
           0 = low  ET ratio  (< median)   — less aggressive

Features: Raw hand-crafted features from `features_raw.csv`
          AND PCA-reduced features from `features_pca.csv`.

Pipeline:
  1. Load features & derive label
  2. Train / Test split (80/20, stratified)
  3. StandardScaler (fit on train only)
  4. Train  SVM (Linear + RBF)  &  Logistic Regression (L2)
  5. Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC
  6. Confusion matrices & ROC curves
  7. Feature-importance bar chart (Logistic Regression coefficients)
  8. Save all results to  results/
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────── paths ───────────────────────────
BASE_DIR    = os.path.dirname(__file__)
RAW_CSV     = os.path.join(BASE_DIR, "preprocessed", "features_raw.csv")
PCA_CSV     = os.path.join(BASE_DIR, "preprocessed", "features_pca.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ════════════════════════════════════════════════════════════════
#  1. DATA LOADING & LABEL DERIVATION
# ════════════════════════════════════════════════════════════════
def load_and_label(csv_path: str):
    """
    Load features CSV.
    Derive a binary label from the enhancing-tumour ratio column:
        1 → ratio_et_wt ≥ median  (high enhancement / aggressive)
        0 → ratio_et_wt <  median  (low enhancement / less aggressive)
    """
    df = pd.read_csv(csv_path)

    # ---------- derive label ----------
    # ratio_et_wt exists in the raw CSV; for PCA CSV we merge it in
    if "ratio_et_wt" not in df.columns:
        raw = pd.read_csv(RAW_CSV)[["subject_id", "ratio_et_wt"]]
        df = df.merge(raw, on="subject_id", how="left")

    median_et = df["ratio_et_wt"].median()
    df["label"] = (df["ratio_et_wt"] >= median_et).astype(int)

    print(f"[Data]  Source        : {os.path.basename(csv_path)}")
    print(f"[Data]  Samples       : {len(df)}")
    print(f"[Data]  ET-ratio med. : {median_et:.4f}")
    print(f"[Data]  Class balance : {dict(df['label'].value_counts())}\n")

    # ---------- features & target ----------
    drop_cols = ["subject_id", "label"]
    # Also drop the raw tumour-ratio cols that leak the label
    leak_cols = [c for c in df.columns
                 if c.startswith("ratio_") or c.startswith("vol_")]
    drop_cols += leak_cols

    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].values.astype(np.float64)
    y = df["label"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feat_cols, df


# ════════════════════════════════════════════════════════════════
#  2. MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════
def get_models():
    """Return a dict of {name: sklearn estimator}."""
    return {
        "SVM_Linear": SVC(
            kernel="linear", C=1.0, probability=True,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "SVM_RBF": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "LogReg_L2": LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    }


# ════════════════════════════════════════════════════════════════
#  3. TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════════
def evaluate(name, model, X_train, X_test, y_train, y_test,
             scaler, feat_cols, tag):
    """Train, predict, score, and save artefacts for one model."""
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auc   = roc_auc_score(y_test, y_proba)

    report_str = classification_report(
        y_test, y_pred,
        target_names=["Low-ET (0)", "High-ET (1)"],
        zero_division=0,
    )

    metrics = dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc)

    print(f"  [{name}]  Acc={acc:.4f}  Prec={prec:.4f}  "
          f"Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    # ── Cross-validation on the FULL train set (sanity check) ──
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipe,
        np.vstack([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        cv=cv, scoring="f1",
    )
    metrics["cv_f1_mean"] = float(cv_scores.mean())
    metrics["cv_f1_std"]  = float(cv_scores.std())
    print(f"           5-Fold CV F1 = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Confusion matrix plot ──
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Low-ET", "High-ET"],
        cmap="Blues", ax=ax_cm,
    )
    ax_cm.set_title(f"{name} — Confusion Matrix ({tag})")
    fig_cm.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f"{tag}_{name}_cm.png")
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)

    # ── ROC curve plot ──
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(
        y_test, y_proba, ax=ax_roc, name=name,
    )
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_title(f"{name} — ROC Curve ({tag})")
    fig_roc.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, f"{tag}_{name}_roc.png")
    fig_roc.savefig(roc_path, dpi=150)
    plt.close(fig_roc)

    return metrics, report_str, model


# ════════════════════════════════════════════════════════════════
#  4. FEATURE IMPORTANCE  (Logistic Regression coefficients)
# ════════════════════════════════════════════════════════════════
def plot_feature_importance(model, feat_cols, tag, top_n=20):
    """Bar chart of the top-N absolute LR coefficients."""
    if not hasattr(model, "coef_"):
        return
    coefs = model.coef_.ravel()
    idx   = np.argsort(np.abs(coefs))[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [feat_cols[i] if i < len(feat_cols) else f"feat_{i}" for i in idx]
    vals   = coefs[idx]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    ax.barh(range(len(idx)), vals, color=colors)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient value")
    ax.set_title(f"Logistic Regression — Top-{top_n} Features ({tag})")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{tag}_logreg_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] Feature importance → {path}")


# ════════════════════════════════════════════════════════════════
#  5. COMPARISON BAR CHART
# ════════════════════════════════════════════════════════════════
def plot_comparison(all_metrics: dict, tag: str):
    """Grouped bar chart comparing all models across metrics."""
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names  = list(all_metrics.keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mname in enumerate(model_names):
        vals = [all_metrics[mname][m] for m in metric_names]
        ax.bar(x + i * width, vals, width, label=mname)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=7)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison ({tag} features)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{tag}_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] Comparison chart  → {path}\n")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def run_experiment(csv_path: str, tag: str):
    """Full experiment on one feature set (raw or PCA)."""
    print("=" * 62)
    print(f"  EXPERIMENT: {tag.upper()} FEATURES")
    print("=" * 62)

    X, y, feat_cols, _ = load_and_label(csv_path)

    # ── Train / Test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"[Split] Train={len(y_train)}  Test={len(y_test)}  "
          f"(test ratio={TEST_SIZE})\n")

    # ── Scale ──
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Evaluate each model ──
    models       = get_models()
    all_metrics  = {}
    all_reports  = {}

    for name, clf in models.items():
        metrics, report, fitted = evaluate(
            name, clf,
            X_train_s, X_test_s, y_train, y_test,
            scaler, feat_cols, tag,
        )
        all_metrics[name] = metrics
        all_reports[name] = report

        # Feature importance for LR
        if name.startswith("LogReg"):
            plot_feature_importance(fitted, feat_cols, tag)

    # ── Comparison chart ──
    plot_comparison(all_metrics, tag)

    # ── Save metrics JSON ──
    json_path = os.path.join(RESULTS_DIR, f"{tag}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ── Save classification reports ──
    report_path = os.path.join(RESULTS_DIR, f"{tag}_classification_reports.txt")
    with open(report_path, "w") as f:
        for name, rpt in all_reports.items():
            f.write(f"{'='*50}\n{name}\n{'='*50}\n{rpt}\n\n")

    return all_metrics


def main():
    print()
    raw_metrics = run_experiment(RAW_CSV, "raw")
    pca_metrics = run_experiment(PCA_CSV, "pca")

    # ── Final summary table ──
    print("=" * 74)
    print("  SUMMARY — ALL EXPERIMENTS")
    print("=" * 74)
    header = f"{'Features':<8} {'Model':<14} {'Acc':>6} {'Prec':>6} " \
             f"{'Rec':>6} {'F1':>6} {'AUC':>6} {'CV-F1':>12}"
    print(header)
    print("-" * 74)
    for tag, metrics in [("raw", raw_metrics), ("pca", pca_metrics)]:
        for name, m in metrics.items():
            print(f"{tag:<8} {name:<14} {m['accuracy']:6.4f} "
                  f"{m['precision']:6.4f} {m['recall']:6.4f} "
                  f"{m['f1']:6.4f} {m['roc_auc']:6.4f} "
                  f"{m['cv_f1_mean']:.4f}±{m['cv_f1_std']:.4f}")
    print("=" * 74)
    print(f"\nAll results saved to  {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
