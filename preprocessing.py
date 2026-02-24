"""
BraTS 2021 - Preprocessing and Feature Extraction Pipeline
===========================================================
Steps:
  1. Load NIfTI volumes (FLAIR, T1, T1CE, T2, Seg)
  2. Crop to brain region (remove background)
  3. Resize to a uniform shape (128 × 128 × 128)
  4. Normalize each modality (Z-score within brain mask)
  5. Extract hand-crafted features per subject
     - First-order statistics (mean, std, skewness, kurtosis, …)
     - Histogram-based features (percentiles, entropy)
     - Tumor sub-region volume & ratio features
     - Texture features (3-D GLCM contrast, energy, homogeneity, correlation)
  6. Combine features → DataFrame
  7. Apply PCA for dimensionality reduction
  8. Save everything (preprocessed volumes + features + PCA model)
"""

import os
import glob
import warnings
import pickle

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage, stats
from scipy.signal import fftconvolve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────── Configuration ───────────────────
DATASET_DIR   = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "preprocessed")
TARGET_SHAPE  = (128, 128, 128)        # uniform volume size
PCA_VARIANCE  = 0.95                    # retain 95 % variance
MODALITIES    = ["flair", "t1", "t1ce", "t2"]


# ════════════════════════════════════════════════════════
#  1. LOADING
# ════════════════════════════════════════════════════════
def load_subject(subject_dir: str) -> dict:
    """Load all modalities and the segmentation mask for one subject."""
    subject_id = os.path.basename(subject_dir)
    data = {"id": subject_id}
    for mod in MODALITIES:
        path = os.path.join(subject_dir, f"{subject_id}_{mod}.nii.gz")
        img  = nib.load(path)
        data[mod] = img.get_fdata().astype(np.float32)
    seg_path = os.path.join(subject_dir, f"{subject_id}_seg.nii.gz")
    data["seg"] = nib.load(seg_path).get_fdata().astype(np.uint8)
    return data


# ════════════════════════════════════════════════════════
#  2. CROP TO BRAIN REGION
# ════════════════════════════════════════════════════════
def get_brain_bbox(volume: np.ndarray, margin: int = 2) -> tuple:
    """Return slices for the tight bounding box around non-zero voxels."""
    nonzero = np.nonzero(volume > 0)
    if len(nonzero[0]) == 0:
        return tuple(slice(0, s) for s in volume.shape)
    mins = [max(0, n.min() - margin) for n in nonzero]
    maxs = [min(s, n.max() + margin + 1) for s, n in zip(volume.shape, nonzero)]
    return tuple(slice(lo, hi) for lo, hi in zip(mins, maxs))


def crop_to_brain(data: dict) -> dict:
    """Crop all modalities + seg to the brain bounding box (union of all mods)."""
    # Union mask across all modalities
    union = np.zeros_like(data[MODALITIES[0]], dtype=bool)
    for mod in MODALITIES:
        union |= data[mod] > 0
    bbox = get_brain_bbox(union.astype(np.float32))
    for mod in MODALITIES:
        data[mod] = data[mod][bbox]
    data["seg"] = data["seg"][bbox]
    return data


# ════════════════════════════════════════════════════════
#  3. RESIZE
# ════════════════════════════════════════════════════════
def resize_volume(volume: np.ndarray, target: tuple, order: int = 1) -> np.ndarray:
    """Resize a 3-D volume to `target` shape using spline interpolation."""
    factors = [t / s for t, s in zip(target, volume.shape)]
    return ndimage.zoom(volume, factors, order=order)


def resize_all(data: dict) -> dict:
    """Resize modalities (trilinear) and seg (nearest-neighbour) to TARGET_SHAPE."""
    for mod in MODALITIES:
        data[mod] = resize_volume(data[mod], TARGET_SHAPE, order=1)
    data["seg"] = resize_volume(data["seg"].astype(np.float32),
                                TARGET_SHAPE, order=0).astype(np.uint8)
    return data


# ════════════════════════════════════════════════════════
#  4. NORMALIZATION  (Z-score within brain mask)
# ════════════════════════════════════════════════════════
def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize non-zero (brain) voxels; background stays 0."""
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    brain_voxels = volume[mask]
    mu, sigma = brain_voxels.mean(), brain_voxels.std() + 1e-8
    out = np.zeros_like(volume)
    out[mask] = (volume[mask] - mu) / sigma
    return out


def normalize_all(data: dict) -> dict:
    for mod in MODALITIES:
        data[mod] = zscore_normalize(data[mod])
    return data


# ════════════════════════════════════════════════════════
#  5. FEATURE EXTRACTION
# ════════════════════════════════════════════════════════

# ---------- 5a. First-order statistics ----------
def first_order_features(volume: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
    """Compute first-order statistics within a binary mask."""
    voxels = volume[mask > 0]
    if len(voxels) == 0:
        return {f"{prefix}_{k}": 0.0 for k in
                ["mean", "std", "min", "max", "median",
                 "skewness", "kurtosis", "energy", "entropy",
                 "p10", "p25", "p75", "p90", "iqr"]}
    hist, bin_edges = np.histogram(voxels, bins=64, density=True)
    hist = hist / (hist.sum() + 1e-12)
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    return {
        f"{prefix}_mean":     float(np.mean(voxels)),
        f"{prefix}_std":      float(np.std(voxels)),
        f"{prefix}_min":      float(np.min(voxels)),
        f"{prefix}_max":      float(np.max(voxels)),
        f"{prefix}_median":   float(np.median(voxels)),
        f"{prefix}_skewness": float(stats.skew(voxels)),
        f"{prefix}_kurtosis": float(stats.kurtosis(voxels)),
        f"{prefix}_energy":   float(np.sum(voxels ** 2)),
        f"{prefix}_entropy":  float(entropy),
        f"{prefix}_p10":      float(np.percentile(voxels, 10)),
        f"{prefix}_p25":      float(np.percentile(voxels, 25)),
        f"{prefix}_p75":      float(np.percentile(voxels, 75)),
        f"{prefix}_p90":      float(np.percentile(voxels, 90)),
        f"{prefix}_iqr":      float(np.percentile(voxels, 75) - np.percentile(voxels, 25)),
    }


# ---------- 5b. Tumor volume / shape features ----------
def tumor_volume_features(seg: np.ndarray) -> dict:
    """
    BraTS label convention:
      0 = background, 1 = NCR/NET, 2 = ED, 4 = ET
    Derived regions:
      Whole Tumor (WT)  = labels 1 + 2 + 4
      Tumor Core  (TC)  = labels 1 + 4
      Enhancing   (ET)  = label 4
    """
    total_voxels = float(seg.size)
    ncr   = np.sum(seg == 1)
    ed    = np.sum(seg == 2)
    et    = np.sum(seg == 4)
    wt    = ncr + ed + et
    tc    = ncr + et

    feats = {
        "vol_ncr":       float(ncr),
        "vol_ed":        float(ed),
        "vol_et":        float(et),
        "vol_wt":        float(wt),
        "vol_tc":        float(tc),
        "ratio_wt":      float(wt / total_voxels) if total_voxels else 0,
        "ratio_et_wt":   float(et / wt) if wt else 0,
        "ratio_tc_wt":   float(tc / wt) if wt else 0,
        "ratio_ncr_wt":  float(ncr / wt) if wt else 0,
        "ratio_ed_wt":   float(ed / wt) if wt else 0,
    }
    return feats


# ---------- 5c. Lightweight 3-D texture features (GLCM approximation) ----------
def glcm_features_3d(volume: np.ndarray, mask: np.ndarray, prefix: str,
                     levels: int = 32) -> dict:
    """
    Approximate 3-D GLCM features (contrast, energy, homogeneity, correlation)
    by computing a co-occurrence matrix along 3 axial offsets and averaging.
    To keep runtime manageable, the volume is quantised to `levels` bins.
    """
    voxels = volume[mask > 0]
    if len(voxels) == 0:
        return {f"{prefix}_glcm_{k}": 0.0 for k in
                ["contrast", "energy", "homogeneity", "correlation"]}

    # Quantise
    vmin, vmax = voxels.min(), voxels.max()
    if vmax - vmin < 1e-8:
        return {f"{prefix}_glcm_{k}": 0.0 for k in
                ["contrast", "energy", "homogeneity", "correlation"]}
    q = np.clip(((volume - vmin) / (vmax - vmin) * (levels - 1)), 0, levels - 1).astype(np.int32)
    q[mask == 0] = -1  # ignore background

    offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    all_contrast, all_energy, all_homogeneity, all_correlation = [], [], [], []

    for dx, dy, dz in offsets:
        glcm = np.zeros((levels, levels), dtype=np.float64)
        v1 = q[max(dx,0):q.shape[0]-max(-dx,0) or None,
               max(dy,0):q.shape[1]-max(-dy,0) or None,
               max(dz,0):q.shape[2]-max(-dz,0) or None]
        v2 = q[max(-dx,0):q.shape[0]-max(dx,0) or None,
               max(-dy,0):q.shape[1]-max(dy,0) or None,
               max(-dz,0):q.shape[2]-max(dz,0) or None]
        valid = (v1 >= 0) & (v2 >= 0)
        pairs_i = v1[valid]
        pairs_j = v2[valid]

        # Build GLCM efficiently with np.add.at
        np.add.at(glcm, (pairs_i, pairs_j), 1)
        glcm = glcm + glcm.T          # symmetrise
        glcm = glcm / (glcm.sum() + 1e-12)

        i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
        diff = (i_idx - j_idx).astype(np.float64)
        mu_i  = np.sum(i_idx * glcm)
        mu_j  = np.sum(j_idx * glcm)
        sig_i = np.sqrt(np.sum((i_idx - mu_i) ** 2 * glcm) + 1e-12)
        sig_j = np.sqrt(np.sum((j_idx - mu_j) ** 2 * glcm) + 1e-12)

        all_contrast.append(np.sum(diff ** 2 * glcm))
        all_energy.append(np.sum(glcm ** 2))
        all_homogeneity.append(np.sum(glcm / (1 + np.abs(diff))))
        corr_num = np.sum((i_idx - mu_i) * (j_idx - mu_j) * glcm)
        all_correlation.append(corr_num / (sig_i * sig_j + 1e-12))

    return {
        f"{prefix}_glcm_contrast":    float(np.mean(all_contrast)),
        f"{prefix}_glcm_energy":      float(np.mean(all_energy)),
        f"{prefix}_glcm_homogeneity": float(np.mean(all_homogeneity)),
        f"{prefix}_glcm_correlation": float(np.mean(all_correlation)),
    }


# ---------- 5d. Combine all features for one subject ----------
def extract_features(data: dict) -> dict:
    """Return a flat feature dictionary for a single subject."""
    seg  = data["seg"]
    brain_mask = np.zeros_like(seg, dtype=np.uint8)
    for mod in MODALITIES:
        brain_mask |= (data[mod] != 0).astype(np.uint8)
    wt_mask = ((seg == 1) | (seg == 2) | (seg == 4)).astype(np.uint8)

    feats = {"subject_id": data["id"]}

    # Per-modality first-order features (whole brain & tumor region)
    for mod in MODALITIES:
        feats.update(first_order_features(data[mod], brain_mask, prefix=f"{mod}_brain"))
        feats.update(first_order_features(data[mod], wt_mask,   prefix=f"{mod}_tumor"))

    # Per-modality GLCM texture features (tumor region only, for speed)
    for mod in MODALITIES:
        feats.update(glcm_features_3d(data[mod], wt_mask, prefix=f"{mod}_tumor"))

    # Tumor volume / shape features
    feats.update(tumor_volume_features(seg))

    return feats


# ════════════════════════════════════════════════════════
#  6. SAVE PREPROCESSED VOLUME (as compressed .npz)
# ════════════════════════════════════════════════════════
def save_preprocessed(data: dict, out_dir: str):
    """Save the preprocessed 4-modality stack and seg as a compressed .npz."""
    os.makedirs(out_dir, exist_ok=True)
    stack = np.stack([data[m] for m in MODALITIES], axis=0)  # (4, D, H, W)
    np.savez_compressed(
        os.path.join(out_dir, f"{data['id']}.npz"),
        images=stack,
        seg=data["seg"],
    )


# ════════════════════════════════════════════════════════
#  7. PCA  (on the extracted feature matrix)
# ════════════════════════════════════════════════════════
def apply_pca(df: pd.DataFrame, variance: float = PCA_VARIANCE):
    """Standardise features ➜ PCA ➜ return transformed array + fitted objects."""
    feat_cols = [c for c in df.columns if c != "subject_id"]
    X = df[feat_cols].values.astype(np.float64)
    # Replace any NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=variance, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print(f"\n[PCA] Original features : {X.shape[1]}")
    print(f"[PCA] Components kept   : {pca.n_components_}")
    print(f"[PCA] Variance retained : {pca.explained_variance_ratio_.sum():.4f}")

    pca_df = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(X_pca.shape[1])],
    )
    pca_df.insert(0, "subject_id", df["subject_id"].values)
    return pca_df, pca, scaler


# ════════════════════════════════════════════════════════
#  8. VISUALISATION HELPERS
# ════════════════════════════════════════════════════════
def plot_pca_variance(pca: PCA, save_path: str):
    """Plot cumulative explained variance."""
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(cumvar) + 1), pca.explained_variance_ratio_,
            alpha=0.6, label="Individual")
    plt.step(range(1, len(cumvar) + 1), cumvar, where="mid",
             color="red", label="Cumulative")
    plt.axhline(y=PCA_VARIANCE, linestyle="--", color="gray",
                label=f"{PCA_VARIANCE*100:.0f}% threshold")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA – Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved PCA variance plot → {save_path}")


def plot_sample_slices(data: dict, save_path: str):
    """Plot the middle axial slice for each modality + seg."""
    mid = data[MODALITIES[0]].shape[2] // 2
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, mod in zip(axes[:4], MODALITIES):
        ax.imshow(data[mod][:, :, mid].T, cmap="gray", origin="lower")
        ax.set_title(mod.upper())
        ax.axis("off")
    axes[4].imshow(data["seg"][:, :, mid].T, cmap="nipy_spectral", origin="lower")
    axes[4].set_title("SEG")
    axes[4].axis("off")
    plt.suptitle(data["id"], fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subjects = sorted(glob.glob(os.path.join(DATASET_DIR, "BraTS2021_*")))
    print(f"Found {len(subjects)} subjects in {DATASET_DIR}\n")

    all_features = []
    sample_dir = os.path.join(OUTPUT_DIR, "sample_slices")
    os.makedirs(sample_dir, exist_ok=True)

    for i, subj_dir in enumerate(tqdm(subjects, desc="Processing")):
        sid = os.path.basename(subj_dir)
        try:
            # 1) Load
            data = load_subject(subj_dir)

            # 2) Crop
            data = crop_to_brain(data)

            # 3) Resize
            data = resize_all(data)

            # 4) Normalize
            data = normalize_all(data)

            # 5) Feature extraction
            feats = extract_features(data)
            all_features.append(feats)

            # 6) Save preprocessed volume
            save_preprocessed(data, os.path.join(OUTPUT_DIR, "volumes"))

            # Save a sample visualisation for the first 5 subjects
            if i < 5:
                plot_sample_slices(
                    data,
                    os.path.join(sample_dir, f"{sid}_slices.png"),
                )

        except Exception as e:
            print(f"\n[ERROR] {sid}: {e}")
            continue

    # ── Build feature DataFrame ──
    df = pd.DataFrame(all_features)
    feat_csv = os.path.join(OUTPUT_DIR, "features_raw.csv")
    df.to_csv(feat_csv, index=False)
    print(f"\n[Features] Saved raw features → {feat_csv}  ({df.shape})")

    # ── PCA ──
    pca_df, pca_model, scaler = apply_pca(df)
    pca_csv = os.path.join(OUTPUT_DIR, "features_pca.csv")
    pca_df.to_csv(pca_csv, index=False)
    print(f"[PCA]      Saved PCA features → {pca_csv}  ({pca_df.shape})")

    # Save fitted PCA + scaler for later reuse
    with open(os.path.join(OUTPUT_DIR, "pca_pipeline.pkl"), "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca_model}, f)
    print("[PCA]      Saved PCA pipeline  → pca_pipeline.pkl")

    # ── Visualise PCA ──
    plot_pca_variance(pca_model, os.path.join(OUTPUT_DIR, "pca_variance.png"))

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  PREPROCESSING & FEATURE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Subjects processed   : {len(all_features)}/{len(subjects)}")
    print(f"  Raw features         : {df.shape[1] - 1}")
    print(f"  PCA components       : {pca_model.n_components_}")
    print(f"  Outputs in           : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
