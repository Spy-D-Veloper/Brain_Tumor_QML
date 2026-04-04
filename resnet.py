"""
Deep Learning Model: ResNet50 for Brain Tumor Classification
=============================================================
Uses pre-trained ResNet50 with transfer learning:
  - Fine-tunes on hand-crafted features from BraTS MRI
  - Binary classification of tumor aggressiveness
  - Compares with classical ML (SVM) and quantum ML (VQC)
  
Pipeline:
  1. Load preprocessed features (PCA-reduced)
  2. Create synthetic images from feature vectors
  3. Train ResNet50 with cross-validation
  4. Evaluate on test set
  5. Compare performance with QML and classical ML
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# ══════════════════════════════════════════════════════════════
#  Configuration & Paths
# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(__file__)
RAW_CSV = os.path.join(BASE_DIR, "preprocessed", "features_raw.csv")
PCA_CSV = os.path.join(BASE_DIR, "preprocessed", "features_pca.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model parameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
RANDOM_STATE = 42
TEST_SIZE = 0.20
EARLY_STOP_PATIENCE = 8


# ══════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════
def load_and_prepare_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed features and create binary labels.
    
    Args:
        csv_path: Path to features CSV file
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Binary labels (n_samples,)
    """
    df = pd.read_csv(csv_path)
    
    # Merge with raw CSV to get the ratio_et_wt column for labeling
    if "ratio_et_wt" not in df.columns:
        raw_df = pd.read_csv(RAW_CSV)[["subject_id", "ratio_et_wt"]]
        df = df.merge(raw_df, on="subject_id", how="left")
    
    # Create binary label from median ratio
    median_ratio = df["ratio_et_wt"].median()
    df["label"] = (df["ratio_et_wt"] >= median_ratio).astype(int)
    
    # Extract features and labels
    drop_cols = ["subject_id", "ratio_et_wt", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols and not c.startswith("vol_")]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    
    # Clean NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"[Data] Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"[Data] Class balance: {dict(pd.Series(y).value_counts())}")
    
    return X, y


# ══════════════════════════════════════════════════════════════
#  Feature Engineering: Convert to Images
# ══════════════════════════════════════════════════════════════
def features_to_images(X: np.ndarray, target_shape: Tuple = (64, 64, 3)) -> np.ndarray:
    """
    Convert feature vectors to images.
    Strategy: Tile features into a grayscale image and add RGB channels.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        target_shape: Target image shape (height, width, channels)
    
    Returns:
        Images array (n_samples, height, width, channels)
    """
    n_samples = X.shape[0]
    height, width, channels = target_shape
    
    images = np.zeros((n_samples, height, width, channels), dtype=np.float32)
    
    for i, features in enumerate(X):
        # Normalize features to [0, 1]
        f_min = features.min()
        f_max = features.max()
        if f_max > f_min:
            features_norm = (features - f_min) / (f_max - f_min)
        else:
            features_norm = features
        
        # Tile features into image
        n_features = len(features_norm)
        tiles_needed = (height * width + n_features - 1) // n_features
        
        tiled = np.tile(features_norm, tiles_needed)[:height * width]
        image_2d = tiled.reshape(height, width)
        
        # Create RGB image
        for c in range(min(channels, 3)):
            images[i, :, :, c] = image_2d
    
    return images


# ══════════════════════════════════════════════════════════════
#  Model Architecture
# ══════════════════════════════════════════════════════════════
def create_resnet_model(input_shape: Tuple = (64, 64, 3)) -> Tuple[models.Model, ResNet50]:
    """
    Create a ResNet50-based model for binary classification.
    Uses transfer learning with fine-tuning.
    
    Args:
        input_shape: Input image shape (height, width, channels)
    
    Returns:
        (Compiled model, Base ResNet50 model)
    """
    # Load pre-trained ResNet50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Create model
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Preprocessing: normalize to ImageNet range
        layers.Lambda(lambda x: tf.keras.applications.resnet50.preprocess_input(x)),
        
        # ResNet50 backbone
        base_model,
        
        # Global pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense head for binary classification
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    return model, base_model


# ══════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════
def train_resnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = True
) -> models.Model:
    """
    Train ResNet50 model with transfer learning.
    
    Uses two-phase training:
    1. Train head layers with frozen backbone
    2. Fine-tune last ResNet blocks with low learning rate
    """
    print("\n[Training Phase 1] Training head layers...")
    
    model, base_model = create_resnet_model()
    
    # Phase 1: Train with frozen backbone
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=0)
    ]
    
    history1 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS // 2,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1 if verbose else 0
    )
    
    # Phase 2: Fine-tune with unfrozen backbone
    print("\n[Training Phase 2] Fine-tuning backbone layers...")
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    history2 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS // 2,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1 if verbose else 0
    )
    
    return model


# ══════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════
def evaluate_model(
    model: models.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Evaluate model on test set.
    
    Returns:
        (metrics dict, predictions, probabilities)
    """
    # Get predictions
    y_proba = model.predict(X_test, verbose=0)[:, 0]
    y_pred = (y_proba > 0.5).astype(int).flatten()
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    
    metrics = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(auc)
    }
    
    if verbose:
        print("\n[ResNet50 Results]")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
    
    return metrics, y_pred, y_proba


# ══════════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════════
def main():
    print("="*70)
    print("   Deep Learning: ResNet50 for Brain Tumor Classification")
    print("="*70)
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    X, y = load_and_prepare_data(PCA_CSV)
    
    # 2. Train/test split
    print("[2/5] Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    # Further split train into train/val (80/20 of train)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE
    )
    
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # 3. Normalize features
    print("[3/5] Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 4. Convert to images
    print("[4/5] Converting features to images...")
    X_train_img = features_to_images(X_train, target_shape=(64, 64, 3))
    X_val_img = features_to_images(X_val, target_shape=(64, 64, 3))
    X_test_img = features_to_images(X_test, target_shape=(64, 64, 3))
    
    print(f"  Image shape: {X_train_img.shape}")
    
    # 5. Train model
    print("[5/5] Training ResNet50...")
    model = train_resnet(X_train_img, y_train, X_val_img, y_val, verbose=True)
    
    # 6. Evaluate
    print("\n" + "="*70)
    print("   EVALUATION RESULTS")
    print("="*70)
    
    metrics, y_pred, y_proba = evaluate_model(model, X_test_img, y_test, verbose=True)
    
    # 7. Save results
    print("\n[Export] Saving results to JSON...")
    results_dict = {
        'model_type': 'ResNet50 (Transfer Learning)',
        'architecture': 'ResNet50 + Dense Head',
        'input_shape': [64, 64, 3],
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'metrics': metrics,
        'best_accuracy': metrics['accuracy'],
    }
    
    with open(os.path.join(RESULTS_DIR, "resnet_metrics.json"), 'w') as f:
        json.dump(results_dict, f, indent=2)
    print("  Saved: resnet_metrics.json")
    


if __name__ == "__main__":
    main()