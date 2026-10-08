#!/usr/bin/env python3
"""
DA-LGBM (ADANN_LightGBM) feature-group ablation under LOSO.

We keep the original 190D hand-crafted feature vector for both ADANN and LightGBM
and apply zero-masks to specific feature groups:
  - A: time-domain (18 dims per channel, 5 channels -> 90 dims)
  - B: frequency-domain (12 dims per channel, 5 channels -> 60 dims)
  - C: wavelet (8 dims per channel, 5 channels -> 40 dims)

This script:
  - loads the LOSO data from csv (re-using pipeline.load_data logic)
  - for each feature_mode in {full, A, B, C, AB, AC, BC}
      - for each subject fold (leave-one-subject-out)
          - trains a fresh ADANN+LightGBM hybrid model using the existing
            AdannLightgbmModelCreator.train_model() helper
          - evaluates on that subject's test set (ensemble prediction)
  - saves a single CSV: outputs/DA_LGBM_ablation_loso_results.csv
    with columns: feature_mode, subject_id, accuracy, macro_f1
"""

import argparse
import os
import glob
import re
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from src.training.train_adann_lightgbm import AdannLightgbmModelCreator


FEATURE_MODES = ["full", "A", "B", "C", "AB", "AC", "BC"]

# Keep key constants consistent with the main pipeline
SEED = 42
SEQUENCE_LENGTH = 100
N_FEATURES = 5


def load_data(csv_dir: str):
    """
    Load and preprocess data, extracting subject IDs for LOSO.
    Copied from src.training.pipeline.load_data to avoid importing
    the full pipeline (which depends on missing evaluation modules).
    """
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    all_data, all_labels, all_subjects = [], [], []

    for csv_file in csv_files:
        try:
            with open(csv_file, "r") as f:
                lines = f.readlines()

            data = []
            for line in lines:
                if "timestamp" in line or line.startswith("#"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    try:
                        values = [float(parts[i]) for i in range(1, 6)]
                        data.append(values)
                    except ValueError:
                        continue

            if data:
                data = np.array(data)
                # Resample to fixed length
                if len(data) != SEQUENCE_LENGTH:
                    indices = np.linspace(0, len(data) - 1, SEQUENCE_LENGTH)
                    resampled = np.zeros((SEQUENCE_LENGTH, N_FEATURES))
                    for i in range(N_FEATURES):
                        resampled[:, i] = np.interp(indices, range(len(data)), data[:, i])
                    data = resampled

                # Extract gesture ID
                gesture_id_match = re.search(r"gesture_(\d+)", csv_file)
                if not gesture_id_match:
                    continue
                gesture_id = int(gesture_id_match.group(1))

                # Extract subject ID
                subject_id_match = re.search(r"user_(\d+)", csv_file)
                if subject_id_match:
                    subject_id = int(subject_id_match.group(1))
                else:
                    subject_id = -1

                # Convert to int16 to match main pipeline (memory/perf)
                data = np.round(data).astype(np.int16)
                all_data.append(data)
                all_labels.append(gesture_id)
                all_subjects.append(subject_id)
        except Exception as e:
            print(f"Warning: Could not process file {csv_file}. Error: {e}")
            continue

    return np.array(all_data), np.array(all_labels), np.array(all_subjects)


def build_feature_mask(mode: str) -> np.ndarray:
    """
    Build a boolean mask over the 190D feature vector.

    Indices layout (per channel, total 5 channels):
      - time-domain: 18 dims      -> A
      - frequency-domain: 12 dims -> B
      - wavelet: 8 dims           -> C
    Per-channel length: 38. The extractor appends features channel by
    channel, so the groups are interleaved across the 190D vector.
    """
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature_mode: {mode}")

    n_total = 190
    mask = np.zeros(n_total, dtype=bool)

    A_parts, B_parts, C_parts = [], [], []
    for ch in range(N_FEATURES):
        base = ch * 38
        A_parts.append(np.arange(base, base + 18))
        B_parts.append(np.arange(base + 18, base + 30))
        C_parts.append(np.arange(base + 30, base + 38))

    A_idx = np.concatenate(A_parts)  # 5 * 18
    B_idx = np.concatenate(B_parts)  # 5 * 12
    C_idx = np.concatenate(C_parts)  # 5 * 8

    if mode == "full":
        mask[:] = True
    elif mode == "A":
        mask[A_idx] = True
    elif mode == "B":
        mask[B_idx] = True
    elif mode == "C":
        mask[C_idx] = True
    elif mode == "AB":
        mask[A_idx] = True
        mask[B_idx] = True
    elif mode == "AC":
        mask[A_idx] = True
        mask[C_idx] = True
    elif mode == "BC":
        mask[B_idx] = True
        mask[C_idx] = True

    return mask


def patch_hybrid_extractor_with_mask(creator: AdannLightgbmModelCreator,
                                     feature_mask: np.ndarray) -> None:
    """
    Apply a zero-mask to the 190D hand-crafted features used by both ADANN
    and LightGBM inside the existing HybridFeatureExtractor.

    This does NOT change any model architecture or pipeline code; it only
    wraps the underlying extract_comprehensive_features() at runtime.
    """
    extractor = creator.hybrid_extractor.enhanced_extractor
    original_fn = extractor.extract_comprehensive_features

    def wrapped_extract(sample):
        feats = original_fn(sample)
        # Safety: ensure length 190 before masking
        if feats.shape[0] != feature_mask.shape[0]:
            raise ValueError(
                f"Expected feature length {feature_mask.shape[0]}, "
                f"got {feats.shape[0]}"
            )
        feats = feats.copy()
        feats[~feature_mask] = 0.0
        return feats

    extractor.extract_comprehensive_features = wrapped_extract


def macro_f1_from_cm(cm: np.ndarray) -> float:
    """Compute macro-F1 from a confusion matrix (shape [n_class, n_class])."""
    n_class = cm.shape[0]
    f1s: List[float] = []
    for i in range(n_class):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = (2 * tp + fp + fn)
        if denom == 0:
            f1 = np.nan
        else:
            f1 = 2 * tp / denom
        f1s.append(f1)
    return float(np.nanmean(f1s))


def get_default_hyperparams() -> Dict:
    """
    A fixed, reasonable hyperparameter set for ADANN+LightGBM.
    We keep this constant across all feature modes so that
    comparisons reflect only the feature groups.
    """
    return {
        # ADANN branch
        "adann_learning_rate": 1e-3,
        "adann_feature_size": 64,
        "adann_epochs": 120,
        "gesture_loss_weight": 1.0,
        "domain_loss_weight": 1.0,
        "batch_size": 32,
        # LightGBM branch (moderate complexity)
        "lgb_num_leaves": 31,
        "lgb_learning_rate": 0.10,
        "lgb_feature_fraction": 0.80,
        "lgb_bagging_fraction": 0.80,
        "lgb_min_child_samples": 20,
        "lgb_n_estimators": 120,
        "lgb_max_depth": 8,
        # Ensemble
        "ensemble_adann_weight": 0.5,
    }


def run_loso_ablation(csv_dir: str,
                      output_path: str,
                      random_seed: int = SEED) -> None:
    """
    Run LOSO DA-LGBM ablation for all feature modes and save a single CSV.
    """
    np.random.seed(random_seed)

    print(f"Loading data from: {csv_dir}")
    X, y, subjects = load_data(csv_dir)

    unique_subjects = np.unique(subjects[subjects != -1])
    if len(unique_subjects) < 2:
        raise RuntimeError("LOSO requires at least 2 subjects.")

    print(f"Found {len(X)} samples, {len(unique_subjects)} subjects.")

    records = []

    for mode in FEATURE_MODES:
        print(f"\n===== Feature mode: {mode} =====")
        feature_mask = build_feature_mask(mode)

        for test_subj in unique_subjects:
            print(f"\n--- LOSO fold: test subject {test_subj} ---")
            test_mask = (subjects == test_subj)
            train_mask = ~test_mask

            X_train = X[train_mask]
            y_train = y[train_mask]
            subj_train = subjects[train_mask]

            X_test = X[test_mask]
            y_test = y[test_mask]

            if X_test.shape[0] == 0:
                print(f"  [Skip] No samples for subject {test_subj}")
                continue

            # Simple train/val split within the training set (e.g., 80/20)
            rng = np.random.RandomState(random_seed)
            idx_all = np.arange(X_train.shape[0])
            rng.shuffle(idx_all)
            n_total = len(idx_all)
            n_val = max(1, int(0.2 * n_total))
            val_idx = idx_all[:n_val]
            train_idx = idx_all[n_val:]

            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            subj_train_fold = subj_train[train_idx]

            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]
            subj_val_fold = subj_train[val_idx]

            # Create model creator and apply feature mask inside hybrid extractor
            creator = AdannLightgbmModelCreator()
            patch_hybrid_extractor_with_mask(creator, feature_mask)

            hyperparams = get_default_hyperparams()
            wrapper = creator.create_model(hyperparams, arduino_mode=False)

            # Train hybrid model using existing helper (ADANN+LGB)
            hybrid_model_dict = wrapper.hybrid_model
            trained_model, val_acc, _ = creator.train_model(
                hybrid_model_dict,
                X_train_fold, y_train_fold, subj_train_fold,
                X_val_fold, y_val_fold, subj_val_fold,
                hyperparams,
                return_history=True,
            )

            print(f"Validation accuracy (ensemble) for subject {test_subj}: {val_acc:.4f}")

            # Evaluate on LOSO test subject using full hybrid (ADANN + LightGBM) ensemble
            y_pred = creator.predict(trained_model, X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
            mf1 = macro_f1_from_cm(cm)

            print(f"Test accuracy (subject {test_subj}, mode {mode}): {acc:.4f}, macro-F1: {mf1:.4f}")

            records.append(
                {
                    "feature_mode": mode,
                    "subject_id": int(test_subj),
                    "accuracy": float(acc),
                    "macro_f1": float(mf1),
                }
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame.from_records(records)
    df.to_csv(output_path, index=False)
    print(f"\nSaved LOSO ablation results to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DA-LGBM (ADANN_LightGBM) 190D feature-group LOSO ablation"
    )
    parser.add_argument(
        "--csv_dir",
        default="datasets/gesture_csv",
        help="Directory containing gesture CSV files (same format as pipeline)",
    )
    parser.add_argument(
        "--output_csv",
        default="outputs/DA_LGBM_ablation/da_lgbm_loso_ablation_results.csv",
        help="Path to save LOSO ablation CSV",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_loso_ablation(args.csv_dir, args.output_csv)
