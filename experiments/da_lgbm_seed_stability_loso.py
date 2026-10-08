#!/usr/bin/env python3
"""
DA-LGBM (ADANN_LightGBM) seed stability under LOSO with fold-wise Optuna.

Pipeline:
  1) For each LOSO fold (test subject):
     - Run Optuna to search best hyperparameters on train/val split
     - Hyperparameter search includes data augmentation parameters
  2) For each seed:
     - Train each fold using that fold's best hyperparameters
     - Evaluate on fold test subject
  3) Save detailed records for table-ready analysis

Outputs (in output_dir):
  1) fold_best_hyperparams.csv
  2) fold_best_hyperparams.json
  3) seed_stability_raw.csv
  4) seed_stability_by_seed.csv
  5) seed_stability_summary.json
"""

import argparse
import glob
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from tsaug import AddNoise, TimeWarp

# Ensure project root is importable when running as:
#   python3 experiments/da_lgbm_seed_stability_loso.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_adann_lightgbm import AdannLightgbmModelCreator


# Keep key constants aligned with the training pipeline
SEQUENCE_LENGTH = 100
N_FEATURES = 5


def set_all_seeds(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_seeds(seeds_str: str) -> List[int]:
    """Parse comma-separated seeds into a validated integer list."""
    seeds = []
    for token in seeds_str.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds provided.")
    return seeds


def _to_basic_type(obj):
    """Convert NumPy scalar/array into JSON/CSV safe basic types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _sample_std(values: np.ndarray) -> float:
    """Sample standard deviation, returning 0 for a single observation."""
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _t_critical_975(df: int) -> float:
    """Two-sided 95% t critical value; normal approximation for large df."""
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
        40: 2.021,
        60: 2.000,
        120: 1.980,
    }
    if df in table:
        return table[df]
    if df < 25:
        return table[max(k for k in table if k < df)]
    if df < 30:
        return table[25]
    if df < 40:
        return table[30]
    if df < 60:
        return table[40]
    if df < 120:
        return table[60]
    return 1.960


def _ci95(values: np.ndarray) -> Dict[str, float]:
    """Two-sided 95% t confidence interval for a mean."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) <= 1:
        return {
            "mean": mean,
            "std": 0.0,
            "ci95_half_width": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
        }
    std = _sample_std(values)
    half_width = float(_t_critical_975(len(values) - 1) * std / np.sqrt(len(values)))
    return {
        "mean": mean,
        "std": std,
        "ci95_half_width": half_width,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def load_data(csv_dir: str):
    """
    Load and preprocess data, extracting subject IDs for LOSO.
    Copied from training pipeline logic for standalone execution.
    """
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
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
                if len(data) != SEQUENCE_LENGTH:
                    indices = np.linspace(0, len(data) - 1, SEQUENCE_LENGTH)
                    resampled = np.zeros((SEQUENCE_LENGTH, N_FEATURES))
                    for i in range(N_FEATURES):
                        resampled[:, i] = np.interp(indices, range(len(data)), data[:, i])
                    data = resampled

                gesture_id_match = re.search(r"gesture_(\d+)", csv_file)
                if not gesture_id_match:
                    continue
                gesture_id = int(gesture_id_match.group(1))

                subject_id_match = re.search(r"user_(\d+)", csv_file)
                if subject_id_match:
                    subject_id = int(subject_id_match.group(1))
                else:
                    subject_id = -1

                data = np.round(data).astype(np.int16)
                all_data.append(data)
                all_labels.append(gesture_id)
                all_subjects.append(subject_id)
        except Exception as exc:
            print(f"Warning: Could not process file {csv_file}. Error: {exc}")
            continue

    return np.array(all_data), np.array(all_labels), np.array(all_subjects)


def apply_sample_normalization(X: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply deterministic per-window normalization.

    These modes use only statistics from each individual input window, so they
    are available at deployment time and do not leak subject labels or test
    labels into training.
    """
    mode = (mode or "none").lower()
    if mode == "none":
        return X

    X_norm = X.astype(np.float32, copy=True)
    eps = 1e-6

    if mode == "center":
        X_norm = X_norm - np.mean(X_norm, axis=1, keepdims=True)
    elif mode == "baseline":
        baseline_len = min(10, X_norm.shape[1])
        baseline = np.mean(X_norm[:, :baseline_len, :], axis=1, keepdims=True)
        X_norm = X_norm - baseline
    elif mode == "zscore":
        mean = np.mean(X_norm, axis=1, keepdims=True)
        std = np.std(X_norm, axis=1, keepdims=True)
        X_norm = (X_norm - mean) / np.maximum(std, eps)
    elif mode == "robust_zscore":
        median = np.median(X_norm, axis=1, keepdims=True)
        q1 = np.percentile(X_norm, 25, axis=1, keepdims=True)
        q3 = np.percentile(X_norm, 75, axis=1, keepdims=True)
        robust_scale = (q3 - q1) / 1.349
        X_norm = (X_norm - median) / np.maximum(robust_scale, eps)
    else:
        raise ValueError(f"Unknown sample normalization mode: {mode}")

    if mode in ("center", "baseline"):
        return np.round(X_norm).astype(np.int16)
    return X_norm.astype(np.float32)


def macro_f1_from_cm(cm: np.ndarray) -> float:
    """Compute macro-F1 from a confusion matrix (shape [n_class, n_class])."""
    f1s: List[float] = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        if denom == 0:
            f1s.append(np.nan)
        else:
            f1s.append(2 * tp / denom)
    return float(np.nanmean(f1s))


def augment_data_with_subjects(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    augment_params: Dict,
    seed: int,
):
    """
    Apply augmentation and keep subject labels aligned for ADANN domain branch.

    Return:
      X_final, y_final, subjects_final
    """
    augment_factor = int(augment_params.get("augment_factor", 0))
    class_augment_extra_factor = int(augment_params.get("class_augment_extra_factor", 0))
    class_augment_labels_raw = augment_params.get("class_augment_labels", [])
    if isinstance(class_augment_labels_raw, str):
        class_augment_labels = [
            int(token.strip())
            for token in class_augment_labels_raw.split(",")
            if token.strip()
        ]
    else:
        class_augment_labels = [int(label) for label in class_augment_labels_raw]
    if augment_factor <= 0 and (class_augment_extra_factor <= 0 or not class_augment_labels):
        return X_train, y_train, subjects_train

    jitter_noise_level = float(augment_params.get("jitter_noise_level", 0.005))
    time_warp_max_speed = float(augment_params.get("time_warp_max_speed", 2.0))
    scale_range = augment_params.get("scale_range")
    if scale_range is not None and "scale_min" not in augment_params and "scale_max" not in augment_params:
        scale_min = float(scale_range[0])
        scale_max = float(scale_range[1])
    else:
        scale_min = float(augment_params.get("scale_min", 0.98))
        scale_max = float(augment_params.get("scale_max", 1.02))
    augment_prob = float(augment_params.get("augment_prob", 0.5))
    channel_augment_prob = float(augment_params.get("channel_augment_prob", 0.0))
    channel_scale_min = float(augment_params.get("channel_scale_min", 1.0))
    channel_scale_max = float(augment_params.get("channel_scale_max", 1.0))
    channel_shift_level = float(augment_params.get("channel_shift_level", 0.0))
    channel_drift_level = float(augment_params.get("channel_drift_level", 0.0))

    X_parts = []
    y_parts = []
    subj_parts = []
    if augment_factor > 0:
        X_parts.append(np.repeat(X_train, augment_factor, axis=0))
        y_parts.append(np.repeat(y_train, augment_factor, axis=0))
        subj_parts.append(np.repeat(subjects_train, augment_factor, axis=0))
    if class_augment_extra_factor > 0 and class_augment_labels:
        hard_class_mask = np.isin(y_train, np.asarray(class_augment_labels))
        if np.any(hard_class_mask):
            X_parts.append(np.repeat(X_train[hard_class_mask], class_augment_extra_factor, axis=0))
            y_parts.append(np.repeat(y_train[hard_class_mask], class_augment_extra_factor, axis=0))
            subj_parts.append(np.repeat(subjects_train[hard_class_mask], class_augment_extra_factor, axis=0))

    X_to_augment = np.vstack(X_parts)
    y_to_augment = np.concatenate(y_parts)
    subj_to_augment = np.concatenate(subj_parts)

    augmenter = (
        AddNoise(scale=jitter_noise_level) @ augment_prob
        + TimeWarp(n_speed_change=5, max_speed_ratio=time_warp_max_speed) @ augment_prob
    )
    X_augmented = augmenter.augment(X_to_augment)

    # deterministic per call
    rng = np.random.RandomState(seed)
    feature_std = np.std(X_train.astype(float).reshape(-1, X_train.shape[-1]), axis=0)
    feature_std = np.where(feature_std < 1.0, 1.0, feature_std)
    drift_axis = np.linspace(-0.5, 0.5, X_augmented.shape[1])[:, None]
    for i in range(X_augmented.shape[0]):
        if rng.rand() < augment_prob:
            scale_factor = rng.uniform(scale_min, scale_max)
            X_augmented[i] = X_augmented[i] * scale_factor
        if channel_augment_prob > 0 and rng.rand() < channel_augment_prob:
            channel_scale = rng.uniform(channel_scale_min, channel_scale_max, size=X_augmented.shape[2])
            X_augmented[i] = X_augmented[i] * channel_scale
        if channel_shift_level > 0 and rng.rand() < channel_augment_prob:
            channel_shift = rng.normal(0.0, channel_shift_level * feature_std, size=X_augmented.shape[2])
            X_augmented[i] = X_augmented[i] + channel_shift
        if channel_drift_level > 0 and rng.rand() < channel_augment_prob:
            drift_scale = rng.normal(0.0, channel_drift_level * feature_std, size=(1, X_augmented.shape[2]))
            X_augmented[i] = X_augmented[i] + drift_axis * drift_scale

    X_final = np.vstack([X_train, X_augmented])
    y_final = np.concatenate([y_train, y_to_augment])
    subj_final = np.concatenate([subjects_train, subj_to_augment])

    # keep same dtype style as main pipeline
    if augment_params.get("preserve_float", False):
        X_final = X_final.astype(np.float32)
    else:
        X_final = np.round(X_final).astype(np.int16)
    return X_final, y_final, subj_final


def suggest_hyperparams(
    trial: optuna.trial.Trial,
    restrict_augmentation: bool = False,
    augmentation_profile: str = "search",
) -> Dict:
    """Optuna search space for ADANN_LightGBM + augmentation."""
    if restrict_augmentation:
        augmentation_profile = "restricted"

    channel_params = {
        "channel_augment_prob": 0.0,
        "channel_scale_min": 1.0,
        "channel_scale_max": 1.0,
        "channel_shift_level": 0.0,
        "channel_drift_level": 0.0,
    }

    if augmentation_profile == "restricted":
        augment_factor = trial.suggest_int("augment_factor", 0, 1)
        augment_prob = trial.suggest_float("augment_prob", 0.2, 0.5)
        jitter_noise_level = trial.suggest_float("jitter_noise_level", 0.001, 0.01, log=True)
        time_warp_max_speed = trial.suggest_float("time_warp_max_speed", 1.2, 2.2)
        scale_min = trial.suggest_float("scale_min", 0.97, 1.0)
        scale_max = trial.suggest_float("scale_max", 1.0, 1.03)
    elif augmentation_profile == "legacy_strong":
        augment_factor = trial.suggest_int("augment_factor", 1, 3)
        augment_prob = trial.suggest_float("augment_prob", 0.3, 0.8)
        jitter_noise_level = trial.suggest_float("jitter_noise_level", 0.005, 0.02, log=True)
        time_warp_max_speed = trial.suggest_float("time_warp_max_speed", 1.5, 3.0)
        scale_min = trial.suggest_float("scale_min", 0.90, 0.98)
        scale_max = trial.suggest_float("scale_max", 1.02, 1.10)
    elif augmentation_profile == "hardware_shift":
        augment_factor = trial.suggest_int("augment_factor", 1, 3)
        augment_prob = trial.suggest_float("augment_prob", 0.25, 0.65)
        jitter_noise_level = trial.suggest_float("jitter_noise_level", 0.002, 0.015, log=True)
        time_warp_max_speed = trial.suggest_float("time_warp_max_speed", 1.2, 2.5)
        scale_min = trial.suggest_float("scale_min", 0.94, 0.99)
        scale_max = trial.suggest_float("scale_max", 1.01, 1.06)
        channel_params = {
            "channel_augment_prob": trial.suggest_float("channel_augment_prob", 0.3, 0.8),
            "channel_scale_min": trial.suggest_float("channel_scale_min", 0.90, 0.98),
            "channel_scale_max": trial.suggest_float("channel_scale_max", 1.02, 1.12),
            "channel_shift_level": trial.suggest_float("channel_shift_level", 0.0, 0.08),
            "channel_drift_level": trial.suggest_float("channel_drift_level", 0.0, 0.08),
        }
    else:
        augment_factor = trial.suggest_int("augment_factor", 0, 2)
        augment_prob = trial.suggest_float("augment_prob", 0.2, 0.8)
        jitter_noise_level = trial.suggest_float("jitter_noise_level", 0.001, 0.02, log=True)
        time_warp_max_speed = trial.suggest_float("time_warp_max_speed", 1.2, 3.0)
        scale_min = trial.suggest_float("scale_min", 0.95, 1.0)
        scale_max = trial.suggest_float("scale_max", 1.0, 1.05)

    return {
        # ADANN
        "adann_learning_rate": trial.suggest_float("adann_learning_rate", 1e-4, 1e-2, log=True),
        "adann_feature_size": trial.suggest_int("adann_feature_size", 32, 128, step=16),
        "adann_dropout": trial.suggest_float("adann_dropout", 0.2, 0.5),
        "adann_classifier_dropout": trial.suggest_float("adann_classifier_dropout", 0.1, 0.4),
        "adann_epochs": trial.suggest_int("adann_epochs", 80, 180),
        "gesture_loss_weight": trial.suggest_float("gesture_loss_weight", 0.5, 2.0),
        "domain_loss_weight": trial.suggest_float("domain_loss_weight", 0.5, 2.0),
        "grl_gamma": trial.suggest_float("grl_gamma", 6.0, 14.0),
        "grl_max": trial.suggest_float("grl_max", 0.90, 0.99),
        "class_balanced_batches": False,
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        # LightGBM
        "lgb_num_leaves": trial.suggest_int("lgb_num_leaves", 10, 100),
        "lgb_learning_rate": trial.suggest_float("lgb_learning_rate", 0.01, 0.3),
        "lgb_feature_fraction": trial.suggest_float("lgb_feature_fraction", 0.4, 1.0),
        "lgb_bagging_fraction": trial.suggest_float("lgb_bagging_fraction", 0.4, 1.0),
        "lgb_min_child_samples": trial.suggest_int("lgb_min_child_samples", 5, 100),
        "lgb_n_estimators": trial.suggest_int("lgb_n_estimators", 50, 500),
        "lgb_max_depth": trial.suggest_int("lgb_max_depth", 3, 15),
        # Ensemble
        "ensemble_adann_weight": trial.suggest_float("ensemble_adann_weight", 0.3, 0.7),
        "auto_tune_ensemble_weight": False,
        # Augmentation
        "augment_factor": augment_factor,
        "jitter_noise_level": jitter_noise_level,
        "time_warp_max_speed": time_warp_max_speed,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "augment_prob": augment_prob,
        **channel_params,
    }


def split_train_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subj_train: np.ndarray,
    split_seed: int,
    val_ratio: float,
    group_aware: bool = False,
):
    """Split train/val inside a LOSO fold.

    For final LOSO model selection, prefer group-aware validation so that the
    validation subject(s) are unseen during training, matching the test protocol
    better than a sample-level random split.
    """
    if group_aware:
        unique_groups = np.unique(subj_train[subj_train != -1])
        if len(unique_groups) >= 2:
            n_splits = min(max(2, int(round(1.0 / max(val_ratio, 1e-6)))), len(unique_groups))
            try:
                sgkf = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=split_seed,
                )
                tr_idx, va_idx = next(sgkf.split(X_train, y_train, groups=subj_train))
                return (
                    X_train[tr_idx],
                    y_train[tr_idx],
                    subj_train[tr_idx],
                    X_train[va_idx],
                    y_train[va_idx],
                    subj_train[va_idx],
                )
            except Exception as exc:
                print(f"⚠️ Group-aware validation fallback due to: {exc}")

    X_tr, X_val, y_tr, y_val, subj_tr, subj_val = train_test_split(
        X_train,
        y_train,
        subj_train,
        test_size=val_ratio,
        random_state=split_seed,
        stratify=y_train,
    )
    return X_tr, y_tr, subj_tr, X_val, y_val, subj_val


def build_group_cv_splits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subj_train: np.ndarray,
    n_splits: int,
    random_seed: int,
    fallback_val_ratio: float = 0.2,
):
    """
    Build group-aware CV splits for Optuna objective.

    Preferred: StratifiedGroupKFold (group by subject).
    Fallback: single stratified train/val split.
    """
    unique_groups = np.unique(subj_train)
    if len(unique_groups) >= 2:
        eff_splits = min(max(2, n_splits), len(unique_groups))
        try:
            sgkf = StratifiedGroupKFold(
                n_splits=eff_splits,
                shuffle=True,
                random_state=random_seed,
            )
            splits = list(sgkf.split(X_train, y_train, groups=subj_train))
            if splits:
                return splits
        except Exception as exc:
            print(f"⚠️ StratifiedGroupKFold fallback due to: {exc}")

    # Fallback: one split (not group-aware)
    idx_all = np.arange(len(X_train))
    tr_idx, va_idx = train_test_split(
        idx_all,
        test_size=fallback_val_ratio,
        random_state=random_seed,
        stratify=y_train,
    )
    return [(tr_idx, va_idx)]


def optimize_fold_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subj_train: np.ndarray,
    fold_subject_id: int,
    n_trials: int,
    optuna_seed: int,
    cv_splits: int = 3,
    restrict_augmentation: bool = False,
    augmentation_profile: str = "search",
    optuna_objective: str = "mean",
    optuna_std_penalty: float = 0.5,
    preserve_float: bool = False,
    val_ratio: float = 0.2,
) -> Dict:
    """Run fold-wise Optuna (with augmentation) and return best params."""
    cv_index_splits = build_group_cv_splits(
        X_train=X_train,
        y_train=y_train,
        subj_train=subj_train,
        n_splits=max(2, cv_splits),
        random_seed=optuna_seed + int(fold_subject_id),
        fallback_val_ratio=val_ratio,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_hyperparams(
            trial,
            restrict_augmentation=restrict_augmentation,
            augmentation_profile=augmentation_profile,
        )
        params["preserve_float"] = preserve_float
        val_scores = []

        for split_idx, (tr_idx, va_idx) in enumerate(cv_index_splits):
            split_seed = (
                optuna_seed
                + int(fold_subject_id) * 10000
                + trial.number * 100
                + split_idx
            )
            set_all_seeds(split_seed)

            X_tr = X_train[tr_idx]
            y_tr = y_train[tr_idx]
            subj_tr = subj_train[tr_idx]
            X_val = X_train[va_idx]
            y_val = y_train[va_idx]
            subj_val = subj_train[va_idx]

            X_aug, y_aug, subj_aug = augment_data_with_subjects(
                X_tr,
                y_tr,
                subj_tr,
                params,
                seed=split_seed,
            )

            creator = AdannLightgbmModelCreator()
            wrapper = creator.create_model(params, arduino_mode=False)
            wrapper.hybrid_model["lightgbm"].set_params(random_state=split_seed)
            trained_model, _, _ = creator.train_model(
                wrapper.hybrid_model,
                X_aug,
                y_aug,
                subj_aug,
                X_val,
                y_val,
                subj_val,
                params,
                return_history=True,
            )
            val_scores.append(float(trained_model.get("val_macro_f1_ensemble", 0.0)))

        val_scores_arr = np.asarray(val_scores, dtype=float)
        val_mean = float(np.mean(val_scores_arr))
        val_std = _sample_std(val_scores_arr)
        val_min = float(np.min(val_scores_arr))
        trial.set_user_attr("val_macro_f1_mean", val_mean)
        trial.set_user_attr("val_macro_f1_std", val_std)
        trial.set_user_attr("val_macro_f1_min", val_min)

        if optuna_objective == "mean_minus_std":
            return float(val_mean - optuna_std_penalty * val_std)
        if optuna_objective == "min":
            return val_min
        return val_mean

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=optuna_seed + int(fold_subject_id)),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = {k: _to_basic_type(v) for k, v in study.best_params.items()}
    best_params["preserve_float"] = bool(preserve_float)
    best_params["augmentation_profile"] = augmentation_profile
    best_params["optuna_objective"] = optuna_objective
    best_params["optuna_std_penalty"] = float(optuna_std_penalty)
    best_params["optuna_val_macro_f1_mean"] = float(
        study.best_trial.user_attrs.get("val_macro_f1_mean", study.best_value)
    )
    best_params["optuna_val_macro_f1_std"] = float(
        study.best_trial.user_attrs.get("val_macro_f1_std", 0.0)
    )
    best_params["optuna_val_macro_f1_min"] = float(
        study.best_trial.user_attrs.get("val_macro_f1_min", study.best_value)
    )
    best_params["optuna_best_value"] = float(study.best_value)
    best_params["optuna_best_trial"] = int(study.best_trial.number)
    return best_params


def evaluate_branch_accuracies(
    creator: AdannLightgbmModelCreator,
    trained_model: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selection_mode: str = "best_val_branch",
    selection_margin: float = 0.01,
):
    """
    Evaluate ADANN/LGB/Ensemble branch accuracies on test set.
    Returns encoded/decoded predictions for downstream metrics.
    """
    X_test_adann = []
    X_test_lgb = []
    for sample in X_test:
        X_test_adann.append(trained_model["hybrid_extractor"].extract_adann_features(sample))
        X_test_lgb.append(trained_model["hybrid_extractor"].extract_lightgbm_features(sample))
    X_test_adann = np.array(X_test_adann)
    X_test_lgb = np.array(X_test_lgb)

    X_test_adann_scaled = trained_model["adann_scaler"].transform(X_test_adann)
    X_test_lgb_scaled = trained_model["lgb_scaler"].transform(X_test_lgb)

    y_test_encoded = trained_model["gesture_encoder"].transform(y_test)

    # Branch hard predictions for diagnostics
    adann_pred = creator._predict_adann(trained_model["adann"], X_test_adann_scaled)
    lgb_pred = trained_model["lightgbm"].predict(X_test_lgb_scaled)

    # Probability-level fusion (better than hard-label onehot fusion)
    trained_model["adann"].eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_adann_scaled).to(creator.device)
        gesture_logits, _, _ = trained_model["adann"](X_tensor, reverse_gradient=False)
        adann_probs = torch.softmax(gesture_logits, dim=1).cpu().numpy()
    lgb_probs = trained_model["lightgbm"].predict_proba(X_test_lgb_scaled)
    ensemble_weight = float(trained_model["ensemble_weight"])
    ensemble_probs = ensemble_weight * adann_probs + (1.0 - ensemble_weight) * lgb_probs
    ensemble_pred = np.argmax(ensemble_probs, axis=1)
    adann_pred_decoded = trained_model["gesture_encoder"].inverse_transform(adann_pred)
    lgb_pred_decoded = trained_model["gesture_encoder"].inverse_transform(lgb_pred)
    ensemble_pred_decoded = trained_model["gesture_encoder"].inverse_transform(ensemble_pred)

    def confidence_gated_predict(
        adann_prob: np.ndarray,
        lgb_prob: np.ndarray,
        adann_threshold: float,
        lgb_threshold: float,
        static_class: int,
    ) -> np.ndarray:
        gated = []
        for p_adann, p_lgb in zip(adann_prob, lgb_prob):
            y_adann = int(np.argmax(p_adann))
            y_lgb = int(np.argmax(p_lgb))
            c_adann = float(p_adann[y_adann])
            c_lgb = float(p_lgb[y_lgb])
            m_adann = float(np.partition(p_adann, -1)[-1] - np.partition(p_adann, -2)[-2])
            m_lgb = float(np.partition(p_lgb, -1)[-1] - np.partition(p_lgb, -2)[-2])
            adann_confident = c_adann >= adann_threshold
            lgb_confident = c_lgb >= lgb_threshold

            if adann_confident and lgb_confident and y_adann == y_lgb:
                gated.append(y_lgb)
            elif adann_confident and not lgb_confident:
                gated.append(y_adann)
            elif lgb_confident and not adann_confident:
                gated.append(y_lgb)
            elif adann_confident and lgb_confident:
                gated.append(y_adann if m_adann > m_lgb else y_lgb)
            else:
                gated.append(static_class)
        return np.asarray(gated, dtype=int)

    adann_threshold = float(trained_model.get("adann_conf_threshold", 0.5))
    lgb_threshold = float(trained_model.get("lgb_conf_threshold", 0.5))
    try:
        static_class = int(trained_model["gesture_encoder"].transform([10])[0])
    except Exception:
        static_class = int(len(trained_model["gesture_encoder"].classes_) - 1)
    gated_pred = confidence_gated_predict(
        adann_probs,
        lgb_probs,
        adann_threshold=adann_threshold,
        lgb_threshold=lgb_threshold,
        static_class=static_class,
    )
    gated_pred_decoded = trained_model["gesture_encoder"].inverse_transform(gated_pred)

    adann_acc = float(accuracy_score(y_test_encoded, adann_pred))
    lgb_acc = float(accuracy_score(y_test_encoded, lgb_pred))
    ensemble_acc = float(accuracy_score(y_test_encoded, ensemble_pred))
    gated_acc = float(accuracy_score(y_test_encoded, gated_pred))
    f1_labels = trained_model["gesture_encoder"].classes_
    adann_f1 = macro_f1_from_cm(confusion_matrix(y_test, adann_pred_decoded, labels=f1_labels))
    lgb_f1 = macro_f1_from_cm(confusion_matrix(y_test, lgb_pred_decoded, labels=f1_labels))
    ensemble_f1 = macro_f1_from_cm(confusion_matrix(y_test, ensemble_pred_decoded, labels=f1_labels))
    gated_f1 = macro_f1_from_cm(confusion_matrix(y_test, gated_pred_decoded, labels=f1_labels))

    # Validation-driven branch selection (stored by train_model). Prefer the
    # paper's validation Macro-F1 criterion; fall back to accuracy for older
    # model packages that do not store Macro-F1.
    val_adann = float(trained_model.get("val_accuracy_adann", 0.0))
    val_lgb = float(trained_model.get("val_accuracy_lgb", 0.0))
    val_ensemble = float(trained_model.get("val_accuracy_ensemble", 0.0))
    val_gated = float(trained_model.get("val_accuracy_gated", 0.0))
    val_f1_adann = float(trained_model.get("val_macro_f1_adann", val_adann))
    val_f1_lgb = float(trained_model.get("val_macro_f1_lgb", val_lgb))
    val_f1_ensemble = float(trained_model.get("val_macro_f1_ensemble", val_ensemble))
    val_f1_gated = float(trained_model.get("val_macro_f1_gated", val_gated))
    branch_by_val = {
        "adann": val_f1_adann,
        "lgb": val_f1_lgb,
        "ensemble": val_f1_ensemble,
    }
    best_val = max(branch_by_val.values())
    margin = max(0.0, float(selection_margin))
    near_best = {k for k, v in branch_by_val.items() if (best_val - v) <= margin}

    if selection_mode == "robust_val_branch":
        # prefer ensemble under close validation scores
        if "ensemble" in near_best:
            selected_branch = "ensemble"
        elif "lgb" in near_best:
            selected_branch = "lgb"
        else:
            selected_branch = "adann"
    else:
        # strict best with deterministic tie-break
        tie_break_priority = ["lgb", "ensemble", "adann"]
        tied = [b for b in tie_break_priority if abs(branch_by_val[b] - best_val) < 1e-12]
        selected_branch = tied[0] if tied else max(branch_by_val, key=branch_by_val.get)

    if selected_branch == "adann":
        selected_pred_decoded = adann_pred_decoded
        selected_acc = adann_acc
    elif selected_branch == "lgb":
        selected_pred_decoded = lgb_pred_decoded
        selected_acc = lgb_acc
    else:
        selected_pred_decoded = ensemble_pred_decoded
        selected_acc = ensemble_acc

    return {
        "adann_test_acc": float(accuracy_score(y_test_encoded, adann_pred)),
        "lgb_test_acc": float(accuracy_score(y_test_encoded, lgb_pred)),
        "ensemble_test_acc": float(accuracy_score(y_test_encoded, ensemble_pred)),
        "gated_test_acc": gated_acc,
        "adann_macro_f1": adann_f1,
        "lgb_macro_f1": lgb_f1,
        "ensemble_macro_f1": ensemble_f1,
        "gated_macro_f1": gated_f1,
        "adann_pred_decoded": adann_pred_decoded,
        "lgb_pred_decoded": lgb_pred_decoded,
        "ensemble_pred_decoded": ensemble_pred_decoded,
        "gated_pred_decoded": gated_pred_decoded,
        "selected_branch": selected_branch,
        "selected_pred_decoded": selected_pred_decoded,
        "selected_test_acc": selected_acc,
        "val_accuracy_adann": val_adann,
        "val_accuracy_lgb": val_lgb,
        "val_accuracy_ensemble": val_ensemble,
        "val_accuracy_gated": val_gated,
        "val_macro_f1_adann": val_f1_adann,
        "val_macro_f1_lgb": val_f1_lgb,
        "val_macro_f1_ensemble": val_f1_ensemble,
        "val_macro_f1_gated": val_f1_gated,
        "adann_conf_threshold": adann_threshold,
        "lgb_conf_threshold": lgb_threshold,
        "selection_margin": margin,
    }


def save_outputs(
    records: List[Dict],
    fold_best_params: Dict[int, Dict],
    output_dir: str,
    seeds: List[int],
) -> None:
    """Save raw records, per-seed aggregation, and summary JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Save fold-wise best params
    fold_json = os.path.join(output_dir, "fold_best_hyperparams.json")
    with open(fold_json, "w") as f:
        json.dump({str(k): v for k, v in fold_best_params.items()}, f, indent=2)

    fold_rows = []
    for subject_id, params in fold_best_params.items():
        row = {"subject_id": int(subject_id)}
        row.update(params)
        fold_rows.append(row)
    fold_df = pd.DataFrame(fold_rows).sort_values("subject_id")
    fold_csv = os.path.join(output_dir, "fold_best_hyperparams.csv")
    fold_df.to_csv(fold_csv, index=False)

    raw_df = pd.DataFrame.from_records(records)
    raw_csv = os.path.join(output_dir, "seed_stability_raw.csv")
    raw_df.to_csv(raw_csv, index=False)

    by_seed_df = (
        raw_df.groupby("seed", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            n_folds=("subject_id", "count"),
        )
        .sort_values("seed")
    )
    by_seed_df["std_accuracy"] = by_seed_df["std_accuracy"].fillna(0.0)
    by_seed_df["std_macro_f1"] = by_seed_df["std_macro_f1"].fillna(0.0)
    by_seed_df["ci95_accuracy"] = by_seed_df.apply(
        lambda row: _ci95(
            raw_df.loc[raw_df["seed"] == row["seed"], "accuracy"].to_numpy()
        )["ci95_half_width"],
        axis=1,
    )
    by_seed_df["ci95_macro_f1"] = by_seed_df.apply(
        lambda row: _ci95(
            raw_df.loc[raw_df["seed"] == row["seed"], "macro_f1"].to_numpy()
        )["ci95_half_width"],
        axis=1,
    )
    by_seed_csv = os.path.join(output_dir, "seed_stability_by_seed.csv")
    by_seed_df.to_csv(by_seed_csv, index=False)

    by_fold_df = (
        raw_df.groupby("subject_id", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            n_seeds=("seed", "count"),
        )
        .sort_values("subject_id")
    )
    by_fold_df["std_accuracy"] = by_fold_df["std_accuracy"].fillna(0.0)
    by_fold_df["std_macro_f1"] = by_fold_df["std_macro_f1"].fillna(0.0)
    by_fold_df["ci95_accuracy"] = by_fold_df.apply(
        lambda row: _ci95(
            raw_df.loc[raw_df["subject_id"] == row["subject_id"], "accuracy"].to_numpy()
        )["ci95_half_width"],
        axis=1,
    )
    by_fold_df["ci95_macro_f1"] = by_fold_df.apply(
        lambda row: _ci95(
            raw_df.loc[raw_df["subject_id"] == row["subject_id"], "macro_f1"].to_numpy()
        )["ci95_half_width"],
        axis=1,
    )
    by_fold_csv = os.path.join(output_dir, "seed_stability_by_fold.csv")
    by_fold_df.to_csv(by_fold_csv, index=False)

    seed_means_acc = by_seed_df["mean_accuracy"].to_numpy()
    seed_means_f1 = by_seed_df["mean_macro_f1"].to_numpy()
    fold_means_acc = by_fold_df["mean_accuracy"].to_numpy()
    fold_means_f1 = by_fold_df["mean_macro_f1"].to_numpy()
    seed_acc_ci = _ci95(seed_means_acc)
    seed_f1_ci = _ci95(seed_means_f1)
    fold_acc_ci = _ci95(fold_means_acc)
    fold_f1_ci = _ci95(fold_means_f1)

    summary = {
        "seeds": [int(s) for s in seeds],
        "n_seeds": int(len(seeds)),
        "n_folds": int(by_fold_df["subject_id"].nunique()),
        "n_records": int(len(raw_df)),
        "overall_mean_accuracy": seed_acc_ci["mean"],
        "overall_std_accuracy": seed_acc_ci["std"],
        "overall_ci95_accuracy": seed_acc_ci["ci95_half_width"],
        "overall_ci95_low_accuracy": seed_acc_ci["ci95_low"],
        "overall_ci95_high_accuracy": seed_acc_ci["ci95_high"],
        "overall_min_accuracy": float(np.min(seed_means_acc)),
        "overall_max_accuracy": float(np.max(seed_means_acc)),
        "overall_mean_macro_f1": seed_f1_ci["mean"],
        "overall_std_macro_f1": seed_f1_ci["std"],
        "overall_ci95_macro_f1": seed_f1_ci["ci95_half_width"],
        "overall_ci95_low_macro_f1": seed_f1_ci["ci95_low"],
        "overall_ci95_high_macro_f1": seed_f1_ci["ci95_high"],
        "overall_min_macro_f1": float(np.min(seed_means_f1)),
        "overall_max_macro_f1": float(np.max(seed_means_f1)),
        "seed_wise_variance_accuracy": float(np.var(seed_means_acc, ddof=1)) if len(seed_means_acc) > 1 else 0.0,
        "seed_wise_variance_macro_f1": float(np.var(seed_means_f1, ddof=1)) if len(seed_means_f1) > 1 else 0.0,
        "fold_wise_mean_accuracy": fold_acc_ci["mean"],
        "fold_wise_std_accuracy": fold_acc_ci["std"],
        "fold_wise_ci95_accuracy": fold_acc_ci["ci95_half_width"],
        "fold_wise_mean_macro_f1": fold_f1_ci["mean"],
        "fold_wise_std_macro_f1": fold_f1_ci["std"],
        "fold_wise_ci95_macro_f1": fold_f1_ci["ci95_half_width"],
        "fold_wise_variance_accuracy": float(np.var(fold_means_acc, ddof=1)) if len(fold_means_acc) > 1 else 0.0,
        "fold_wise_variance_macro_f1": float(np.var(fold_means_f1, ddof=1)) if len(fold_means_f1) > 1 else 0.0,
        "best_seed_by_accuracy": int(by_seed_df.loc[by_seed_df["mean_accuracy"].idxmax(), "seed"]),
        "worst_seed_by_accuracy": int(by_seed_df.loc[by_seed_df["mean_accuracy"].idxmin(), "seed"]),
    }

    summary_json = os.path.join(output_dir, "seed_stability_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs:")
    print(f"  - Fold best params (CSV): {fold_csv}")
    print(f"  - Fold best params (JSON): {fold_json}")
    print(f"  - Raw records: {raw_csv}")
    print(f"  - By-seed stats: {by_seed_csv}")
    print(f"  - By-fold stats: {by_fold_csv}")
    print(f"  - Summary JSON: {summary_json}")


def run_seed_stability_loso(
    csv_dir: str,
    output_dir: str,
    seeds: List[int],
    n_trials: int,
    val_ratio: float,
    optuna_seed: int,
    optuna_cv_splits: int,
    restrict_augmentation: bool,
    augmentation_profile: str,
    optuna_objective: str,
    optuna_std_penalty: float,
    sample_normalization: str,
    auto_tune_gate_thresholds: bool,
    selection_mode: str,
    selection_margin: float,
    train_on_full_fold: bool,
    final_val_on_train: bool,
    final_val_ratio: float,
    group_aware_final_val: bool,
    fixed_hyperparams_path: str,
) -> None:
    """Run LOSO seed stability with fold-wise Optuna best hyperparameters."""
    print(f"Loading data from: {csv_dir}")
    X, y, subjects = load_data(csv_dir)
    sample_normalization = (sample_normalization or "none").lower()
    preserve_float = sample_normalization in ("zscore", "robust_zscore")
    X = apply_sample_normalization(X, sample_normalization)

    unique_subjects = np.unique(subjects[subjects != -1])
    if len(unique_subjects) < 2:
        raise RuntimeError("LOSO requires at least 2 subjects.")

    print(f"Found {len(X)} samples, {len(unique_subjects)} subjects.")
    print(f"Seeds: {seeds}")
    print(f"Sample normalization: {sample_normalization}")
    if restrict_augmentation:
        augmentation_profile = "restricted"
    print(f"Augmentation profile: {augmentation_profile}")
    print(f"Auto-tune gated thresholds: {auto_tune_gate_thresholds}")
    print(f"Optuna objective: {optuna_objective} (std penalty={optuna_std_penalty})")

    records: List[Dict] = []
    all_labels = np.unique(y)
    fold_best_params: Dict[int, Dict] = {}
    fold_split_data: Dict[int, Dict] = {}
    fixed_hyperparams = None
    fixed_hyperparams_by_fold = None

    if fixed_hyperparams_path:
        with open(fixed_hyperparams_path, "r") as f:
            loaded = json.load(f)
        if (
            isinstance(loaded, dict)
            and loaded
            and all(str(k).lstrip("-").isdigit() for k in loaded.keys())
            and all(isinstance(v, dict) for v in loaded.values())
        ):
            fixed_hyperparams_by_fold = {int(k): v for k, v in loaded.items()}
        else:
            fixed_hyperparams = (
                loaded.get("best_hyperparameters_for_deployment")
                or loaded.get("best_params")
                or loaded
            )
        print(f"\n✅ Loaded fixed hyperparams from: {fixed_hyperparams_path}")

    # Stage 1: fold-wise Optuna
    print("\n========== Stage 1/2: Fold-wise Hyperparams ==========")
    for test_subj in unique_subjects:
        print(f"\n[Fold] test subject {test_subj}")
        test_mask = subjects == test_subj
        train_mask = ~test_mask
        X_train = X[train_mask]
        y_train = y[train_mask]
        subj_train = subjects[train_mask]

        split_seed = optuna_seed + int(test_subj) * 1000
        X_tr, y_tr, subj_tr, X_val, y_val, subj_val = split_train_val(
            X_train,
            y_train,
            subj_train,
            split_seed=split_seed,
            val_ratio=val_ratio,
            group_aware=group_aware_final_val,
        )
        fold_split_data[int(test_subj)] = {
            "X_tr": X_tr,
            "y_tr": y_tr,
            "subj_tr": subj_tr,
            "X_val": X_val,
            "y_val": y_val,
            "subj_val": subj_val,
        }

        if fixed_hyperparams is not None or fixed_hyperparams_by_fold is not None:
            source_params = (
                fixed_hyperparams_by_fold.get(int(test_subj), {})
                if fixed_hyperparams_by_fold is not None
                else fixed_hyperparams
            )
            if not source_params:
                raise RuntimeError(f"No fixed hyperparams found for subject {test_subj}.")
            best_params = {k: _to_basic_type(v) for k, v in source_params.items()}
            best_params["preserve_float"] = bool(best_params.get("preserve_float", preserve_float))
            best_params.setdefault("augmentation_profile", augmentation_profile)
            best_params.setdefault("optuna_objective", optuna_objective)
            best_params.setdefault("optuna_std_penalty", float(optuna_std_penalty))
            best_params.setdefault("optuna_val_macro_f1_mean", float("nan"))
            best_params.setdefault("optuna_val_macro_f1_std", float("nan"))
            best_params.setdefault("optuna_val_macro_f1_min", float("nan"))
            best_params["auto_tune_gate_thresholds"] = bool(auto_tune_gate_thresholds)
            best_params["optuna_best_value"] = float("nan")
            best_params["optuna_best_trial"] = -1
        else:
            best_params = optimize_fold_hyperparams(
                X_train=X_train,
                y_train=y_train,
                subj_train=subj_train,
                fold_subject_id=int(test_subj),
                n_trials=n_trials,
                optuna_seed=optuna_seed,
                cv_splits=optuna_cv_splits,
                restrict_augmentation=restrict_augmentation,
                augmentation_profile=augmentation_profile,
                optuna_objective=optuna_objective,
                optuna_std_penalty=optuna_std_penalty,
                preserve_float=preserve_float,
                val_ratio=val_ratio,
            )
            best_params["auto_tune_gate_thresholds"] = bool(auto_tune_gate_thresholds)
        fold_best_params[int(test_subj)] = best_params
        if fixed_hyperparams is None and fixed_hyperparams_by_fold is None:
            print(
                f"  Best val={best_params['optuna_best_value']:.4f}, "
                f"trial={best_params['optuna_best_trial']}, "
                f"mean={best_params.get('optuna_val_macro_f1_mean', float('nan')):.4f}, "
                f"std={best_params.get('optuna_val_macro_f1_std', float('nan')):.4f}"
            )
        else:
            print("  Using fixed hyperparams (skip Optuna)")

    # Stage 2: seed stability with fold-wise best params
    print("\n========== Stage 2/2: Seed Stability ==========")
    for seed in seeds:
        print(f"\n===== Seed: {seed} =====")
        set_all_seeds(seed)

        for test_subj in unique_subjects:
            print(f"--- LOSO fold: test subject {test_subj} ---")
            test_mask = subjects == test_subj
            train_mask = ~test_mask

            X_train = X[train_mask]
            y_train = y[train_mask]
            subj_train = subjects[train_mask]

            X_test = X[test_mask]
            y_test = y[test_mask]

            if X_test.shape[0] == 0:
                print(f"  [Skip] No samples for subject {test_subj}")
                continue

            fold_params = dict(fold_best_params[int(test_subj)])
            # remove logging metadata from params used in actual training
            fold_params.pop("optuna_best_value", None)
            fold_params.pop("optuna_best_trial", None)
            fold_params["auto_tune_gate_thresholds"] = bool(auto_tune_gate_thresholds)

            if train_on_full_fold:
                if final_val_on_train:
                    # Final model evaluation protocol: after hyperparameter
                    # selection, retrain on all non-test subjects. The same
                    # training data are passed as validation only to satisfy
                    # the existing early-stopping/diagnostic API; no held-out
                    # test-subject information is used here.
                    X_train_fold = X_train
                    y_train_fold = y_train
                    subj_train_fold = subj_train
                    X_val_fold = X_train
                    y_val_fold = y_train
                    subj_val_fold = subj_train
                else:
                    # Re-train with as much data as possible for final seed evaluation.
                    # Keep a small holdout only for early stopping/branch diagnostics.
                    final_split_seed = optuna_seed + int(test_subj) * 7777
                    (
                        X_train_fold,
                        y_train_fold,
                        subj_train_fold,
                        X_val_fold,
                        y_val_fold,
                        subj_val_fold,
                    ) = split_train_val(
                        X_train,
                        y_train,
                        subj_train,
                        split_seed=final_split_seed,
                        val_ratio=final_val_ratio,
                        group_aware=group_aware_final_val,
                    )
            else:
                split_data = fold_split_data[int(test_subj)]
                X_train_fold = split_data["X_tr"]
                y_train_fold = split_data["y_tr"]
                subj_train_fold = split_data["subj_tr"]
                X_val_fold = split_data["X_val"]
                y_val_fold = split_data["y_val"]
                subj_val_fold = split_data["subj_val"]

            set_all_seeds(seed)
            X_train_aug, y_train_aug, subj_train_aug = augment_data_with_subjects(
                X_train_fold,
                y_train_fold,
                subj_train_fold,
                fold_params,
                seed=seed + int(test_subj) * 10000,
            )

            creator = AdannLightgbmModelCreator()
            wrapper = creator.create_model(fold_params, arduino_mode=False)
            wrapper.hybrid_model["lightgbm"].set_params(random_state=seed)

            trained_model, _, _ = creator.train_model(
                wrapper.hybrid_model,
                X_train_aug,
                y_train_aug,
                subj_train_aug,
                X_val_fold,
                y_val_fold,
                subj_val_fold,
                fold_params,
                return_history=True,
            )

            branch_eval = evaluate_branch_accuracies(
                creator=creator,
                trained_model=trained_model,
                X_test=X_test,
                y_test=y_test,
                selection_mode=selection_mode,
                selection_margin=selection_margin,
            )
            if selection_mode in ("best_val_branch", "robust_val_branch"):
                y_pred = branch_eval["selected_pred_decoded"]
                acc = float(branch_eval["selected_test_acc"])
            elif selection_mode == "lgb":
                y_pred = branch_eval["lgb_pred_decoded"]
                acc = float(branch_eval["lgb_test_acc"])
            elif selection_mode == "adann":
                y_pred = branch_eval["adann_pred_decoded"]
                acc = float(branch_eval["adann_test_acc"])
            elif selection_mode == "gated":
                y_pred = branch_eval["gated_pred_decoded"]
                acc = float(branch_eval["gated_test_acc"])
            else:
                y_pred = branch_eval["ensemble_pred_decoded"]
                acc = float(branch_eval["ensemble_test_acc"])
            cm = confusion_matrix(y_test, y_pred, labels=all_labels)
            mf1 = macro_f1_from_cm(cm)

            print(f"  Test acc: {acc:.4f}, macro_f1: {mf1:.4f}")

            records.append(
                {
                    "seed": int(seed),
                    "subject_id": int(test_subj),
                    "n_test": int(X_test.shape[0]),
                    "accuracy": float(acc),
                    "macro_f1": float(mf1),
                    "adann_test_acc": float(branch_eval["adann_test_acc"]),
                    "lgb_test_acc": float(branch_eval["lgb_test_acc"]),
                    "ensemble_test_acc": float(branch_eval["ensemble_test_acc"]),
                    "gated_test_acc": float(branch_eval["gated_test_acc"]),
                    "adann_macro_f1": float(branch_eval["adann_macro_f1"]),
                    "lgb_macro_f1": float(branch_eval["lgb_macro_f1"]),
                    "ensemble_macro_f1": float(branch_eval["ensemble_macro_f1"]),
                    "gated_macro_f1": float(branch_eval["gated_macro_f1"]),
                    "selected_branch": branch_eval["selected_branch"],
                    "selected_test_acc": float(branch_eval["selected_test_acc"]),
                    "selected_ensemble_weight": float(trained_model["ensemble_weight"]),
                    "val_accuracy_adann": float(branch_eval["val_accuracy_adann"]),
                    "val_accuracy_lgb": float(branch_eval["val_accuracy_lgb"]),
                    "val_accuracy_ensemble": float(branch_eval["val_accuracy_ensemble"]),
                    "val_accuracy_gated": float(branch_eval["val_accuracy_gated"]),
                    "val_macro_f1_adann": float(branch_eval["val_macro_f1_adann"]),
                    "val_macro_f1_lgb": float(branch_eval["val_macro_f1_lgb"]),
                    "val_macro_f1_ensemble": float(branch_eval["val_macro_f1_ensemble"]),
                    "val_macro_f1_gated": float(branch_eval["val_macro_f1_gated"]),
                    "adann_conf_threshold": float(branch_eval["adann_conf_threshold"]),
                    "lgb_conf_threshold": float(branch_eval["lgb_conf_threshold"]),
                    "selection_mode": selection_mode,
                    "augmentation_profile": augmentation_profile,
                    "sample_normalization": sample_normalization,
                    "selection_margin": float(branch_eval["selection_margin"]),
                    "train_on_full_fold": bool(train_on_full_fold),
                    "final_val_on_train": bool(final_val_on_train),
                    "final_val_ratio": float(final_val_ratio),
                    "group_aware_final_val": bool(group_aware_final_val),
                    "optuna_best_value": float(
                        fold_best_params[int(test_subj)]["optuna_best_value"]
                    ),
                    "optuna_best_trial": int(
                        fold_best_params[int(test_subj)]["optuna_best_trial"]
                    ),
                }
            )

    save_outputs(records, fold_best_params, output_dir, seeds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DA-LGBM (ADANN_LightGBM) LOSO seed stability experiment"
    )
    parser.add_argument(
        "--csv_dir",
        default="datasets/gesture_csv",
        help="Directory containing gesture CSV files",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/DA_LGBM_seed_stability",
        help="Directory to save CSV/JSON records",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4,5,10,20,42,123",
        help="Comma-separated integer seeds",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Optuna trials for each LOSO fold",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Train/val split ratio used during fold-wise Optuna",
    )
    parser.add_argument(
        "--optuna_seed",
        type=int,
        default=42,
        help="Base random seed used for fold-wise Optuna search",
    )
    parser.add_argument(
        "--optuna_cv_splits",
        type=int,
        default=3,
        help="Average each Optuna trial over N repeated runs",
    )
    parser.add_argument(
        "--restrict_augmentation",
        action="store_true",
        help="Use a narrower augmentation search range for better generalization",
    )
    parser.add_argument(
        "--augmentation_profile",
        type=str,
        default="search",
        choices=["search", "restricted", "legacy_strong", "hardware_shift"],
        help=(
            "Augmentation search profile: search=current space, restricted=narrow, "
            "legacy_strong=wider historical scale/factor, hardware_shift=channel-wise "
            "scale/offset/drift for cross-subject shift"
        ),
    )
    parser.add_argument(
        "--optuna_objective",
        type=str,
        default="mean",
        choices=["mean", "mean_minus_std", "min"],
        help=(
            "Objective used to select Optuna trials from CV validation scores. "
            "mean_minus_std penalizes unstable trials across validation splits."
        ),
    )
    parser.add_argument(
        "--optuna_std_penalty",
        type=float,
        default=0.5,
        help="Penalty multiplier for --optuna_objective mean_minus_std",
    )
    parser.add_argument(
        "--sample_normalization",
        type=str,
        default="none",
        choices=["none", "center", "baseline", "zscore", "robust_zscore"],
        help=(
            "Per-window input normalization applied before LOSO splitting. "
            "Uses only each sample's own channel statistics."
        ),
    )
    parser.add_argument(
        "--auto_tune_gate_thresholds",
        action="store_true",
        help="Tune ADANN/LGB confidence-gate thresholds on the inner validation split",
    )
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="best_val_branch",
        choices=["best_val_branch", "robust_val_branch", "ensemble", "gated", "lgb", "adann"],
        help="How to choose final test prediction branch",
    )
    parser.add_argument(
        "--selection_margin",
        type=float,
        default=0.01,
        help="Validation margin used by robust_val_branch",
    )
    parser.add_argument(
        "--train_on_full_fold",
        action="store_true",
        default=True,
        help="For seed stage, re-split from full fold training data (more samples)",
    )
    parser.add_argument(
        "--no_train_on_full_fold",
        dest="train_on_full_fold",
        action="store_false",
        help="Use the Stage-1 split again during seed-stage training",
    )
    parser.add_argument(
        "--final_val_on_train",
        action="store_true",
        help=(
            "During seed-stage final evaluation, train on all non-test subjects "
            "and pass the same non-test data as validation diagnostics. This "
            "matches final retraining after hyperparameter selection and does "
            "not use held-out test-subject data."
        ),
    )
    parser.add_argument(
        "--final_val_ratio",
        type=float,
        default=0.2,
        help="Validation ratio for final seed-stage re-training when --train_on_full_fold is set",
    )
    parser.add_argument(
        "--sample_level_final_val",
        dest="group_aware_final_val",
        action="store_false",
        default=False,
        help="Use sample-level random final validation inside the outer LOSO training subjects",
    )
    parser.add_argument(
        "--group_aware_final_val",
        dest="group_aware_final_val",
        action="store_true",
        help="Use subject-held-out final validation inside each LOSO training fold",
    )
    parser.add_argument(
        "--fixed_hyperparams_path",
        type=str,
        default="",
        help="If set, skip fold Optuna and use this hyperparams JSON",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_list = parse_seeds(args.seeds)
    run_seed_stability_loso(
        csv_dir=args.csv_dir,
        output_dir=args.output_dir,
        seeds=seed_list,
        n_trials=args.n_trials,
        val_ratio=args.val_ratio,
        optuna_seed=args.optuna_seed,
        optuna_cv_splits=args.optuna_cv_splits,
        restrict_augmentation=args.restrict_augmentation,
        augmentation_profile=args.augmentation_profile,
        optuna_objective=args.optuna_objective,
        optuna_std_penalty=args.optuna_std_penalty,
        sample_normalization=args.sample_normalization,
        auto_tune_gate_thresholds=args.auto_tune_gate_thresholds,
        selection_mode=args.selection_mode,
        selection_margin=args.selection_margin,
        train_on_full_fold=args.train_on_full_fold,
        final_val_on_train=args.final_val_on_train,
        final_val_ratio=args.final_val_ratio,
        group_aware_final_val=args.group_aware_final_val,
        fixed_hyperparams_path=args.fixed_hyperparams_path.strip(),
    )
