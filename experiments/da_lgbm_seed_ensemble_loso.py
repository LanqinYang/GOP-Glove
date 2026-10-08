#!/usr/bin/env python3
"""
DA-LGBM seed ensemble under LOSO.

This script evaluates a pre-specified seed ensemble: for each LOSO fold, train
the same fixed hyperparameter configuration with multiple random seeds and
average their predicted probabilities on the held-out subject.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.da_lgbm_seed_stability_loso import (  # noqa: E402
    augment_data_with_subjects,
    apply_sample_normalization,
    load_data,
    macro_f1_from_cm,
    parse_seeds,
    set_all_seeds,
    split_train_val,
    _ci95,
    _to_basic_type,
)
from src.training.train_adann_lightgbm import AdannLightgbmModelCreator  # noqa: E402


def load_fixed_hyperparams(path: str) -> Dict:
    with open(path, "r") as f:
        loaded = json.load(f)
    if (
        isinstance(loaded, dict)
        and loaded
        and all(str(k).lstrip("-").isdigit() for k in loaded.keys())
        and all(isinstance(v, dict) for v in loaded.values())
    ):
        return {int(k): v for k, v in loaded.items()}
    return loaded.get("best_hyperparameters_for_deployment") or loaded.get("best_params") or loaded


def extract_test_probabilities(
    creator: AdannLightgbmModelCreator,
    trained_model: Dict,
    X_test: np.ndarray,
) -> Dict[str, np.ndarray]:
    X_test_adann = []
    X_test_lgb = []
    for sample in X_test:
        X_test_adann.append(trained_model["hybrid_extractor"].extract_adann_features(sample))
        X_test_lgb.append(trained_model["hybrid_extractor"].extract_lightgbm_features(sample))
    X_test_adann = np.asarray(X_test_adann)
    X_test_lgb = np.asarray(X_test_lgb)

    X_test_adann_scaled = trained_model["adann_scaler"].transform(X_test_adann)
    X_test_lgb_scaled = trained_model["lgb_scaler"].transform(X_test_lgb)

    trained_model["adann"].eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_adann_scaled).to(creator.device)
        gesture_logits, _, _ = trained_model["adann"](X_tensor, reverse_gradient=False)
        adann_probs = torch.softmax(gesture_logits, dim=1).cpu().numpy()

    lgb_probs_raw = trained_model["lightgbm"].predict_proba(X_test_lgb_scaled)
    lgb_probs = np.zeros_like(adann_probs)
    for prob_col, class_label in enumerate(trained_model["lightgbm"].classes_):
        class_idx = int(class_label)
        if class_idx < lgb_probs.shape[1]:
            lgb_probs[:, class_idx] = lgb_probs_raw[:, prob_col]

    ensemble_weight = float(trained_model["ensemble_weight"])
    ensemble_probs = ensemble_weight * adann_probs + (1.0 - ensemble_weight) * lgb_probs

    try:
        static_class = int(trained_model["gesture_encoder"].transform([10])[0])
    except Exception:
        static_class = int(len(trained_model["gesture_encoder"].classes_) - 1)

    adann_threshold = float(trained_model.get("adann_conf_threshold", 0.5))
    lgb_threshold = float(trained_model.get("lgb_conf_threshold", 0.5))
    gated_onehot = np.zeros_like(ensemble_probs)
    for i, (p_adann, p_lgb) in enumerate(zip(adann_probs, lgb_probs)):
        y_adann = int(np.argmax(p_adann))
        y_lgb = int(np.argmax(p_lgb))
        c_adann = float(p_adann[y_adann])
        c_lgb = float(p_lgb[y_lgb])
        m_adann = float(np.partition(p_adann, -1)[-1] - np.partition(p_adann, -2)[-2])
        m_lgb = float(np.partition(p_lgb, -1)[-1] - np.partition(p_lgb, -2)[-2])
        adann_confident = c_adann >= adann_threshold
        lgb_confident = c_lgb >= lgb_threshold
        if adann_confident and lgb_confident and y_adann == y_lgb:
            chosen = y_lgb
        elif adann_confident and not lgb_confident:
            chosen = y_adann
        elif lgb_confident and not adann_confident:
            chosen = y_lgb
        elif adann_confident and lgb_confident:
            chosen = y_adann if m_adann > m_lgb else y_lgb
        else:
            chosen = static_class
        gated_onehot[i, chosen] = 1.0

    return {
        "adann": adann_probs,
        "lgb": lgb_probs,
        "ensemble": ensemble_probs,
        "gated": gated_onehot,
    }


def evaluate_probs(y_true: np.ndarray, class_labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    pred_encoded = np.argmax(probs, axis=1)
    y_pred = class_labels[pred_encoded]
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1_from_cm(cm)),
    }


def _pred_labels_from_probs(class_labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    pred_encoded = np.argmax(probs, axis=1)
    return class_labels[pred_encoded]


def append_prediction_diagnostics(
    prediction_records: List[Dict],
    class_records: List[Dict],
    confusion_records: List[Dict],
    subject_id: int,
    branch: str,
    selection: str,
    y_true: np.ndarray,
    class_labels: np.ndarray,
    probs: np.ndarray,
) -> None:
    y_pred = _pred_labels_from_probs(class_labels, probs)
    top_prob = np.max(probs, axis=1)
    for sample_idx, (yt, yp, conf) in enumerate(zip(y_true, y_pred, top_prob)):
        prediction_records.append(
            {
                "subject_id": int(subject_id),
                "branch": branch,
                "selection": selection,
                "sample_index": int(sample_idx),
                "true_label": int(yt),
                "pred_label": int(yp),
                "correct": bool(yt == yp),
                "top_probability": float(conf),
            }
        )

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    for true_idx, true_label in enumerate(class_labels):
        support = int(cm[true_idx, :].sum())
        tp = int(cm[true_idx, true_idx])
        fn = int(cm[true_idx, :].sum() - tp)
        fp = int(cm[:, true_idx].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        class_records.append(
            {
                "subject_id": int(subject_id),
                "branch": branch,
                "selection": selection,
                "class_label": int(true_label),
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
        for pred_idx, pred_label in enumerate(class_labels):
            count = int(cm[true_idx, pred_idx])
            if count:
                confusion_records.append(
                    {
                        "subject_id": int(subject_id),
                        "branch": branch,
                        "selection": selection,
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                        "count": count,
                    }
                )


def _simplex_weight_grid(n_models: int, step: float) -> List[np.ndarray]:
    """Generate non-negative weights summing to 1.0 on a coarse simplex grid."""
    if n_models <= 0:
        return []
    units = int(round(1.0 / step))
    if units <= 0 or not np.isclose(units * step, 1.0):
        raise ValueError("--val_weight_step must divide 1.0, e.g. 0.1, 0.05, 0.02")

    weights = []

    def rec(prefix: List[int], remaining: int, slots_left: int) -> None:
        if slots_left == 1:
            weights.append(np.asarray(prefix + [remaining], dtype=float) / units)
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, slots_left - 1)

    rec([], units, n_models)
    return weights


def _weighted_average(prob_list: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack(prob_list, axis=0)
    return np.tensordot(weights, stacked, axes=(0, 0))


def select_validation_weights(
    y_val: np.ndarray,
    class_labels: np.ndarray,
    val_prob_list: List[np.ndarray],
    test_prob_list: List[np.ndarray],
    step: float,
    max_models: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], List[int], Dict[str, float]]:
    """Choose ensemble members and weights only from validation performance."""
    individual_scores = [
        evaluate_probs(y_val, class_labels, probs)["macro_f1"]
        for probs in val_prob_list
    ]
    selected_indices = list(np.argsort(individual_scores)[::-1][:max_models])
    selected_indices.sort()
    selected_val = [val_prob_list[i] for i in selected_indices]
    selected_test = [test_prob_list[i] for i in selected_indices]

    best_weights = None
    best_val_metrics = None
    best_key = None
    for weights in _simplex_weight_grid(len(selected_val), step):
        avg_val = _weighted_average(selected_val, weights)
        val_metrics = evaluate_probs(y_val, class_labels, avg_val)
        key = (val_metrics["macro_f1"], val_metrics["accuracy"], -float(np.count_nonzero(weights)))
        if best_key is None or key > best_key:
            best_key = key
            best_weights = weights
            best_val_metrics = val_metrics

    avg_test = _weighted_average(selected_test, best_weights)
    return avg_test, best_weights, best_val_metrics, selected_indices, {
        "mean_individual_val_macro_f1": float(np.mean(individual_scores)),
        "max_individual_val_macro_f1": float(np.max(individual_scores)),
    }


def run_seed_ensemble(
    csv_dir: str,
    output_dir: str,
    seeds: List[int],
    fixed_hyperparams_paths: List[str],
    sample_normalization: str,
    final_val_ratio: float,
    final_val_on_train: bool,
    group_aware_final_val: bool,
    optuna_seed: int,
    val_weight_step: float,
    max_weight_grid_models: int,
    targeted_augment_profile: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading data from: {csv_dir}")
    X, y, subjects = load_data(csv_dir)
    sample_normalization = (sample_normalization or "none").lower()
    X = apply_sample_normalization(X, sample_normalization)

    fixed_configs = []
    for path in fixed_hyperparams_paths:
        fixed = load_fixed_hyperparams(path)
        fixed_by_fold = isinstance(fixed, dict) and fixed and all(isinstance(k, int) for k in fixed.keys())
        fixed_configs.append((path, fixed, fixed_by_fold))

    unique_subjects = np.unique(subjects[subjects != -1])
    all_labels = np.unique(y)
    records = []
    prediction_records = []
    class_records = []
    confusion_records = []

    for test_subj in unique_subjects:
        print(f"\n--- LOSO fold: test subject {test_subj} ---")
        test_mask = subjects == test_subj
        train_mask = ~test_mask
        X_train = X[train_mask]
        y_train = y[train_mask]
        subj_train = subjects[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        if final_val_on_train:
            X_train_fold, y_train_fold, subj_train_fold = X_train, y_train, subj_train
            X_val_fold, y_val_fold, subj_val_fold = X_train, y_train, subj_train
        else:
            X_train_fold, y_train_fold, subj_train_fold, X_val_fold, y_val_fold, subj_val_fold = split_train_val(
                X_train,
                y_train,
                subj_train,
                split_seed=optuna_seed + int(test_subj) * 7777,
                val_ratio=final_val_ratio,
                group_aware=group_aware_final_val,
            )

        per_branch_test_probs = {"adann": [], "lgb": [], "ensemble": [], "gated": []}
        per_branch_val_probs = {"adann": [], "lgb": [], "ensemble": [], "gated": []}
        class_labels = None
        for config_idx, (config_path, fixed, fixed_by_fold) in enumerate(fixed_configs, start=1):
            source_params = fixed[int(test_subj)] if fixed_by_fold else fixed
            fold_params = {k: _to_basic_type(v) for k, v in source_params.items()}
            fold_params.setdefault("preserve_float", sample_normalization in ("zscore", "robust_zscore"))
            if targeted_augment_profile == "hard_017_mild":
                fold_params.update(
                    {
                        "augment_factor": int(fold_params.get("augment_factor", 0) or 0),
                        "class_augment_labels": [0, 1, 7],
                        "class_augment_extra_factor": 1,
                        "augment_prob": 0.6,
                        "jitter_noise_level": 0.004,
                        "time_warp_max_speed": 1.6,
                        "scale_min": 0.97,
                        "scale_max": 1.03,
                    }
                )
            elif targeted_augment_profile == "hard_017_strong":
                fold_params.update(
                    {
                        "augment_factor": int(fold_params.get("augment_factor", 0) or 0),
                        "class_augment_labels": [0, 1, 7],
                        "class_augment_extra_factor": 2,
                        "augment_prob": 0.75,
                        "jitter_noise_level": 0.008,
                        "time_warp_max_speed": 2.0,
                        "scale_min": 0.95,
                        "scale_max": 1.05,
                    }
                )
            print(f"  Config {config_idx}/{len(fixed_configs)}: {config_path}")

            for seed in seeds:
                print(f"    Seed {seed}")
                set_all_seeds(seed)
                X_aug, y_aug, subj_aug = augment_data_with_subjects(
                    X_train_fold,
                    y_train_fold,
                    subj_train_fold,
                    fold_params,
                    seed=seed + int(test_subj) * 10000 + config_idx * 1000000,
                )
                creator = AdannLightgbmModelCreator()
                wrapper = creator.create_model(fold_params, arduino_mode=False)
                wrapper.hybrid_model["lightgbm"].set_params(random_state=seed + config_idx * 1000000)
                trained_model, _ = creator.train_model(
                    wrapper.hybrid_model,
                    X_aug,
                    y_aug,
                    subj_aug,
                    X_val_fold,
                    y_val_fold,
                    subj_val_fold,
                    fold_params,
                    return_history=False,
                )
                class_labels = trained_model["gesture_encoder"].classes_
                test_probs = extract_test_probabilities(creator, trained_model, X_test)
                val_probs = extract_test_probabilities(creator, trained_model, X_val_fold)
                for branch, branch_probs in test_probs.items():
                    per_branch_test_probs[branch].append(branch_probs)
                for branch, branch_probs in val_probs.items():
                    per_branch_val_probs[branch].append(branch_probs)

        for branch, prob_list in per_branch_test_probs.items():
            avg_probs = np.mean(np.stack(prob_list, axis=0), axis=0)
            metrics = evaluate_probs(y_test, class_labels, avg_probs)
            append_prediction_diagnostics(
                prediction_records,
                class_records,
                confusion_records,
                int(test_subj),
                branch,
                "equal_weight",
                y_test,
                class_labels,
                avg_probs,
            )
            avg_val_probs = np.mean(np.stack(per_branch_val_probs[branch], axis=0), axis=0)
            val_metrics = evaluate_probs(y_val_fold, class_labels, avg_val_probs)
            record = {
                "subject_id": int(test_subj),
                "branch": branch,
                "selection": "equal_weight",
                "n_seeds": int(len(seeds)),
                "n_configs": int(len(fixed_configs)),
                "n_models": int(len(prob_list)),
                "n_selected_models": int(len(prob_list)),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "selected_model_indices": "",
                "selected_weights": "",
            }
            records.append(record)
            print(
                f"  {branch:>8s} ensemble acc={metrics['accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )
            weighted_probs, weights, weighted_val_metrics, selected_indices, val_diagnostics = select_validation_weights(
                y_val_fold,
                class_labels,
                per_branch_val_probs[branch],
                per_branch_test_probs[branch],
                step=val_weight_step,
                max_models=min(max_weight_grid_models, len(prob_list)),
            )
            weighted_metrics = evaluate_probs(y_test, class_labels, weighted_probs)
            append_prediction_diagnostics(
                prediction_records,
                class_records,
                confusion_records,
                int(test_subj),
                branch,
                "validation_weighted",
                y_test,
                class_labels,
                weighted_probs,
            )
            weighted_record = {
                "subject_id": int(test_subj),
                "branch": branch,
                "selection": "validation_weighted",
                "n_seeds": int(len(seeds)),
                "n_configs": int(len(fixed_configs)),
                "n_models": int(len(prob_list)),
                "n_selected_models": int(len(selected_indices)),
                "accuracy": weighted_metrics["accuracy"],
                "macro_f1": weighted_metrics["macro_f1"],
                "val_accuracy": weighted_val_metrics["accuracy"],
                "val_macro_f1": weighted_val_metrics["macro_f1"],
                "selected_model_indices": json.dumps([int(i) for i in selected_indices]),
                "selected_weights": json.dumps([float(w) for w in weights]),
                **val_diagnostics,
            }
            records.append(weighted_record)
            print(
                f"  {branch:>8s} val-weighted acc={weighted_metrics['accuracy']:.4f}, "
                f"macro_f1={weighted_metrics['macro_f1']:.4f}, "
                f"val_f1={weighted_val_metrics['macro_f1']:.4f}"
            )

    raw_df = pd.DataFrame(records)
    raw_path = os.path.join(output_dir, "seed_ensemble_by_fold_branch.csv")
    raw_df.to_csv(raw_path, index=False)

    predictions_path = os.path.join(output_dir, "seed_ensemble_predictions.csv")
    pd.DataFrame(prediction_records).to_csv(predictions_path, index=False)
    class_metrics_path = os.path.join(output_dir, "seed_ensemble_class_metrics.csv")
    pd.DataFrame(class_records).to_csv(class_metrics_path, index=False)
    confusion_path = os.path.join(output_dir, "seed_ensemble_confusion_long.csv")
    pd.DataFrame(confusion_records).to_csv(confusion_path, index=False)

    summary_rows = []
    for (branch, selection), group in raw_df.groupby(["branch", "selection"]):
        acc_ci = _ci95(group["accuracy"].to_numpy())
        f1_ci = _ci95(group["macro_f1"].to_numpy())
        summary_rows.append(
            {
                "branch": branch,
                "selection": selection,
                "n_folds": int(len(group)),
                "mean_accuracy": acc_ci["mean"],
                "std_accuracy": acc_ci["std"],
                "ci95_accuracy": acc_ci["ci95_half_width"],
                "mean_macro_f1": f1_ci["mean"],
                "std_macro_f1": f1_ci["std"],
                "ci95_macro_f1": f1_ci["ci95_half_width"],
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_macro_f1", ascending=False)
    summary_path = os.path.join(output_dir, "seed_ensemble_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    summary = {
        "seeds": [int(s) for s in seeds],
        "n_seeds": int(len(seeds)),
        "fixed_hyperparams_paths": fixed_hyperparams_paths,
        "n_configs": int(len(fixed_configs)),
        "sample_normalization": sample_normalization,
        "final_val_ratio": float(final_val_ratio),
        "final_val_on_train": bool(final_val_on_train),
        "group_aware_final_val": bool(group_aware_final_val),
        "val_weight_step": float(val_weight_step),
        "max_weight_grid_models": int(max_weight_grid_models),
        "targeted_augment_profile": targeted_augment_profile,
        "branches": summary_df.to_dict(orient="records"),
    }
    summary_json = os.path.join(output_dir, "seed_ensemble_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs:")
    print(f"  - {raw_path}")
    print(f"  - {predictions_path}")
    print(f"  - {class_metrics_path}")
    print(f"  - {confusion_path}")
    print(f"  - {summary_path}")
    print(f"  - {summary_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DA-LGBM LOSO seed-probability ensemble")
    parser.add_argument("--csv_dir", default="datasets/gesture_csv")
    parser.add_argument("--output_dir", default="outputs/DA_LGBM_seed_ensemble_loso")
    parser.add_argument("--seeds", default="42,123,2024")
    parser.add_argument("--fixed_hyperparams_path", default="")
    parser.add_argument(
        "--fixed_hyperparams_paths",
        default="",
        help="Comma-separated JSON paths. If set, overrides --fixed_hyperparams_path.",
    )
    parser.add_argument(
        "--sample_normalization",
        choices=["none", "center", "baseline", "zscore", "robust_zscore"],
        default="none",
    )
    parser.add_argument("--final_val_ratio", type=float, default=0.2)
    parser.add_argument("--final_val_on_train", action="store_true")
    parser.add_argument("--group_aware_final_val", action="store_true")
    parser.add_argument("--optuna_seed", type=int, default=42)
    parser.add_argument(
        "--val_weight_step",
        type=float,
        default=0.05,
        help="Simplex grid step for validation-selected ensemble weights.",
    )
    parser.add_argument(
        "--max_weight_grid_models",
        type=int,
        default=5,
        help="Use the top-k validation models for grid weight search.",
    )
    parser.add_argument(
        "--targeted_augment_profile",
        choices=["none", "hard_017_mild", "hard_017_strong"],
        default="none",
        help="Optional class-targeted augmentation profile for confusion-heavy classes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.fixed_hyperparams_paths.strip():
        fixed_paths = [p.strip() for p in args.fixed_hyperparams_paths.split(",") if p.strip()]
    elif args.fixed_hyperparams_path.strip():
        fixed_paths = [args.fixed_hyperparams_path.strip()]
    else:
        raise ValueError("Provide --fixed_hyperparams_path or --fixed_hyperparams_paths")
    run_seed_ensemble(
        csv_dir=args.csv_dir,
        output_dir=args.output_dir,
        seeds=parse_seeds(args.seeds),
        fixed_hyperparams_paths=fixed_paths,
        sample_normalization=args.sample_normalization,
        final_val_ratio=args.final_val_ratio,
        final_val_on_train=args.final_val_on_train,
        group_aware_final_val=args.group_aware_final_val,
        optuna_seed=args.optuna_seed,
        val_weight_step=args.val_weight_step,
        max_weight_grid_models=args.max_weight_grid_models,
        targeted_augment_profile=args.targeted_augment_profile,
    )
