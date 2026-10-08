#!/usr/bin/env python3
"""
Augmentation probability sensitivity under LOSO for DA-LGBM.

This experiment fixes the LOSO protocol and all augmentation strengths, then
sweeps only the probability p used by the training-time augmentations. It saves
raw fold/seed records, summary CSV/LaTeX tables, and publication-ready plots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.training.train_adann_lightgbm import AdannLightgbmModelCreator
from experiments.da_lgbm_seed_stability_loso import (
    _ci95,
    _sample_std,
    _to_basic_type,
    apply_sample_normalization,
    augment_data_with_subjects,
    evaluate_branch_accuracies,
    load_data,
    macro_f1_from_cm,
    parse_seeds,
    set_all_seeds,
    split_train_val,
)


DEFAULT_PROBABILITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_BASE_PARAMS: Dict[str, Any] = {
    "adann_learning_rate": 1e-3,
    "adann_feature_size": 64,
    "adann_dropout": 0.3,
    "adann_classifier_dropout": 0.2,
    "adann_epochs": 120,
    "gesture_loss_weight": 1.0,
    "domain_loss_weight": 1.0,
    "grl_gamma": 10.0,
    "grl_max": 0.99,
    "class_balanced_batches": False,
    "batch_size": 32,
    "lgb_num_leaves": 31,
    "lgb_learning_rate": 0.10,
    "lgb_feature_fraction": 0.80,
    "lgb_bagging_fraction": 0.80,
    "lgb_min_child_samples": 20,
    "lgb_n_estimators": 120,
    "lgb_max_depth": 8,
    "ensemble_adann_weight": 0.5,
    "auto_tune_ensemble_weight": False,
    "auto_tune_gate_thresholds": False,
    "augment_factor": 1,
    "jitter_noise_level": 0.005,
    "time_warp_max_speed": 2.0,
    "scale_range": [0.98, 1.02],
    "augment_prob": 0.3,
    "channel_augment_prob": 0.0,
    "channel_scale_min": 1.0,
    "channel_scale_max": 1.0,
    "channel_shift_level": 0.0,
    "channel_drift_level": 0.0,
}


def parse_float_list(value: str) -> List[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one probability value.")
    return [float(item) for item in items]


def parse_int_list(value: str) -> Optional[List[int]]:
    value = (value or "").strip()
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_fold_params(path: str) -> Optional[Dict[int, Dict[str, Any]]]:
    if not path:
        return None
    with open(path, "r") as f:
        loaded = json.load(f)
    if (
        isinstance(loaded, dict)
        and loaded
        and all(str(k).lstrip("-").isdigit() for k in loaded.keys())
        and all(isinstance(v, dict) for v in loaded.values())
    ):
        return {int(k): {kk: _to_basic_type(vv) for kk, vv in v.items()} for k, v in loaded.items()}
    params = (
        loaded.get("best_hyperparameters_for_deployment")
        or loaded.get("best_params")
        or loaded
    )
    return {-1: {k: _to_basic_type(v) for k, v in params.items()}}


def prepare_params(
    base_params_by_fold: Optional[Dict[int, Dict[str, Any]]],
    subject_id: int,
    augment_prob: float,
    preserve_float: bool,
    force_augment_factor: Optional[int],
    force_augmentation_strength: bool,
    jitter_noise_level: float,
    time_warp_max_speed: float,
    scale_min: float,
    scale_max: float,
    adann_epochs_override: Optional[int],
) -> Dict[str, Any]:
    if base_params_by_fold is None:
        params = dict(DEFAULT_BASE_PARAMS)
    else:
        params = dict(base_params_by_fold.get(subject_id) or base_params_by_fold.get(-1) or {})
        merged = dict(DEFAULT_BASE_PARAMS)
        merged.update(params)
        params = merged

    params["augment_prob"] = float(augment_prob)
    params["preserve_float"] = bool(preserve_float)
    if force_augment_factor is not None:
        params["augment_factor"] = int(force_augment_factor)
    if force_augmentation_strength:
        params["jitter_noise_level"] = float(jitter_noise_level)
        params["time_warp_max_speed"] = float(time_warp_max_speed)
        params["scale_min"] = float(scale_min)
        params["scale_max"] = float(scale_max)
        params.pop("scale_range", None)
    if adann_epochs_override is not None:
        params["adann_epochs"] = int(adann_epochs_override)
    params["auto_tune_gate_thresholds"] = bool(params.get("auto_tune_gate_thresholds", False))
    return params


def augment_for_probability(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    params: Dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply training augmentation, treating p=0 as the no-augmentation baseline."""
    if float(params.get("augment_prob", 0.0)) <= 0.0:
        return X_train, y_train, subjects_train
    return augment_data_with_subjects(
        X_train,
        y_train,
        subjects_train,
        params,
        seed=seed,
    )


def select_subjects(all_subjects: np.ndarray, subjects_arg: str, max_subjects: int) -> np.ndarray:
    unique_subjects = np.unique(all_subjects[all_subjects != -1])
    requested = parse_int_list(subjects_arg)
    if requested is not None:
        unique_subjects = np.asarray([s for s in unique_subjects if int(s) in set(requested)], dtype=int)
    if max_subjects and max_subjects > 0:
        unique_subjects = unique_subjects[:max_subjects]
    if len(unique_subjects) < 1:
        raise RuntimeError("No valid LOSO subjects selected.")
    return unique_subjects


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prob, group in raw_df.groupby("augment_prob", sort=True):
        acc_ci = _ci95(group["accuracy"].to_numpy())
        f1_ci = _ci95(group["macro_f1"].to_numpy())
        rows.append(
            {
                "augment_prob": float(prob),
                "mean_accuracy": acc_ci["mean"],
                "std_accuracy": acc_ci["std"],
                "ci95_accuracy": acc_ci["ci95_half_width"],
                "mean_macro_f1": f1_ci["mean"],
                "std_macro_f1": f1_ci["std"],
                "ci95_macro_f1": f1_ci["ci95_half_width"],
                "n_records": int(len(group)),
                "n_seeds": int(group["seed"].nunique()),
                "n_folds": int(group["subject_id"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("augment_prob")


def summarize_by_seed(raw_df: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_df.groupby(["augment_prob", "seed"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", _sample_std),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", _sample_std),
            n_folds=("subject_id", "count"),
        )
        .sort_values(["augment_prob", "seed"])
    )


def save_plot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    x = summary_df["augment_prob"].to_numpy(dtype=float)
    f1 = summary_df["mean_macro_f1"].to_numpy(dtype=float)
    f1_err = summary_df["ci95_macro_f1"].to_numpy(dtype=float)
    acc = summary_df["mean_accuracy"].to_numpy(dtype=float)
    acc_err = summary_df["ci95_accuracy"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
    ax.errorbar(
        x,
        f1,
        yerr=f1_err,
        marker="o",
        linewidth=1.7,
        capsize=3,
        color="#1f77b4",
        label="Macro-F1",
    )
    ax.errorbar(
        x,
        acc,
        yerr=acc_err,
        marker="s",
        linewidth=1.3,
        linestyle="--",
        capsize=3,
        color="#2ca02c",
        label="Accuracy",
    )
    ax.set_xlabel("Augmentation probability p")
    ax.set_ylabel("LOSO score")
    ax.set_xticks(x)
    ax.set_ylim(max(0.0, min(float(np.min(f1 - f1_err)), float(np.min(acc - acc_err))) - 0.04), 1.0)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"augmentation_probability_sweep.{suffix}", bbox_inches="tight")
    plt.close(fig)


def save_tex(summary_df: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "\\begin{tabular}{c c c}",
        "\\hline",
        "$p$ & Accuracy & Macro-F1 \\\\",
        "\\hline",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['augment_prob']:.1f} & "
            f"{row['mean_accuracy'] * 100:.2f} $\\pm$ {row['ci95_accuracy'] * 100:.2f} & "
            f"{row['mean_macro_f1'] * 100:.2f} $\\pm$ {row['ci95_macro_f1'] * 100:.2f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    (output_dir / "augmentation_probability_table.tex").write_text("\n".join(lines))


def run_sweep(args: argparse.Namespace) -> None:
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = parse_float_list(args.probabilities)
    seeds = parse_seeds(args.seeds)
    base_params_by_fold = load_fold_params(args.fixed_hyperparams_path)

    print(f"Loading data from: {args.csv_dir}")
    X, y, subjects = load_data(str(PROJECT_ROOT / args.csv_dir))
    sample_normalization = (args.sample_normalization or "none").lower()
    preserve_float = sample_normalization in ("zscore", "robust_zscore")
    X = apply_sample_normalization(X, sample_normalization)
    selected_subjects = select_subjects(subjects, args.subjects, args.max_subjects)
    all_labels = np.unique(y)

    print(f"Samples: {len(X)}")
    print(f"Subjects: {[int(s) for s in selected_subjects]}")
    print(f"Seeds: {seeds}")
    print(f"Probabilities: {probabilities}")
    if args.fixed_hyperparams_path:
        print(f"Base hyperparameters: {args.fixed_hyperparams_path}")

    records: List[Dict[str, Any]] = []
    for prob in probabilities:
        print(f"\n========== p = {prob:.2f} ==========")
        for seed in seeds:
            set_all_seeds(seed)
            for test_subj in selected_subjects:
                print(f"--- seed={seed}, held-out subject={int(test_subj)} ---")
                test_mask = subjects == test_subj
                train_mask = ~test_mask
                X_train = X[train_mask]
                y_train = y[train_mask]
                subj_train = subjects[train_mask]
                X_test = X[test_mask]
                y_test = y[test_mask]

                split_seed = args.split_seed + int(test_subj) * 1000 + int(seed)
                if args.final_val_on_train:
                    X_train_fold, y_train_fold, subj_train_fold = X_train, y_train, subj_train
                    X_val_fold, y_val_fold, subj_val_fold = X_train, y_train, subj_train
                else:
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
                        split_seed=split_seed,
                        val_ratio=args.final_val_ratio,
                        group_aware=args.group_aware_final_val,
                    )

                force_factor = None if args.keep_base_augment_factor else args.augment_factor
                params = prepare_params(
                    base_params_by_fold=base_params_by_fold,
                    subject_id=int(test_subj),
                    augment_prob=float(prob),
                    preserve_float=preserve_float,
                    force_augment_factor=force_factor,
                    force_augmentation_strength=args.force_augmentation_strength,
                    jitter_noise_level=args.jitter_noise_level,
                    time_warp_max_speed=args.time_warp_max_speed,
                    scale_min=args.scale_min,
                    scale_max=args.scale_max,
                    adann_epochs_override=args.adann_epochs_override,
                )
                set_all_seeds(seed)
                X_aug, y_aug, subj_aug = augment_for_probability(
                    X_train_fold,
                    y_train_fold,
                    subj_train_fold,
                    params,
                    seed=seed + int(test_subj) * 10000 + int(round(prob * 1000)),
                )

                creator = AdannLightgbmModelCreator()
                wrapper = creator.create_model(params, arduino_mode=False)
                wrapper.hybrid_model["lightgbm"].set_params(random_state=seed, n_jobs=1)
                trained_model, _, _ = creator.train_model(
                    wrapper.hybrid_model,
                    X_aug,
                    y_aug,
                    subj_aug,
                    X_val_fold,
                    y_val_fold,
                    subj_val_fold,
                    params,
                    return_history=True,
                )
                branch_eval = evaluate_branch_accuracies(
                    creator=creator,
                    trained_model=trained_model,
                    X_test=X_test,
                    y_test=y_test,
                    selection_mode=args.selection_mode,
                    selection_margin=args.selection_margin,
                )

                if args.selection_mode == "lgb":
                    y_pred = branch_eval["lgb_pred_decoded"]
                    accuracy = float(branch_eval["lgb_test_acc"])
                elif args.selection_mode == "adann":
                    y_pred = branch_eval["adann_pred_decoded"]
                    accuracy = float(branch_eval["adann_test_acc"])
                elif args.selection_mode == "gated":
                    y_pred = branch_eval["gated_pred_decoded"]
                    accuracy = float(branch_eval["gated_test_acc"])
                elif args.selection_mode == "ensemble":
                    y_pred = branch_eval["ensemble_pred_decoded"]
                    accuracy = float(branch_eval["ensemble_test_acc"])
                else:
                    y_pred = branch_eval["selected_pred_decoded"]
                    accuracy = float(branch_eval["selected_test_acc"])

                cm = confusion_matrix(y_test, y_pred, labels=all_labels)
                macro_f1 = macro_f1_from_cm(cm)
                print(f"  Accuracy={accuracy:.4f}, Macro-F1={macro_f1:.4f}")
                records.append(
                    {
                        "augment_prob": float(prob),
                        "seed": int(seed),
                        "subject_id": int(test_subj),
                        "n_train_original": int(len(X_train_fold)),
                        "n_train_augmented": int(len(X_aug)),
                        "n_test": int(len(X_test)),
                        "accuracy": float(accuracy),
                        "macro_f1": float(macro_f1),
                        "selected_branch": branch_eval["selected_branch"],
                        "adann_test_acc": float(branch_eval["adann_test_acc"]),
                        "lgb_test_acc": float(branch_eval["lgb_test_acc"]),
                        "ensemble_test_acc": float(branch_eval["ensemble_test_acc"]),
                        "gated_test_acc": float(branch_eval["gated_test_acc"]),
                        "adann_macro_f1": float(branch_eval["adann_macro_f1"]),
                        "lgb_macro_f1": float(branch_eval["lgb_macro_f1"]),
                        "ensemble_macro_f1": float(branch_eval["ensemble_macro_f1"]),
                        "gated_macro_f1": float(branch_eval["gated_macro_f1"]),
                        "ensemble_weight": float(trained_model["ensemble_weight"]),
                        "augment_factor": int(params.get("augment_factor", 0)),
                        "jitter_noise_level": float(params.get("jitter_noise_level", 0.0)),
                        "time_warp_max_speed": float(params.get("time_warp_max_speed", 0.0)),
                        "scale_min": float(params.get("scale_min", params.get("scale_range", [0.98, 1.02])[0])),
                        "scale_max": float(params.get("scale_max", params.get("scale_range", [0.98, 1.02])[1])),
                        "selection_mode": args.selection_mode,
                        "sample_normalization": sample_normalization,
                    }
                )

    raw_df = pd.DataFrame.from_records(records)
    summary_df = summarize(raw_df)
    by_seed_df = summarize_by_seed(raw_df)

    raw_df.to_csv(output_dir / "augmentation_probability_raw.csv", index=False)
    summary_df.to_csv(output_dir / "augmentation_probability_summary.csv", index=False)
    by_seed_df.to_csv(output_dir / "augmentation_probability_by_seed.csv", index=False)
    save_plot(summary_df, output_dir)
    save_tex(summary_df, output_dir)

    best_idx = summary_df["mean_macro_f1"].idxmax()
    metadata = {
        "probabilities": probabilities,
        "seeds": seeds,
        "subjects": [int(s) for s in selected_subjects],
        "selection_mode": args.selection_mode,
        "sample_normalization": sample_normalization,
        "fixed_hyperparams_path": args.fixed_hyperparams_path,
        "keep_base_augment_factor": bool(args.keep_base_augment_factor),
        "augment_factor": None if args.keep_base_augment_factor else int(args.augment_factor),
        "adann_epochs_override": args.adann_epochs_override,
        "best_probability_by_macro_f1": float(summary_df.loc[best_idx, "augment_prob"]),
        "best_mean_macro_f1": float(summary_df.loc[best_idx, "mean_macro_f1"]),
    }
    with (output_dir / "augmentation_probability_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved outputs:")
    print(f"  - {output_dir / 'augmentation_probability_raw.csv'}")
    print(f"  - {output_dir / 'augmentation_probability_summary.csv'}")
    print(f"  - {output_dir / 'augmentation_probability_by_seed.csv'}")
    print(f"  - {output_dir / 'augmentation_probability_sweep.png'}")
    print(f"  - {output_dir / 'augmentation_probability_table.tex'}")
    print("\nSummary:")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DA-LGBM LOSO augmentation probability sensitivity sweep"
    )
    parser.add_argument("--csv_dir", default="datasets/gesture_csv")
    parser.add_argument("--output_dir", default="outputs/augmentation_probability_sweep")
    parser.add_argument("--probabilities", default=",".join(str(p) for p in DEFAULT_PROBABILITIES))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--subjects", default="", help="Optional comma-separated held-out subject IDs")
    parser.add_argument("--max_subjects", type=int, default=0, help="Optional smoke-test subject limit")
    parser.add_argument(
        "--fixed_hyperparams_path",
        default="",
        help="Optional fold_best_hyperparams.json or single params JSON. Only p is swept.",
    )
    parser.add_argument("--sample_normalization", default="none", choices=["none", "center", "baseline", "zscore", "robust_zscore"])
    parser.add_argument("--selection_mode", default="gated", choices=["best_val_branch", "robust_val_branch", "ensemble", "gated", "lgb", "adann"])
    parser.add_argument("--selection_margin", type=float, default=0.01)
    parser.add_argument("--final_val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--group_aware_final_val", action="store_true")
    parser.add_argument("--final_val_on_train", action="store_true")
    parser.add_argument("--augment_factor", type=int, default=1)
    parser.add_argument("--jitter_noise_level", type=float, default=0.005)
    parser.add_argument("--time_warp_max_speed", type=float, default=2.0)
    parser.add_argument("--scale_min", type=float, default=0.98)
    parser.add_argument("--scale_max", type=float, default=1.02)
    parser.add_argument(
        "--force_augmentation_strength",
        action="store_true",
        help="Force jitter/time-warp/scale strengths from CLI while sweeping only augment_prob.",
    )
    parser.add_argument(
        "--keep_base_augment_factor",
        action="store_true",
        default=True,
        help="Keep augment_factor from fixed/base params instead of forcing --augment_factor.",
    )
    parser.add_argument(
        "--force_augment_factor",
        dest="keep_base_augment_factor",
        action="store_false",
        help="Force --augment_factor instead of keeping the value from fixed/base params.",
    )
    parser.add_argument(
        "--adann_epochs_override",
        type=int,
        default=None,
        help="Debug/smoke option. Leave unset for publication runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
