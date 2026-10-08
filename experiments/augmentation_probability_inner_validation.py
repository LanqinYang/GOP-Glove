#!/usr/bin/env python3
"""
Inner-validation sensitivity of DA-LGBM to augmentation probability.

This script is intentionally different from a LOSO test-set re-run. For each
outer LOSO subject, the held-out subject is excluded entirely; the remaining
subjects are split into train/validation, and p is swept only on that inner
validation split. This provides support for the training augmentation setting
without replacing the final LOSO test result in the manuscript.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.training.train_adann_lightgbm import AdannLightgbmModelCreator
from experiments.augmentation_probability_sweep import (
    DEFAULT_BASE_PARAMS,
    augment_for_probability,
    load_fold_params,
    parse_float_list,
    prepare_params,
    select_subjects,
)
from experiments.da_lgbm_seed_stability_loso import (
    _ci95,
    apply_sample_normalization,
    load_data,
    parse_seeds,
    set_all_seeds,
    split_train_val,
)


DEFAULT_PROBABILITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prob, group in raw_df.groupby("augment_prob", sort=True):
        acc_ci = _ci95(group["val_accuracy"].to_numpy())
        f1_ci = _ci95(group["val_macro_f1"].to_numpy())
        rows.append(
            {
                "augment_prob": float(prob),
                "mean_val_accuracy": acc_ci["mean"],
                "std_val_accuracy": acc_ci["std"],
                "ci95_val_accuracy": acc_ci["ci95_half_width"],
                "mean_val_macro_f1": f1_ci["mean"],
                "std_val_macro_f1": f1_ci["std"],
                "ci95_val_macro_f1": f1_ci["ci95_half_width"],
                "n_records": int(len(group)),
                "n_seeds": int(group["seed"].nunique()),
                "n_outer_folds": int(group["outer_subject_id"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("augment_prob")


def save_plot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    x = summary_df["augment_prob"].to_numpy(float)
    y = summary_df["mean_val_macro_f1"].to_numpy(float) * 100.0
    yerr = summary_df["ci95_val_macro_f1"].to_numpy(float) * 100.0

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.75,
        }
    )

    fig, ax = plt.subplots(figsize=(3.2, 2.0), dpi=300)
    line_color = "#2563A9"
    fill_color = "#8CB9E8"
    highlight_color = "#C2410C"

    ax.fill_between(
        x,
        y - yerr,
        y + yerr,
        color=fill_color,
        alpha=0.24,
        linewidth=0,
    )
    ax.plot(
        x,
        y,
        color=line_color,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.1,
        linewidth=1.45,
    )
    best_idx = int(np.argmax(y))
    ax.scatter(
        [x[best_idx]],
        [y[best_idx]],
        s=30,
        color=highlight_color,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.axvline(x[best_idx], color=highlight_color, linewidth=0.75, linestyle=(0, (3, 2)), alpha=0.65)
    ax.annotate(
        f"p={x[best_idx]:.1f}",
        xy=(x[best_idx], y[best_idx]),
        xytext=(x[best_idx] + 0.045, y[best_idx] + 0.55),
        fontsize=6.5,
        color=highlight_color,
        arrowprops=dict(arrowstyle="-", color=highlight_color, linewidth=0.7),
    )
    ax.text(
        0.03,
        0.08,
        "mean ± 95% CI",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#52616F",
    )
    ax.set_xlabel("Augmentation probability p")
    ax.set_ylabel("Inner-val Macro-F1 (%)")
    ax.set_xticks(x)
    y_min = float(np.floor((np.min(y - yerr) - 0.4) * 2) / 2)
    y_max = float(np.ceil((np.max(y + yerr) + 0.4) * 2) / 2)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.55)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3, width=0.7)
    fig.tight_layout()

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"augmentation_probability_inner_validation.{suffix}", bbox_inches="tight")
    plt.close(fig)


def save_tex(summary_df: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "\\begin{tabular}{c c c}",
        "\\hline",
        "$p$ & Inner-val Accuracy & Inner-val Macro-F1 \\\\",
        "\\hline",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['augment_prob']:.1f} & "
            f"{row['mean_val_accuracy'] * 100:.2f} $\\pm$ {row['ci95_val_accuracy'] * 100:.2f} & "
            f"{row['mean_val_macro_f1'] * 100:.2f} $\\pm$ {row['ci95_val_macro_f1'] * 100:.2f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    (output_dir / "augmentation_probability_inner_validation_table.tex").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = parse_float_list(args.probabilities)
    seeds = parse_seeds(args.seeds)
    base_params_by_fold = load_fold_params(args.fixed_hyperparams_path)

    X, y, subjects = load_data(str(PROJECT_ROOT / args.csv_dir))
    sample_normalization = (args.sample_normalization or "none").lower()
    preserve_float = sample_normalization in ("zscore", "robust_zscore")
    X = apply_sample_normalization(X, sample_normalization)
    selected_subjects = select_subjects(subjects, args.subjects, args.max_subjects)

    print(f"Samples: {len(X)}")
    print(f"Outer subjects excluded from inner validation: {[int(s) for s in selected_subjects]}")
    print(f"Seeds: {seeds}")
    print(f"Probabilities: {probabilities}")
    print("Held-out LOSO test subjects are not evaluated in this script.")

    records: List[Dict[str, Any]] = []
    for prob in probabilities:
        print(f"\n========== p = {prob:.2f} ==========")
        for seed in seeds:
            for outer_subj in selected_subjects:
                print(f"--- seed={seed}, outer held-out subject={int(outer_subj)} ---")
                non_test_mask = subjects != outer_subj
                X_non_test = X[non_test_mask]
                y_non_test = y[non_test_mask]
                subj_non_test = subjects[non_test_mask]

                split_seed = args.split_seed + int(outer_subj) * 1000 + int(seed)
                (
                    X_train,
                    y_train,
                    subj_train,
                    X_val,
                    y_val,
                    subj_val,
                ) = split_train_val(
                    X_non_test,
                    y_non_test,
                    subj_non_test,
                    split_seed=split_seed,
                    val_ratio=args.val_ratio,
                    group_aware=args.group_aware_val,
                )

                params = prepare_params(
                    base_params_by_fold=base_params_by_fold,
                    subject_id=int(outer_subj),
                    augment_prob=float(prob),
                    preserve_float=preserve_float,
                    force_augment_factor=args.augment_factor,
                    force_augmentation_strength=True,
                    jitter_noise_level=args.jitter_noise_level,
                    time_warp_max_speed=args.time_warp_max_speed,
                    scale_min=args.scale_min,
                    scale_max=args.scale_max,
                    adann_epochs_override=args.adann_epochs_override,
                )

                set_all_seeds(seed)
                X_aug, y_aug, subj_aug = augment_for_probability(
                    X_train,
                    y_train,
                    subj_train,
                    params,
                    seed=seed + int(outer_subj) * 10000 + int(round(prob * 1000)),
                )

                creator = AdannLightgbmModelCreator()
                wrapper = creator.create_model(params, arduino_mode=False)
                wrapper.hybrid_model["lightgbm"].set_params(random_state=seed, n_jobs=1)
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

                val_key = f"val_macro_f1_{args.selection_mode}"
                acc_key = f"val_accuracy_{args.selection_mode}"
                val_f1 = float(trained_model.get(val_key, trained_model.get("val_macro_f1_gated", 0.0)))
                val_acc = float(trained_model.get(acc_key, trained_model.get("val_accuracy_gated", 0.0)))
                print(f"  Inner-val acc={val_acc:.4f}, Macro-F1={val_f1:.4f}")

                records.append(
                    {
                        "augment_prob": float(prob),
                        "seed": int(seed),
                        "outer_subject_id": int(outer_subj),
                        "n_train_original": int(len(X_train)),
                        "n_train_augmented": int(len(X_aug)),
                        "n_val": int(len(X_val)),
                        "val_accuracy": val_acc,
                        "val_macro_f1": val_f1,
                        "val_accuracy_adann": float(trained_model.get("val_accuracy_adann", 0.0)),
                        "val_accuracy_lgb": float(trained_model.get("val_accuracy_lgb", 0.0)),
                        "val_accuracy_ensemble": float(trained_model.get("val_accuracy_ensemble", 0.0)),
                        "val_accuracy_gated": float(trained_model.get("val_accuracy_gated", 0.0)),
                        "val_macro_f1_adann": float(trained_model.get("val_macro_f1_adann", 0.0)),
                        "val_macro_f1_lgb": float(trained_model.get("val_macro_f1_lgb", 0.0)),
                        "val_macro_f1_ensemble": float(trained_model.get("val_macro_f1_ensemble", 0.0)),
                        "val_macro_f1_gated": float(trained_model.get("val_macro_f1_gated", 0.0)),
                        "selection_mode": args.selection_mode,
                        "augment_factor": int(params.get("augment_factor", 0)),
                        "jitter_noise_level": float(params.get("jitter_noise_level", 0.0)),
                        "time_warp_max_speed": float(params.get("time_warp_max_speed", 0.0)),
                        "scale_min": float(params.get("scale_min", 0.0)),
                        "scale_max": float(params.get("scale_max", 0.0)),
                    }
                )

    raw_df = pd.DataFrame.from_records(records)
    summary_df = summarize(raw_df)
    raw_df.to_csv(output_dir / "augmentation_probability_inner_validation_raw.csv", index=False)
    summary_df.to_csv(output_dir / "augmentation_probability_inner_validation_summary.csv", index=False)
    save_plot(summary_df, output_dir)
    save_tex(summary_df, output_dir)

    best_idx = summary_df["mean_val_macro_f1"].idxmax()
    metadata = {
        "probabilities": probabilities,
        "seeds": seeds,
        "outer_subjects": [int(s) for s in selected_subjects],
        "selection_mode": args.selection_mode,
        "val_ratio": float(args.val_ratio),
        "group_aware_val": bool(args.group_aware_val),
        "heldout_loso_test_subject_used": False,
        "fixed_hyperparams_path": args.fixed_hyperparams_path,
        "augment_factor": int(args.augment_factor),
        "jitter_noise_level": float(args.jitter_noise_level),
        "time_warp_max_speed": float(args.time_warp_max_speed),
        "scale_min": float(args.scale_min),
        "scale_max": float(args.scale_max),
        "best_probability_by_inner_val_macro_f1": float(summary_df.loc[best_idx, "augment_prob"]),
        "best_mean_inner_val_macro_f1": float(summary_df.loc[best_idx, "mean_val_macro_f1"]),
    }
    with (output_dir / "augmentation_probability_inner_validation_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSummary:")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inner-validation augmentation probability sensitivity")
    parser.add_argument("--csv_dir", default="datasets/gesture_csv")
    parser.add_argument("--output_dir", default="outputs/augmentation_probability_inner_validation")
    parser.add_argument("--probabilities", default=",".join(str(p) for p in DEFAULT_PROBABILITIES))
    parser.add_argument("--seeds", default="42,123,2024,2025,3407")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--max_subjects", type=int, default=0)
    parser.add_argument("--fixed_hyperparams_path", default="")
    parser.add_argument("--sample_normalization", default="none", choices=["none", "center", "baseline", "zscore", "robust_zscore"])
    parser.add_argument("--selection_mode", default="gated", choices=["adann", "lgb", "ensemble", "gated"])
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--group_aware_val", action="store_true")
    parser.add_argument("--augment_factor", type=int, default=1)
    parser.add_argument("--jitter_noise_level", type=float, default=0.005)
    parser.add_argument("--time_warp_max_speed", type=float, default=2.0)
    parser.add_argument("--scale_min", type=float, default=0.98)
    parser.add_argument("--scale_max", type=float, default=1.02)
    parser.add_argument("--adann_epochs_override", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
