#!/usr/bin/env python3
"""
Controlled sampling-jitter robustness experiment for LOSO evaluation.

The script trains each model using clean LOSO training subjects, perturbs only
the held-out test windows by simulating sampling-interval variation, and
reports Macro-F1 degradation at clean, +/-2%, +/-5%, and +/-10% jitter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.awgn_robustness import (  # noqa: E402
    DEFAULT_MAIN_OUTPUT_MODELS,
    DISPLAY_NAMES,
    MODEL_ALIASES,
    SEED,
    VAL_SIZE,
    augment_with_subjects,
    load_best_params,
    load_data,
    maybe_cap_epochs,
    predict_labels,
    set_seeds,
    train_one_fold,
    with_default_params,
)

DEFAULT_JITTER_LEVELS = ["clean", "0.02", "0.05", "0.10"]


def parse_jitter_levels(values: Iterable[str]) -> List[Optional[float]]:
    parsed: List[Optional[float]] = []
    for value in values:
        if str(value).lower() == "clean":
            parsed.append(None)
        else:
            level = float(value)
            if level < 0:
                raise ValueError("Jitter levels must be non-negative.")
            if level > 1.0:
                level = level / 100.0
            parsed.append(level)
    return parsed


def jitter_label(jitter_fraction: Optional[float]) -> str:
    if jitter_fraction is None or jitter_fraction == 0:
        return "clean"
    percent = 100.0 * float(jitter_fraction)
    return f"+/-{percent:g}%"


def jitter_sort_value(label: str) -> float:
    if label == "clean":
        return -1.0
    return float(label.replace("+/-", "").replace("%", ""))


def apply_sampling_jitter(
    X: np.ndarray,
    jitter_fraction: Optional[float],
    rng: np.random.Generator,
    clip_min: float = 0.0,
    clip_max: float = 1023.0,
) -> np.ndarray:
    """Simulate sampling-interval jitter and keep windows at 100 samples.

    Each interval is scaled by U(1-j, 1+j), then the perturbed cumulative time
    axis is normalized back to the original window duration. Values are sampled
    from the clean signal at these irregular timestamps, which creates a
    controlled timing-irregularity simulation without changing labels or folds.
    """
    if jitter_fraction is None or jitter_fraction == 0:
        return X.astype(np.float32, copy=True)

    X_float = X.astype(np.float32, copy=False)
    n_samples, n_steps, n_channels = X_float.shape
    base_time = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    jittered = np.empty_like(X_float, dtype=np.float32)

    low = max(0.0, 1.0 - float(jitter_fraction))
    high = 1.0 + float(jitter_fraction)
    for sample_idx in range(n_samples):
        intervals = rng.uniform(low, high, size=n_steps - 1).astype(np.float32)
        warped_time = np.concatenate(([0.0], np.cumsum(intervals))).astype(np.float32)
        warped_time /= warped_time[-1] if warped_time[-1] > 0 else 1.0
        for channel_idx in range(n_channels):
            jittered[sample_idx, :, channel_idx] = np.interp(
                warped_time,
                base_time,
                X_float[sample_idx, :, channel_idx],
            )

    return np.clip(jittered, clip_min, clip_max).astype(np.float32)


def ci95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(values.size))


def select_output_models(summary: pd.DataFrame, model_names: Iterable[str]) -> pd.DataFrame:
    selected = [MODEL_ALIASES[name] for name in model_names]
    filtered = summary[summary["model"].isin(selected)].copy()
    order_map = {model: idx for idx, model in enumerate(selected)}
    filtered["_model_order"] = filtered["model"].map(order_map)
    filtered["_jitter_order"] = filtered["jitter_label"].map(jitter_sort_value)
    filtered = filtered.sort_values(["_model_order", "_jitter_order"])
    return filtered.drop(columns=["_model_order", "_jitter_order"])


def iter_model_groups(summary: pd.DataFrame) -> Iterable[Tuple[str, pd.DataFrame]]:
    seen = []
    for _, row in summary[["model", "model_display"]].drop_duplicates().iterrows():
        seen.append((row["model"], row["model_display"]))
    for model, model_display in seen:
        yield model_display, summary[summary["model"] == model]


def ordered_labels(summary: pd.DataFrame) -> List[str]:
    labels = list(summary["jitter_label"].drop_duplicates())
    return sorted(labels, key=jitter_sort_value)


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    order = ordered_labels(summary)
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model_name, group in iter_model_groups(summary):
        group = group.set_index("jitter_label").reindex(order)
        ax.errorbar(
            x,
            group["macro_f1_mean"],
            yerr=group["macro_f1_ci95"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=model_name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("Sampling-interval perturbation")
    ax.set_ylabel("LOSO Macro-F1")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(3, summary["model_display"].nunique()))
    fig.tight_layout()
    fig.savefig(output_dir / "sampling_jitter_macro_f1.png", dpi=300)
    fig.savefig(output_dir / "sampling_jitter_macro_f1.pdf")
    fig.savefig(output_dir / "sampling_jitter_macro_f1.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model_name, group in iter_model_groups(summary):
        group = group.set_index("jitter_label").reindex(order)
        ax.plot(x, group["delta_macro_f1_mean"], marker="o", linewidth=1.8, label=model_name)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("Sampling-interval perturbation")
    ax.set_ylabel("Delta Macro-F1 vs clean")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(3, summary["model_display"].nunique()))
    fig.tight_layout()
    fig.savefig(output_dir / "sampling_jitter_delta_macro_f1.png", dpi=300)
    fig.savefig(output_dir / "sampling_jitter_delta_macro_f1.pdf")
    fig.savefig(output_dir / "sampling_jitter_delta_macro_f1.svg")
    plt.close(fig)


def plot_combined_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    order = ordered_labels(summary)
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharex=True)

    for model_name, group in iter_model_groups(summary):
        group = group.set_index("jitter_label").reindex(order)
        axes[0].errorbar(
            x,
            group["macro_f1_mean"],
            yerr=group["macro_f1_ci95"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=model_name,
        )
        axes[1].plot(x, group["delta_macro_f1_mean"], marker="o", linewidth=1.8, label=model_name)

    axes[0].set_title("(a) Macro-F1 under sampling jitter")
    axes[0].set_ylabel("LOSO Macro-F1")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_title("(b) Change from clean baseline")
    axes[1].set_ylabel("Delta Macro-F1 vs clean")
    axes[1].axhline(0.0, color="black", linewidth=0.8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_xlabel("Sampling-interval perturbation")
        ax.grid(True, axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=min(3, len(labels)))
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output_dir / "sampling_jitter_macro_f1_combined.png", dpi=300)
    fig.savefig(output_dir / "sampling_jitter_macro_f1_combined.pdf")
    fig.savefig(output_dir / "sampling_jitter_macro_f1_combined.svg")
    plt.close(fig)


def make_latex_table(summary: pd.DataFrame, output_dir: Path) -> None:
    order = ordered_labels(summary)
    rows = []
    for model_name, group in iter_model_groups(summary):
        group = group.set_index("jitter_label").reindex(order)
        row = [model_name]
        for label in order:
            record = group.loc[label]
            row.append(f"{record['macro_f1_mean']:.4f} $\\pm$ {record['macro_f1_ci95']:.4f}")
        rows.append(row)

    column_spec = "l" + "c" * len(order)
    header = "Model & " + " & ".join(order).replace("+/-", "$\\pm$") + " \\\\"
    lines = [
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (output_dir / "sampling_jitter_table.tex").write_text("\n".join(lines))


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    # Match the manuscript LOSO convention: repeated jitter realizations are
    # averaged within each held-out subject fold first, and 95% CIs are then
    # computed across the six fold-level means.
    fold_level = (
        raw.groupby(["model", "model_display", "jitter_label", "jitter_fraction", "fold"], sort=False)
        .agg(
            macro_f1=("macro_f1", "mean"),
            accuracy=("accuracy", "mean"),
            delta_macro_f1=("delta_macro_f1", "mean"),
            n_observations=("macro_f1", "size"),
        )
        .reset_index()
    )

    summary_rows = []
    group_cols = ["model", "model_display", "jitter_label", "jitter_fraction"]
    for (model_type, model_display, label, jitter_fraction), group in fold_level.groupby(group_cols, sort=False):
        macro = group["macro_f1"].to_numpy(dtype=float)
        acc = group["accuracy"].to_numpy(dtype=float)
        delta = group["delta_macro_f1"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "model": model_type,
                "model_display": model_display,
                "jitter_label": label,
                "jitter_fraction": jitter_fraction,
                "macro_f1_mean": float(macro.mean()),
                "macro_f1_std": float(macro.std(ddof=1)) if macro.size > 1 else 0.0,
                "macro_f1_ci95": ci95(macro),
                "accuracy_mean": float(acc.mean()),
                "accuracy_ci95": ci95(acc),
                "delta_macro_f1_mean": float(delta.mean()),
                "delta_macro_f1_ci95": ci95(delta),
                "n": int(len(group)),
                "n_observations": int(group["n_observations"].sum()),
                "ci_unit": "held-out subject folds",
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["_model_order"] = summary["model"].map(
        {model: idx for idx, model in enumerate(summary["model"].drop_duplicates())}
    )
    summary["_jitter_order"] = summary["jitter_label"].map(jitter_sort_value)
    summary = summary.sort_values(["_model_order", "_jitter_order"])
    return summary.drop(columns=["_model_order", "_jitter_order"])


def run(args: argparse.Namespace) -> None:
    os.chdir(PROJECT_ROOT)
    requested_models = [MODEL_ALIASES[name] for name in args.models]
    set_seeds(args.seed, enable_tf=("DSCNN" in requested_models and args.dscnn_backend == "tensorflow"))

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    jitter_levels = parse_jitter_levels(args.jitter_levels)
    X, y, subjects = load_data(args.csv_dir)
    unique_subjects = sorted(int(s) for s in np.unique(subjects[subjects != -1]))
    if args.subjects:
        requested_subjects = {int(s) for s in args.subjects}
        unique_subjects = [s for s in unique_subjects if s in requested_subjects]

    rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {
        "seed": args.seed,
        "csv_dir": args.csv_dir,
        "jitter_levels": [jitter_label(level) for level in jitter_levels],
        "models": requested_models,
        "param_files": {},
        "simulation_note": (
            "Sampling jitter is applied only to held-out LOSO test windows by "
            "randomly perturbing sampling intervals and resampling back to 100 points."
        ),
    }

    for model_type in requested_models:
        model_display = DISPLAY_NAMES[model_type]
        metadata["param_files"].setdefault(model_type, {})
        for fold_id in unique_subjects:
            print(f"\n=== {model_display} | LOSO subject {fold_id} ===", flush=True)
            train_idx = np.where(subjects != fold_id)[0]
            test_idx = np.where(subjects == fold_id)[0]
            X_train_full, y_train_full = X[train_idx], y[train_idx]
            subjects_train_full = subjects[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            params, params_path = load_best_params(
                model_type,
                fold_id,
                args.optimization_mode,
                recursive=args.recursive_params,
            )
            params = with_default_params(model_type, params)
            params = maybe_cap_epochs(params, model_type, args.max_epochs)
            if model_type == "DSCNN":
                params.setdefault("scale_range", [params.get("scale_min", 0.98), params.get("scale_max", 1.02)])
            metadata["param_files"][model_type][str(fold_id)] = str(params_path) if params_path else None

            X_train, X_val, y_train, y_val, subjects_train, subjects_val = train_test_split(
                X_train_full,
                y_train_full,
                subjects_train_full,
                test_size=VAL_SIZE,
                random_state=args.seed,
                stratify=y_train_full,
            )
            if not args.no_augment:
                X_train, y_train, subjects_train = augment_with_subjects(X_train, y_train, subjects_train, params)

            creator, trained, scaler = train_one_fold(
                model_type,
                params,
                X_train,
                y_train,
                subjects_train,
                X_val,
                y_val,
                subjects_val,
                epochs=args.epochs,
                dscnn_backend=args.dscnn_backend,
            )

            y_pred_clean = predict_labels(model_type, creator, trained, X_test, scaler)
            clean_f1 = float(f1_score(y_test, y_pred_clean, average="macro", zero_division=0))
            clean_acc = float(accuracy_score(y_test, y_pred_clean))
            rows.append(
                {
                    "model": model_type,
                    "model_display": model_display,
                    "fold": fold_id,
                    "repeat": 0,
                    "jitter_fraction": "clean",
                    "jitter_label": "clean",
                    "macro_f1": clean_f1,
                    "accuracy": clean_acc,
                    "delta_macro_f1": 0.0,
                }
            )
            print(f"{model_display} fold {fold_id} clean: Macro-F1={clean_f1:.4f}, Acc={clean_acc:.4f}", flush=True)

            for repeat in range(args.jitter_repeats):
                for jitter_fraction in jitter_levels:
                    if jitter_fraction is None or jitter_fraction == 0:
                        continue
                    repeat_offset = int(round(10000 * jitter_fraction))
                    rng = np.random.default_rng(args.seed + 1000 * fold_id + 100 * repeat + repeat_offset)
                    X_eval = apply_sampling_jitter(
                        X_test,
                        jitter_fraction,
                        rng,
                        clip_min=args.clip_min,
                        clip_max=args.clip_max,
                    )
                    y_pred = predict_labels(model_type, creator, trained, X_eval, scaler)
                    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
                    accuracy = float(accuracy_score(y_test, y_pred))
                    rows.append(
                        {
                            "model": model_type,
                            "model_display": model_display,
                            "fold": fold_id,
                            "repeat": repeat,
                            "jitter_fraction": float(jitter_fraction),
                            "jitter_label": jitter_label(jitter_fraction),
                            "macro_f1": macro_f1,
                            "accuracy": accuracy,
                            "delta_macro_f1": float(macro_f1 - clean_f1),
                        }
                    )
                    print(
                        f"{model_display} fold {fold_id} {jitter_label(jitter_fraction)}: "
                        f"Macro-F1={macro_f1:.4f}, Acc={accuracy:.4f}",
                        flush=True,
                    )

    raw = pd.DataFrame(rows)
    raw_path = output_dir / "sampling_jitter_macro_f1_by_fold.csv"
    raw.to_csv(raw_path, index=False)

    summary = summarize(raw)
    summary.to_csv(output_dir / "sampling_jitter_summary.csv", index=False)
    output_summary = select_output_models(summary, args.output_models)
    output_summary.to_csv(output_dir / "sampling_jitter_summary_main.csv", index=False)
    plot_summary(output_summary, output_dir)
    plot_combined_summary(output_summary, output_dir)
    make_latex_table(output_summary, output_dir)

    metadata["raw_csv"] = str(raw_path)
    metadata["summary_csv"] = str(output_dir / "sampling_jitter_summary.csv")
    (output_dir / "sampling_jitter_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved sampling jitter robustness outputs to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOSO sampling-jitter robustness experiment.")
    parser.add_argument("--csv_dir", default="datasets/gesture_csv")
    parser.add_argument("--output_dir", default="outputs/robustness/sampling_jitter")
    parser.add_argument("--models", nargs="+", default=["DA_LGBM", "ADANN", "LightGBM"], choices=sorted(MODEL_ALIASES))
    parser.add_argument(
        "--output_models",
        nargs="+",
        default=DEFAULT_MAIN_OUTPUT_MODELS,
        choices=sorted(MODEL_ALIASES),
        help="Models included in paper-facing figures and LaTeX table. Raw CSV still keeps every trained model.",
    )
    parser.add_argument("--jitter_levels", nargs="+", default=DEFAULT_JITTER_LEVELS, help='Use "clean", fractions, or percentages.')
    parser.add_argument("--subjects", nargs="*", type=int, help="Optional subset of LOSO subject IDs for smoke tests.")
    parser.add_argument("--optimization_mode", default="full", choices=["full", "arduino"])
    parser.add_argument("--recursive_params", action="store_true", help="Also search nested output folders for best_params files.")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for DSCNN training.")
    parser.add_argument("--dscnn_backend", default="torch", choices=["torch", "tensorflow"])
    parser.add_argument("--max_epochs", type=int, default=None, help="Optional cap for ADANN/DA-LGBM epochs.")
    parser.add_argument("--jitter_repeats", type=int, default=1)
    parser.add_argument("--clip_min", type=float, default=0.0)
    parser.add_argument("--clip_max", type=float, default=1023.0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
