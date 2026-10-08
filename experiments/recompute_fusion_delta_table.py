#!/usr/bin/env python3
"""Recompute reviewer-requested fusion deltas from branch probabilities.

This script reruns the fixed-parameter ADANN-LightGBM LOSO probe and computes
only paired deltas relative to the deployed confidence-gated rule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.da_lgbm_seed_ensemble_loso import extract_test_probabilities  # noqa: E402
from experiments.da_lgbm_seed_stability_loso import (  # noqa: E402
    _ci95,
    apply_sample_normalization,
    augment_data_with_subjects,
    load_data,
    macro_f1_from_cm,
    set_all_seeds,
    split_train_val,
)
from experiments.param_probe_fusion_alternatives import load_saved_params  # noqa: E402
from experiments.v7_fusion_calibration_analysis import (  # noqa: E402
    apply_temperature,
    calibration_metrics,
    logistic_stacking_probs,
    tune_temperature,
    tune_weight,
)
from src.training.train_adann_lightgbm import AdannLightgbmModelCreator  # noqa: E402


DEFAULT_PARAM_SOURCE = (
    PROJECT_ROOT
    / "outputs"
    / "ADANN_LightGBM"
    / "loso"
    / "full"
    / "ablation_A"
    / "loso_summary_ADANN_LightGBM_20260313_184330.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v7_fusion_delta_audit"


def top_two_margin(probs: np.ndarray) -> np.ndarray:
    part = np.partition(probs, -2, axis=1)
    return part[:, -1] - part[:, -2]


def labels_from_probs(class_labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    return class_labels[np.argmax(probs, axis=1)]


def onehot_from_labels(class_labels: np.ndarray, labels: np.ndarray) -> np.ndarray:
    label_to_idx = {int(label): i for i, label in enumerate(class_labels)}
    out = np.zeros((len(labels), len(class_labels)), dtype=float)
    for row_idx, label in enumerate(labels):
        out[row_idx, label_to_idx[int(label)]] = 1.0
    return out


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_labels: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    return {
        "macro_f1": float(macro_f1_from_cm(cm)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def evaluate_probs(y_true: np.ndarray, class_labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    return evaluate_predictions(y_true, labels_from_probs(class_labels, probs), class_labels)


def confidence_gated_labels(
    class_labels: np.ndarray,
    adann: np.ndarray,
    lgb: np.ndarray,
    adann_threshold: float,
    lgb_threshold: float,
    static_label: int = 10,
) -> tuple[np.ndarray, Dict[str, int]]:
    adann_label = labels_from_probs(class_labels, adann)
    lgb_label = labels_from_probs(class_labels, lgb)
    adann_conf = np.max(adann, axis=1)
    lgb_conf = np.max(lgb, axis=1)
    adann_margin = top_two_margin(adann)
    lgb_margin = top_two_margin(lgb)
    labels: List[int] = []
    cases = {
        "agree_confident": 0,
        "adann_only_confident": 0,
        "lgb_only_confident": 0,
        "confident_disagreement_margin": 0,
        "fallback_static": 0,
    }
    for i in range(len(adann)):
        a_ok = adann_conf[i] >= adann_threshold
        l_ok = lgb_conf[i] >= lgb_threshold
        if a_ok and l_ok and adann_label[i] == lgb_label[i]:
            labels.append(int(lgb_label[i]))
            cases["agree_confident"] += 1
        elif a_ok and not l_ok:
            labels.append(int(adann_label[i]))
            cases["adann_only_confident"] += 1
        elif l_ok and not a_ok:
            labels.append(int(lgb_label[i]))
            cases["lgb_only_confident"] += 1
        elif a_ok and l_ok:
            labels.append(int(adann_label[i] if adann_margin[i] > lgb_margin[i] else lgb_label[i]))
            cases["confident_disagreement_margin"] += 1
        else:
            labels.append(int(static_label))
            cases["fallback_static"] += 1
    return np.asarray(labels, dtype=int), cases


def confidence_gate_cases(
    class_labels: np.ndarray,
    adann: np.ndarray,
    lgb: np.ndarray,
    adann_threshold: float,
    lgb_threshold: float,
) -> np.ndarray:
    adann_label = labels_from_probs(class_labels, adann)
    lgb_label = labels_from_probs(class_labels, lgb)
    adann_conf = np.max(adann, axis=1)
    lgb_conf = np.max(lgb, axis=1)
    cases: List[str] = []
    for i in range(len(adann)):
        a_ok = adann_conf[i] >= adann_threshold
        l_ok = lgb_conf[i] >= lgb_threshold
        if a_ok and l_ok and adann_label[i] == lgb_label[i]:
            cases.append("agree_confident")
        elif a_ok and not l_ok:
            cases.append("adann_only_confident")
        elif l_ok and not a_ok:
            cases.append("lgb_only_confident")
        elif a_ok and l_ok:
            cases.append("confident_disagreement_margin")
        else:
            cases.append("fallback_static")
    return np.asarray(cases, dtype=object)


def margin_only_labels(class_labels: np.ndarray, adann: np.ndarray, lgb: np.ndarray) -> np.ndarray:
    adann_label = labels_from_probs(class_labels, adann)
    lgb_label = labels_from_probs(class_labels, lgb)
    adann_margin = top_two_margin(adann)
    lgb_margin = top_two_margin(lgb)
    return np.where(adann_margin > lgb_margin, adann_label, lgb_label).astype(int)


def max_confidence_labels(class_labels: np.ndarray, adann: np.ndarray, lgb: np.ndarray) -> np.ndarray:
    adann_label = labels_from_probs(class_labels, adann)
    lgb_label = labels_from_probs(class_labels, lgb)
    adann_conf = np.max(adann, axis=1)
    lgb_conf = np.max(lgb, axis=1)
    return np.where(adann_conf >= lgb_conf, adann_label, lgb_label).astype(int)


def summarize_delta(rows: List[Dict[str, Any]], disagreement_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    disagreement_df = pd.DataFrame(disagreement_rows)
    out_rows = []
    for strategy, group in df.groupby("fusion_strategy", sort=False):
        row: Dict[str, Any] = {"fusion_strategy": strategy, "n_folds": int(len(group))}
        for metric in ["delta_macro_f1", "delta_accuracy"]:
            ci = _ci95(group[metric].astype(float).to_numpy())
            row[f"mean_{metric}"] = ci["mean"]
            row[f"std_{metric}"] = ci["std"]
            row[f"ci95_{metric}"] = ci["ci95_half_width"]
        if not disagreement_df.empty and strategy in set(disagreement_df["fusion_strategy"]):
            sg = disagreement_df[disagreement_df["fusion_strategy"] == strategy]
            row["total_disagree_with_gate"] = int(sg["n_disagree_with_gate"].sum())
            row["mean_disagree_rate_with_gate"] = float(sg["rate_disagree_with_gate"].mean())
        else:
            row["total_disagree_with_gate"] = 0
            row["mean_disagree_rate_with_gate"] = 0.0
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def summarize_calibration(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out_rows: List[Dict[str, Any]] = []
    metric_cols = [
        "accuracy",
        "mean_confidence",
        "ece_10bin",
        "mce_10bin",
        "brier_multiclass",
        "nll",
    ]
    for (branch, probability_source), group in df.groupby(["branch", "probability_source"], sort=False):
        row: Dict[str, Any] = {
            "branch": branch,
            "probability_source": probability_source,
            "n_folds": int(len(group)),
            "n_samples": int(group["n"].sum()),
        }
        for metric in metric_cols:
            ci = _ci95(group[metric].astype(float).to_numpy())
            row[f"mean_{metric}"] = ci["mean"]
            row[f"std_{metric}"] = ci["std"]
            row[f"ci95_{metric}"] = ci["ci95_half_width"]
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params, source_payload = load_saved_params(Path(args.param_source))
    params.setdefault("preserve_float", args.sample_normalization in ("zscore", "robust_zscore"))

    set_all_seeds(args.seed)
    X, y, subjects = load_data(args.csv_dir)
    X = apply_sample_normalization(X, args.sample_normalization)
    unique_subjects = sorted(int(s) for s in np.unique(subjects[subjects != -1]))

    by_fold_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    disagreement_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    sample_audit_rows: List[Dict[str, Any]] = []
    full_calibration_rows: List[Dict[str, Any]] = []

    for test_subj in unique_subjects:
        print(f"\n--- recompute fusion deltas fold {test_subj} ---", flush=True)
        set_all_seeds(args.seed + int(test_subj) * 1000)
        test_mask = subjects == test_subj
        train_mask = ~test_mask
        X_train, y_train, subj_train = X[train_mask], y[train_mask], subjects[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        X_train_fold, y_train_fold, subj_train_fold, X_val_fold, y_val_fold, subj_val_fold = split_train_val(
            X_train,
            y_train,
            subj_train,
            split_seed=args.seed + int(test_subj) * 7777,
            val_ratio=args.final_val_ratio,
            group_aware=args.group_aware_final_val,
        )
        X_aug, y_aug, subj_aug = augment_data_with_subjects(
            X_train_fold,
            y_train_fold,
            subj_train_fold,
            params,
            seed=args.seed + int(test_subj) * 10000,
        )

        creator = AdannLightgbmModelCreator()
        wrapper = creator.create_model(params, arduino_mode=False)
        wrapper.hybrid_model["lightgbm"].set_params(random_state=args.seed + int(test_subj) * 1000)
        trained_model, _ = creator.train_model(
            wrapper.hybrid_model,
            X_aug,
            y_aug,
            subj_aug,
            X_val_fold,
            y_val_fold,
            subj_val_fold,
            params,
            return_history=False,
        )

        class_labels = trained_model["gesture_encoder"].classes_
        test_probs = extract_test_probabilities(creator, trained_model, X_test)
        val_probs = extract_test_probabilities(creator, trained_model, X_val_fold)
        adann_test, lgb_test = test_probs["adann"], test_probs["lgb"]
        adann_val, lgb_val = val_probs["adann"], val_probs["lgb"]

        adann_threshold = float(trained_model.get("adann_conf_threshold", params.get("adann_conf_threshold", 0.5)))
        lgb_threshold = float(trained_model.get("lgb_conf_threshold", params.get("lgb_conf_threshold", 0.5)))
        gate_labels, gate_cases = confidence_gated_labels(
            class_labels,
            adann_test,
            lgb_test,
            adann_threshold,
            lgb_threshold,
        )
        gate_case_labels = confidence_gate_cases(
            class_labels,
            adann_test,
            lgb_test,
            adann_threshold,
            lgb_threshold,
        )
        gate_metrics = evaluate_predictions(y_test, gate_labels, class_labels)

        t_adann = tune_temperature(y_val_fold, class_labels, adann_val)
        t_lgb = tune_temperature(y_val_fold, class_labels, lgb_val)
        adann_test_cal = apply_temperature(adann_test, t_adann)
        lgb_test_cal = apply_temperature(lgb_test, t_lgb)
        validation_weight = tune_weight(y_val_fold, class_labels, adann_val, lgb_val)
        logistic_probs, logistic_status = logistic_stacking_probs(
            y_val_fold,
            class_labels,
            adann_val,
            lgb_val,
            adann_test,
            lgb_test,
        )
        parameter_weight = float(trained_model.get("ensemble_weight", params.get("ensemble_adann_weight", 0.5)))

        branch_probability_sets = [
            ("ADANN", "raw", adann_test),
            ("LightGBM", "raw", lgb_test),
            ("ADANN", f"temperature_scaled_T={t_adann:.3f}", adann_test_cal),
            ("LightGBM", f"temperature_scaled_T={t_lgb:.3f}", lgb_test_cal),
        ]
        for branch_name, probability_source, probs_for_metrics in branch_probability_sets:
            metrics = calibration_metrics(y_test, class_labels, probs_for_metrics)
            full_calibration_rows.append(
                {
                    "subject_id": int(test_subj),
                    "branch": branch_name,
                    "probability_source": probability_source,
                    "n": int(len(y_test)),
                    **metrics,
                }
            )

        strategies = {
            "Confidence-gated rule": (
                onehot_from_labels(class_labels, gate_labels),
                "reference",
            ),
            "Calibrated probability averaging": (
                0.5 * adann_test_cal + 0.5 * lgb_test_cal,
                f"T_adann={t_adann:.3f}; T_lgb={t_lgb:.3f}",
            ),
            "Validation-weighted fusion": (
                validation_weight * adann_test + (1.0 - validation_weight) * lgb_test,
                f"validation_weight_adann={validation_weight:.2f}",
            ),
            "Maximum-confidence selection": (
                onehot_from_labels(class_labels, max_confidence_labels(class_labels, adann_test, lgb_test)),
                "independent recomputation",
            ),
            "Margin-only selection": (
                onehot_from_labels(class_labels, margin_only_labels(class_labels, adann_test, lgb_test)),
                "independent recomputation",
            ),
            "Logistic stacking": (
                logistic_probs,
                logistic_status,
            ),
            "Parameter-weighted probability mixture": (
                parameter_weight * adann_test + (1.0 - parameter_weight) * lgb_test,
                f"parameter_weight_adann={parameter_weight:.3f}",
            ),
        }

        adann_labels = labels_from_probs(class_labels, adann_test)
        lgb_labels = labels_from_probs(class_labels, lgb_test)
        adann_conf = np.max(adann_test, axis=1)
        lgb_conf = np.max(lgb_test, axis=1)
        adann_margin = top_two_margin(adann_test)
        lgb_margin = top_two_margin(lgb_test)
        margin_labels = margin_only_labels(class_labels, adann_test, lgb_test)
        max_conf_labels = max_confidence_labels(class_labels, adann_test, lgb_test)

        for sample_idx in range(len(y_test)):
            sample_audit_rows.append(
                {
                    "subject_id": int(test_subj),
                    "sample_index": int(sample_idx),
                    "true_label": int(y_test[sample_idx]),
                    "gate_case": str(gate_case_labels[sample_idx]),
                    "adann_label": int(adann_labels[sample_idx]),
                    "lgb_label": int(lgb_labels[sample_idx]),
                    "adann_conf": float(adann_conf[sample_idx]),
                    "lgb_conf": float(lgb_conf[sample_idx]),
                    "adann_margin": float(adann_margin[sample_idx]),
                    "lgb_margin": float(lgb_margin[sample_idx]),
                    "gate_label": int(gate_labels[sample_idx]),
                    "margin_only_label": int(margin_labels[sample_idx]),
                    "max_confidence_label": int(max_conf_labels[sample_idx]),
                    "margin_only_differs_from_gate": bool(margin_labels[sample_idx] != gate_labels[sample_idx]),
                    "max_confidence_differs_from_gate": bool(max_conf_labels[sample_idx] != gate_labels[sample_idx]),
                }
            )

        for strategy, (probs, note) in strategies.items():
            y_pred = labels_from_probs(class_labels, probs)
            metrics = evaluate_probs(y_test, class_labels, probs)
            diff_mask = y_pred != gate_labels
            disagreement_rows.append(
                {
                    "subject_id": int(test_subj),
                    "fusion_strategy": strategy,
                    "n_disagree_with_gate": int(np.sum(diff_mask)),
                    "rate_disagree_with_gate": float(np.mean(diff_mask)),
                }
            )
            by_fold_rows.append(
                {
                    "subject_id": int(test_subj),
                    "fusion_strategy": strategy,
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "delta_macro_f1": metrics["macro_f1"] - gate_metrics["macro_f1"],
                    "delta_accuracy": metrics["accuracy"] - gate_metrics["accuracy"],
                    "note": note,
                }
            )
            for sample_idx, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
                prediction_rows.append(
                    {
                        "subject_id": int(test_subj),
                        "sample_index": int(sample_idx),
                        "fusion_strategy": strategy,
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                        "gate_pred_label": int(gate_labels[sample_idx]),
                        "different_from_gate": bool(pred_label != gate_labels[sample_idx]),
                    }
                )
            if strategy != "Confidence-gated rule":
                for sample_idx in np.where(diff_mask)[0]:
                    prediction_rows.append(
                        {
                            "subject_id": int(test_subj),
                            "sample_index": int(sample_idx),
                            "fusion_strategy": f"{strategy} audit disagreement",
                            "true_label": int(y_test[sample_idx]),
                            "pred_label": int(y_pred[sample_idx]),
                            "gate_pred_label": int(gate_labels[sample_idx]),
                            "different_from_gate": True,
                            "adann_label": int(adann_labels[sample_idx]),
                            "lgb_label": int(lgb_labels[sample_idx]),
                            "adann_conf": float(adann_conf[sample_idx]),
                            "lgb_conf": float(lgb_conf[sample_idx]),
                            "adann_margin": float(adann_margin[sample_idx]),
                            "lgb_margin": float(lgb_margin[sample_idx]),
                        }
                    )
            print(
                f"{strategy}: dF1={metrics['macro_f1'] - gate_metrics['macro_f1']:+.6f}, "
                f"dAcc={metrics['accuracy'] - gate_metrics['accuracy']:+.6f}",
                flush=True,
            )

        for case_name, count in gate_cases.items():
            case_rows.append({"subject_id": int(test_subj), "gate_case": case_name, "count": int(count)})

    by_fold_df = pd.DataFrame(by_fold_rows)
    summary_df = summarize_delta(by_fold_rows, disagreement_rows)
    order = [
        "Confidence-gated rule",
        "Calibrated probability averaging",
        "Validation-weighted fusion",
        "Maximum-confidence selection",
        "Margin-only selection",
        "Logistic stacking",
        "Parameter-weighted probability mixture",
    ]
    by_fold_df["fusion_strategy"] = pd.Categorical(by_fold_df["fusion_strategy"], categories=order, ordered=True)
    summary_df["fusion_strategy"] = pd.Categorical(summary_df["fusion_strategy"], categories=order, ordered=True)
    by_fold_df = by_fold_df.sort_values(["fusion_strategy", "subject_id"])
    summary_df = summary_df.sort_values("fusion_strategy")

    by_fold_df.to_csv(output_dir / "fusion_delta_by_fold.csv", index=False)
    summary_df.to_csv(output_dir / "fusion_delta_summary.csv", index=False)
    full_calibration_df = pd.DataFrame(full_calibration_rows)
    full_calibration_summary_df = summarize_calibration(full_calibration_rows)
    full_calibration_df.to_csv(output_dir / "branch_full_calibration_by_fold.csv", index=False)
    full_calibration_summary_df.to_csv(output_dir / "branch_full_calibration_summary.csv", index=False)
    pd.DataFrame(disagreement_rows).to_csv(output_dir / "selection_disagreement_with_gate.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "fusion_delta_prediction_audit.csv", index=False)
    pd.DataFrame(sample_audit_rows).to_csv(output_dir / "gate_margin_sample_audit.csv", index=False)
    pd.DataFrame(case_rows).to_csv(output_dir / "gate_case_counts.csv", index=False)
    metadata = {
        "param_source": str(Path(args.param_source).resolve()),
        "source_average_macro_f1": source_payload.get("average_f1_macro"),
        "source_average_accuracy": source_payload.get("average_accuracy"),
        "csv_dir": str(Path(args.csv_dir).resolve()),
        "seed": args.seed,
        "sample_normalization": args.sample_normalization,
        "final_val_ratio": args.final_val_ratio,
        "group_aware_final_val": args.group_aware_final_val,
        "metric_definition": "paired delta relative to the confidence-gated rule within each held-out subject fold",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved delta outputs to {output_dir}", flush=True)
    print(summary_df.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute fusion delta table from branch probabilities")
    parser.add_argument("--csv_dir", default="datasets/gesture_csv")
    parser.add_argument("--param_source", default=str(DEFAULT_PARAM_SOURCE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final_val_ratio", type=float, default=0.2)
    parser.add_argument("--group_aware_final_val", action="store_true")
    parser.add_argument(
        "--sample_normalization",
        choices=["none", "center", "baseline", "zscore", "robust_zscore"],
        default="none",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "42")
    run(parse_args())
