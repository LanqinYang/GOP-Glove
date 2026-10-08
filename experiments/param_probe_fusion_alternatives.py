#!/usr/bin/env python3
"""Fixed-parameter LOSO probe for ADANN-LightGBM fusion alternatives.

This diagnostic script reruns the existing ADANN-LightGBM training pipeline
with one saved hyperparameter set and exports branch-level fusion comparisons.
It is intentionally separate from manuscript-generation scripts.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.da_lgbm_ablation_loso import (  # noqa: E402
    FEATURE_MODES,
    build_feature_mask,
    patch_hybrid_extractor_with_mask,
)
from experiments.da_lgbm_seed_ensemble_loso import extract_test_probabilities  # noqa: E402
from experiments.da_lgbm_seed_stability_loso import (  # noqa: E402
    _ci95,
    _to_basic_type,
    apply_sample_normalization,
    augment_data_with_subjects,
    load_data,
    set_all_seeds,
    split_train_val,
)
from experiments.v7_fusion_calibration_analysis import (  # noqa: E402
    apply_temperature,
    calibration_metrics,
    evaluate_probs,
    gated_probs,
    labels_from_probs,
    logistic_stacking_probs,
    margin_selection_probs,
    max_confidence_probs,
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
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "param_probe_ablation_A_20260313_184330"


def load_saved_params(path: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    params = (
        payload.get("best_hyperparameters_for_deployment")
        or payload.get("best_params")
        or payload.get("hyperparameters")
        or payload
    )
    if not isinstance(params, dict) or not params:
        raise ValueError(f"No usable hyperparameters found in {path}")
    return {k: _to_basic_type(v) for k, v in params.items()}, payload


def parse_folds(folds: str | None, available_subjects: List[int]) -> List[int]:
    if not folds:
        return available_subjects
    requested = [int(token.strip()) for token in folds.split(",") if token.strip()]
    missing = sorted(set(requested) - set(available_subjects))
    if missing:
        raise ValueError(f"Requested folds not present in data: {missing}")
    return requested


def summarize(rows: List[Dict[str, Any]], group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summary_rows: List[Dict[str, Any]] = []
    for key, group in df.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row["n_folds"] = int(len(group))
        for metric in metric_cols:
            values = group[metric].astype(float).to_numpy()
            ci = _ci95(values)
            row[f"mean_{metric}"] = ci["mean"]
            row[f"std_{metric}"] = ci["std"]
            row[f"ci95_{metric}"] = ci["ci95_half_width"]
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def extract_historical_fold_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for item in payload.get("fold_details", []):
        run_name = str(item.get("run_name", ""))
        subject_id = None
        if "_fold_" in run_name:
            try:
                subject_id = int(run_name.rsplit("_fold_", 1)[1])
            except ValueError:
                subject_id = None
        report = item.get("classification_report", {})
        macro_f1 = None
        if isinstance(report, dict):
            macro = report.get("macro avg", {})
            if isinstance(macro, dict) and "f1-score" in macro:
                macro_f1 = float(macro["f1-score"])
        rows.append(
            {
                "subject_id": subject_id,
                "historical_accuracy": float(item["test_accuracy"]) if "test_accuracy" in item else np.nan,
                "historical_macro_f1": macro_f1 if macro_f1 is not None else np.nan,
                "run_name": run_name,
            }
        )
    return rows


def append_strategy_rows(
    rows: List[Dict[str, Any]],
    subject_id: int,
    y_test: np.ndarray,
    class_labels: np.ndarray,
    strategies: Dict[str, tuple[np.ndarray, str]],
) -> None:
    for strategy, (probs, notes) in strategies.items():
        metrics = evaluate_probs(y_test, class_labels, probs)
        pred = labels_from_probs(class_labels, probs)
        rows.append(
            {
                "subject_id": int(subject_id),
                "strategy": strategy,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "static_pred_rate": float(np.mean(pred == 10)),
                "notes": notes,
            }
        )


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(args.seed)

    params, source_payload = load_saved_params(Path(args.param_source))
    params.setdefault("preserve_float", args.sample_normalization in ("zscore", "robust_zscore"))

    X, y, subjects = load_data(args.csv_dir)
    X = apply_sample_normalization(X, args.sample_normalization)
    all_subjects = sorted(int(s) for s in np.unique(subjects[subjects != -1]))
    selected_subjects = parse_folds(args.folds, all_subjects)

    feature_mode = args.feature_mode
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature_mode: {feature_mode}; choose from {FEATURE_MODES}")
    feature_mask = build_feature_mask(feature_mode)

    print(f"Loaded {len(X)} samples from {args.csv_dir}", flush=True)
    print(f"Running subjects: {selected_subjects}", flush=True)
    print(f"Feature mode: {feature_mode}", flush=True)
    print(f"Parameter source: {Path(args.param_source).resolve()}", flush=True)

    fusion_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []

    for test_subj in selected_subjects:
        print(f"\n--- Fixed-param LOSO fold: subject {test_subj} ---", flush=True)
        set_all_seeds(args.seed + int(test_subj) * 1000)

        test_mask = subjects == test_subj
        train_mask = ~test_mask
        X_train = X[train_mask]
        y_train = y[train_mask]
        subj_train = subjects[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

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
        if feature_mode != "full":
            patch_hybrid_extractor_with_mask(creator, feature_mask)
        wrapper = creator.create_model(params, arduino_mode=False)
        wrapper.hybrid_model["lightgbm"].set_params(random_state=args.seed + int(test_subj) * 1000)
        trained_model, val_accuracy = creator.train_model(
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

        adann_test = test_probs["adann"]
        lgb_test = test_probs["lgb"]
        ensemble_test = test_probs["ensemble"]
        gated_test = test_probs["gated"]
        adann_val = val_probs["adann"]
        lgb_val = val_probs["lgb"]

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

        explicit_gated, gate_cases = gated_probs(
            class_labels,
            adann_test,
            lgb_test,
            float(trained_model.get("adann_conf_threshold", params.get("adann_conf_threshold", 0.5))),
            float(trained_model.get("lgb_conf_threshold", params.get("lgb_conf_threshold", 0.5))),
        )

        strategies = {
            "ADANN only": (adann_test, "diagnostic branch"),
            "LightGBM only": (lgb_test, "diagnostic branch"),
            "Parameter-weighted probability mixture": (
                ensemble_test,
                f"weight={float(trained_model.get('ensemble_weight', params.get('ensemble_adann_weight', 0.5))):.3f}",
            ),
            "Confidence-gated rule": (
                gated_test,
                (
                    "thresholds "
                    f"{float(trained_model.get('adann_conf_threshold', 0.5)):.2f}/"
                    f"{float(trained_model.get('lgb_conf_threshold', 0.5)):.2f}"
                ),
            ),
            "Confidence-gated rule (recomputed)": (explicit_gated, "same thresholds, independent recomputation"),
            "Calibrated probability averaging": (
                0.5 * adann_test_cal + 0.5 * lgb_test_cal,
                f"T_adann={t_adann:.3f}; T_lgb={t_lgb:.3f}",
            ),
            "Validation-weighted fusion": (
                validation_weight * adann_test + (1.0 - validation_weight) * lgb_test,
                f"validation weight={validation_weight:.2f}",
            ),
            "Max-confidence selection": (
                max_confidence_probs(class_labels, adann_test, lgb_test),
                "branch with higher top probability",
            ),
            "Margin-based selection": (
                margin_selection_probs(class_labels, adann_test, lgb_test),
                "branch with larger top-two margin",
            ),
            "Logistic stacking (small model)": (logistic_probs, logistic_status),
        }
        append_strategy_rows(fusion_rows, test_subj, y_test, class_labels, strategies)

        for strategy, (probs, _notes) in strategies.items():
            pred = labels_from_probs(class_labels, probs)
            for sample_idx, (true_label, pred_label, conf) in enumerate(zip(y_test, pred, np.max(probs, axis=1))):
                prediction_rows.append(
                    {
                        "subject_id": int(test_subj),
                        "strategy": strategy,
                        "sample_index": int(sample_idx),
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                        "correct": bool(true_label == pred_label),
                        "top_probability": float(conf),
                    }
                )

        for source_name, probs in [
            ("ADANN", adann_test),
            ("LightGBM", lgb_test),
            ("Parameter-weighted probability mixture", ensemble_test),
            ("Calibrated probability averaging", 0.5 * adann_test_cal + 0.5 * lgb_test_cal),
        ]:
            calibration_rows.append(
                {
                    "subject_id": int(test_subj),
                    "probability_source": source_name,
                    **calibration_metrics(y_test, class_labels, probs),
                }
            )

        total_gate = sum(gate_cases.values())
        for case_name, count in gate_cases.items():
            gate_rows.append(
                {
                    "subject_id": int(test_subj),
                    "case": case_name,
                    "count": int(count),
                    "rate": float(count / total_gate) if total_gate else 0.0,
                }
            )

        main_metrics = evaluate_probs(y_test, class_labels, ensemble_test)
        gated_metrics = evaluate_probs(y_test, class_labels, gated_test)
        fold_rows.append(
            {
                "subject_id": int(test_subj),
                "val_accuracy_returned_by_train_model": float(val_accuracy),
                "parameter_mixture_accuracy": main_metrics["accuracy"],
                "parameter_mixture_macro_f1": main_metrics["macro_f1"],
                "confidence_gated_accuracy": gated_metrics["accuracy"],
                "confidence_gated_macro_f1": gated_metrics["macro_f1"],
                "temperature_adann": t_adann,
                "temperature_lgb": t_lgb,
                "validation_weight_adann": validation_weight,
            }
        )
        print(
            (
                f"Subject {test_subj}: mixture Macro-F1={main_metrics['macro_f1']:.4f}, "
                f"Acc={main_metrics['accuracy']:.4f}; gated Macro-F1={gated_metrics['macro_f1']:.4f}, "
                f"Acc={gated_metrics['accuracy']:.4f}"
            ),
            flush=True,
        )

    fusion_df = pd.DataFrame(fusion_rows)
    fusion_df.to_csv(output_dir / "fusion_alternatives_by_fold.csv", index=False)
    fusion_summary = summarize(
        fusion_rows,
        ["strategy"],
        ["accuracy", "macro_f1", "static_pred_rate"],
    ).sort_values(["mean_macro_f1", "mean_accuracy"], ascending=False)
    fusion_summary.to_csv(output_dir / "fusion_alternatives_summary.csv", index=False)

    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(output_dir / "calibration_by_fold.csv", index=False)
    summarize(
        calibration_rows,
        ["probability_source"],
        ["accuracy", "mean_confidence", "ece_10bin", "mce_10bin", "brier_multiclass", "nll"],
    ).sort_values("mean_ece_10bin").to_csv(output_dir / "calibration_summary.csv", index=False)

    pd.DataFrame(gate_rows).to_csv(output_dir / "gate_case_by_fold.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "fusion_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_probe_summary.csv", index=False)

    historical_rows = extract_historical_fold_rows(source_payload)
    if historical_rows:
        pd.DataFrame(historical_rows).to_csv(output_dir / "historical_source_fold_summary.csv", index=False)

    metadata = {
        "param_source": str(Path(args.param_source).resolve()),
        "output_dir": str(output_dir.resolve()),
        "csv_dir": str(Path(args.csv_dir).resolve()),
        "seed": args.seed,
        "feature_mode": feature_mode,
        "sample_normalization": args.sample_normalization,
        "final_val_ratio": args.final_val_ratio,
        "group_aware_final_val": args.group_aware_final_val,
        "subjects": selected_subjects,
        "source_average_accuracy": source_payload.get("average_accuracy"),
        "source_average_macro_f1": source_payload.get("average_f1_macro"),
        "notes": [
            "This is a fresh fixed-parameter diagnostic rerun using the saved hyperparameter set.",
            "Historical source metrics may differ from the rerun because saved training artifacts do not contain branch validation probabilities.",
            "Fusion alternatives use only the inner validation split for calibration, weighting, or stacking.",
        ],
        "hyperparameters": params,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved outputs to {output_dir}", flush=True)
    print("\nFusion summary:", flush=True)
    print(
        fusion_summary[
            [
                "strategy",
                "n_folds",
                "mean_macro_f1",
                "std_macro_f1",
                "mean_accuracy",
                "std_accuracy",
                "mean_static_pred_rate",
            ]
        ].to_string(index=False),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-parameter ADANN-LightGBM fusion probe")
    parser.add_argument("--csv_dir", default="datasets/gesture_csv")
    parser.add_argument("--param_source", default=str(DEFAULT_PARAM_SOURCE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_mode", choices=FEATURE_MODES, default="A")
    parser.add_argument("--folds", default=None, help="Optional comma-separated LOSO subjects, e.g. 1,2")
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
