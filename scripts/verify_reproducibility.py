"""Check dataset identity, leakage in exported splits and raw diagnostic summaries."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    manifest = rows(ROOT / "reproducibility/dataset_manifest.csv")
    assert len(manifest) == 660
    for item in manifest:
        path = ROOT / item["relative_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
    counts = Counter((item["participant_id"], item["class_id"]) for item in manifest)
    assert len(counts) == 66 and set(counts.values()) == {10}
    generated = ROOT / "reproducibility/generated"
    iid = rows(generated / "iid_seed42.csv")
    assert len(iid) == 660 and len({r["filename"] for r in iid}) == 660
    assert Counter(r["partition"] for r in iid) == {"train": 422, "validation": 106, "test": 132}
    loso = rows(generated / "loso_seed42.csv")
    for fold in range(1, 7):
        current = [r for r in loso if int(r["fold"]) == fold]
        assert len(current) == 660 and len({r["filename"] for r in current}) == 660
        assert Counter(r["partition"] for r in current) == {"train": 440, "validation": 110, "test": 110}
        assert all((int(r["participant_id"]) == fold) == (r["partition"] == "test") for r in current)
    features = np.load(generated / "legacy_unscaled_features.npy")
    assert features.shape == (660, 190) and np.isfinite(features).all()
    archived = []
    for path in sorted((ROOT / "reproducibility/results/archived_confusion_matrices").glob("DA_LGBM_confusion_matrix_fold*.csv")):
        matrix = np.loadtxt(path, delimiter=",")
        assert matrix.shape == (11, 11) and (matrix >= 0).all()
        assert np.allclose(matrix.sum(axis=1), 10) and np.isclose(matrix.sum(), 110)
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        archived.append((np.mean(2 * matrix.diagonal() / denominator), matrix.trace() / matrix.sum()))
    assert len(archived) == 6
    archived_means = np.mean(archived, axis=0)
    assert np.isclose(archived_means[0], 0.8366122159953537, atol=1e-12)
    assert np.isclose(archived_means[1], 0.853030303030303, atol=1e-12)
    seed_dir = ROOT / "reproducibility/results/DA_LGBM_seed_stability_gated_seeds_42_123_2025_2026_3047"
    summary = json.loads((seed_dir / "seed_stability_summary.json").read_text())
    by_seed = rows(seed_dir / "seed_stability_by_seed.csv")
    for metric in ("macro_f1", "accuracy"):
        values = np.array([float(r["mean_" + metric]) for r in by_seed])
        assert np.isclose(values.mean(), summary["overall_mean_" + metric], atol=1e-12)
        assert np.isclose(values.std(ddof=1), summary["overall_std_" + metric], atol=1e-12)
    audit_dir = ROOT / "reproducibility/results/v7_fusion_delta_final_seed810"
    gate = rows(audit_dir / "gate_margin_sample_audit.csv")
    assert len(gate) == 660
    fallback = sum(r["gate_case"] == "fallback_static" for r in gate)
    static = sum(int(r["gate_label"]) == 10 for r in gate)
    false_static = sum(int(r["gate_label"]) == 10 and int(r["true_label"]) != 10 for r in gate)
    assert (fallback, static, false_static) == (1, 64, 6)
    # Check the actual legacy Python gate against all stored branch summaries.
    for r in gate:
        a, b = float(r["adann_conf"]), float(r["lgb_conf"])
        al, bl = int(r["adann_label"]), int(r["lgb_label"])
        if a >= 0.5 and b >= 0.5 and al == bl:
            predicted = bl
        elif a >= 0.5 and b < 0.5:
            predicted = al
        elif b >= 0.5 and a < 0.5:
            predicted = bl
        elif a >= 0.5 and b >= 0.5:
            predicted = al if float(r["adann_margin"]) > float(r["lgb_margin"]) else bl
        else:
            predicted = 10
        assert predicted == int(r["gate_label"])
    raw_robustness_groups = 0
    for name, prefix, level in (("awgn_core", "awgn", "snr_label"), ("sampling_jitter_core_r5", "sampling_jitter", "jitter_label"), ("sensor_drift_core_r5", "sensor_drift", "drift_label")):
        folder = ROOT / "reproducibility/results/robustness" / name
        raw = rows(folder / (prefix + "_macro_f1_by_fold.csv"))
        saved = rows(folder / (prefix + "_summary.csv"))
        groups = defaultdict(list)
        for r in raw:
            groups[(r["model"], r[level])].append(float(r["macro_f1"]))
        for r in saved:
            values = groups[(r["model"], r[level])]
            assert np.isclose(np.mean(values), float(r["macro_f1_mean"]), atol=1e-12)
            raw_robustness_groups += 1
        meta = json.loads((folder / (prefix + "_metadata.json")).read_text())
        for paths in meta.get("param_files", {}).values():
            assert all((ROOT / path).is_file() for path in paths.values())
    python_files = [p for directory in ("src", "experiments", "scripts", "reproducibility") for p in (ROOT / directory).rglob("*.py")]
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
    result = {
        "dataset_files_verified": 660, "subject_class_cells": 66,
        "iid_partition_counts": {"train": 422, "validation": 106, "test": 132},
        "loso_folds_verified": 6, "test_subject_leakage": False,
        "feature_shape": [660, 190], "feature_values_finite": True,
        "archived_matrix_arithmetic": {"macro_f1": float(archived_means[0]), "accuracy": float(archived_means[1])},
        "seed_summary_recomputed": True, "static_audit": [fallback, static, false_static],
        "gate_decisions_checked": 660, "raw_robustness_groups_checked": raw_robustness_groups,
        "python_files_syntax_checked": len(python_files),
        "historical_primary_run_reproduced": False, "physical_firmware_retested": False,
        "status": "review_materials_verified_with_documented_reproduction_gaps",
    }
    (ROOT / "reproducibility/verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
