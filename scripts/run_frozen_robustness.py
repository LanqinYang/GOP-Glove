"""Run a diagnostic with recorded parameters rather than selecting the newest file.

The original environment and every historical CLI flag have not been recovered.
Inspect the saved metadata and provide additional flags explicitly when needed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPERIMENTS = {
    "awgn": ("awgn_robustness", "awgn_core", "awgn_metadata.json"),
    "jitter": ("sampling_jitter_robustness", "sampling_jitter_core_r5", "sampling_jitter_metadata.json"),
    "drift": ("sensor_drift_robustness", "sensor_drift_core_r5", "sensor_drift_metadata.json"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    args, forwarded = parser.parse_known_args()
    module_name, folder, metadata_file = EXPERIMENTS[args.experiment]
    meta = json.loads((ROOT / "reproducibility/results/robustness" / folder / metadata_file).read_text())

    def frozen_params(model_type, fold_id, optimization_mode, recursive=False):
        path = ROOT / meta["param_files"][model_type][str(fold_id)]
        payload = json.loads(path.read_text())
        return payload.get("best_params", payload), path

    module = importlib.import_module("experiments." + module_name)
    module.load_best_params = frozen_params
    sys.argv = [module_name, "--csv_dir", str(ROOT / "datasets/gesture_csv"),
                "--output_dir", str(ROOT / "outputs/reproduced" / args.experiment),
                "--seed", str(meta["seed"]), "--models", *meta["models"], *forwarded]
    module.main()


if __name__ == "__main__":
    main()
