"""Run the manuscript AWGN, jitter and drift experiments with fixed parameters."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODULES = {'awgn':'awgn_robustness', 'jitter':'sampling_jitter_robustness',
           'drift':'sensor_drift_robustness'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=MODULES)
    args, forwarded = parser.parse_known_args()
    from scripts.run_manuscript import parameters

    def fixed_params(model_type, fold_id, optimization_mode, recursive=False):
        model = 'DA_LGBM' if model_type == 'ADANN_LightGBM' else model_type
        params = parameters(model, fold_id, 200)
        return params, ROOT / ('configs/da_lgbm_loso.json' if model == 'DA_LGBM' else 'configs/baseline_configs.json')

    module = importlib.import_module('experiments.' + MODULES[args.experiment])
    module.load_best_params = fixed_params
    sys.argv = [MODULES[args.experiment], '--csv_dir', str(ROOT / 'datasets/gesture_csv'),
                '--output_dir', str(ROOT / 'outputs/manuscript/robustness' / args.experiment),
                '--seed', '42', '--models', 'DA_LGBM', 'ADANN', 'LightGBM', *forwarded]
    module.main()


if __name__ == '__main__':
    main()
