# BSL Gesture Recognition System

Data and code for isolated 11-class British Sign Language gesture recognition using a five-channel graphite-on-paper glove and an Arduino Nano 33 BLE Sense Rev2.

## Reproducibility materials

| Material | Location |
|---|---|
| Pseudonymised dataset | `datasets/gesture_csv/`; column definitions in `datasets/README.md`; recording identities in `reproducibility/dataset_manifest.csv` |
| Feature extraction scripts | `src/training/manuscript_features.py`, `src/training/feature_utils.py`, `scripts/export_reproducibility.py` |
| IID/LOSO split files | `reproducibility/generated/iid_seed42.csv`, `reproducibility/generated/loso_seed42.csv` |
| Baseline configurations | `configs/baseline_configs.json` and its referenced fold parameter JSONs; DA-LGBM parameters in `configs/da_lgbm_loso.json` |
| Seed-stability scripts | `experiments/da_lgbm_seed_stability_loso.py` |
| Robustness-analysis scripts | `experiments/awgn_robustness.py`, `experiments/sampling_jitter_robustness.py`, `experiments/sensor_drift_robustness.py` |
| Embedded inference code | `arduino/tinyml_inference/ADANN_LightGBM_inference/BSL_Gesture_Demo/`, including both model headers and the confidence gate |

Shared settings are in `configs/manuscript_protocol.json`. The training modules under `src/training/` and the command runners under `scripts/` support these workflows.

## Setup and preparation

Run from the repository root with Python 3.12:

```bash
git clone https://github.com/LanqinYang/GOP-Glove.git
cd GOP-Glove
python -m pip install -r requirements.txt
python scripts/run_manuscript.py prepare
python scripts/verify_reproducibility.py
```

For feature extraction and split checks alone:

```bash
python -m pip install -r reproducibility/requirements-features.txt
python scripts/export_reproducibility.py
python scripts/verify_reproducibility.py
```

Preparation generates the unscaled 660-by-190 feature array and filename-based partitions. Fitted scalers and augmentation use training windows only.

## Training, seed stability and robustness

```bash
# DA-LGBM: six LOSO folds or pooled IID evaluation
python scripts/run_manuscript.py train --model DA_LGBM --evaluation loso
python scripts/run_manuscript.py train --model DA_LGBM --evaluation iid

# Baseline example
python scripts/run_manuscript.py train --model LightGBM --evaluation loso

# Five seeds with fixed LOSO partitions: 42, 123, 2025, 2026, 3047
python scripts/run_manuscript.py seeds

# Controlled held-out-window perturbations
python scripts/run_manuscript.py robustness awgn
python scripts/run_manuscript.py robustness jitter
python scripts/run_manuscript.py robustness drift
```

Training supports `--folds`, `--seed`, `--epochs` and `--output_dir`; use `train --help` for the model choices. Runs save their computed metrics, predictions and parameters locally in the selected output directory. Robustness runs perturb test windows while keeping the fitted model and preprocessing fixed.

## Embedded inference

```bash
arduino-cli core install arduino:mbed_nano
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble arduino/tinyml_inference/ADANN_LightGBM_inference/BSL_Gesture_Demo
```

Upload the sketch to the Nano 33 BLE Sense Rev2 using the Arduino IDE or `arduino-cli upload`. Send `s` over serial to start an independent two-second acquisition and `R` for timing summaries. Full-cycle timing includes acquisition; post-acquisition timing covers feature extraction, inference, gating and output.

The project code is distributed under the [MIT License](LICENSE).
