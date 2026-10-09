# BSL Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A controlled six-participant feasibility study of an isolated 11-class British Sign Language (BSL) vocabulary using a five-channel graphite-on-paper glove and an Arduino Nano 33 BLE Sense Rev2.

## Manuscript

*Domain-Adversarial Light Gradient Boosting Machine for On-Device Recognition of Isolated Gestures from a Constrained BSL Vocabulary Using Graphite-on-Paper Sensors*

The manuscript reports LOSO Macro-F1 of 0.8366 and accuracy of 0.8530. Mean branch improvements were not statistically significant after Holm correction. The reported full recognition cycle is approximately 2.5 s, comprising two-second acquisition and post-acquisition processing below 0.5 s, under short-term indoor, USB-connected profiling.

## 🚀 Method

- Five GoP channels acquired at 50 Hz in independent two-second windows.
- Linear resampling to 100 points per channel and a shared 190-D descriptor: 18 time-domain, 12 frequency-domain and eight Ricker-wavelet features per channel.
- Parallel ADANN and LightGBM branches; fold-dependent scalers fit on training data only.
- Fixed 0.5 confidence thresholds. Confident agreement returns the agreed class; one confident branch supplies its label; confident disagreement uses the larger top-two probability margin; two low-confidence branches return Static.
- Training-only jitter, amplitude scaling and time warping, with augmentation probability 0.3.
- Six LOSO folds and five stability seeds: 42, 123, 2025, 2026 and 3047.
- FP32 ADANN forward pass and plain-C LightGBM inference on the Arduino.

## 📁 Reproducibility materials

| Material | Location |
|---|---|
| Pseudonymised dataset | `datasets/gesture_csv/`; file identities in `reproducibility/dataset_manifest.csv` |
| Feature extraction scripts | `src/training/manuscript_features.py`, `scripts/export_reproducibility.py` |
| IID/LOSO split files | `reproducibility/generated/iid_seed42.csv`, `reproducibility/generated/loso_seed42.csv` |
| Baseline configurations | `configs/baseline_configs.json`, `configs/da_lgbm_loso.json`, recorded fold parameter files |
| Seed-stability scripts | `experiments/da_lgbm_seed_stability_loso.py`, `scripts/run_manuscript.py seeds` |
| Robustness-analysis scripts | `experiments/awgn_robustness.py`, `experiments/sampling_jitter_robustness.py`, `experiments/sensor_drift_robustness.py` |
| Embedded inference code | `arduino/tinyml_inference/ADANN_LightGBM_inference/BSL_Gesture_Demo/` |

The shared settings are in `configs/manuscript_protocol.json`. Training commands write predictions, confusion matrices, parameters and computed metrics into their output directories.

## 🛠️ Quick start

Use Python 3.12 from the repository root:

```bash
git clone https://github.com/LanqinYang/GOP-Glove.git
cd GOP-Glove
python -m pip install -r requirements.txt
python scripts/run_manuscript.py prepare
python scripts/verify_reproducibility.py
```

Feature extraction and partition checks alone use the smaller environment:

```bash
python -m pip install -r reproducibility/requirements-features.txt
python scripts/export_reproducibility.py
python scripts/verify_reproducibility.py
```

## Training and evaluation

```bash
# DA-LGBM, six LOSO folds
python scripts/run_manuscript.py train --model DA_LGBM --evaluation loso

# Pooled IID evaluation
python scripts/run_manuscript.py train --model DA_LGBM --evaluation iid

# Baseline example
python scripts/run_manuscript.py train --model LightGBM --evaluation loso

# Five stability seeds, using the same LOSO train/validation/test partitions
python scripts/run_manuscript.py seeds
```

Validation windows and held-out-subject windows are excluded from augmentation and scaler fitting. `--folds`, `--seed`, `--epochs` and `--output_dir` select the training run. Saved experiment outputs are under `reproducibility/results/`.

## Controlled robustness analysis

```bash
python scripts/run_manuscript.py robustness awgn
python scripts/run_manuscript.py robustness jitter
python scripts/run_manuscript.py robustness drift
```

AWGN levels are clean, 20, 10 and 5 dB; sampling jitter is clean and ±2%, ±5% and ±10%; drift is clean, light, medium and heavy. Each experiment trains on clean windows, keeps the fitted model and preprocessing fixed, and perturbs held-out test windows only. Results are saved directly from the computed predictions.

## 🚀 Arduino deployment

The sketch contains the 190-D feature extractor, branch forward passes, confidence gate and phase/timing markers. The two branch model headers are included with the sketch.

```bash
arduino-cli core install arduino:mbed_nano
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble   arduino/tinyml_inference/ADANN_LightGBM_inference/BSL_Gesture_Demo
```

Use the Arduino IDE or `arduino-cli upload` to upload to the connected Nano 33 BLE Sense Rev2. The serial interface starts an independent acquisition with `s` and prints timing summaries with `R`. Full-cycle timing includes acquisition; post-acquisition timing covers feature extraction, inference, gating and output.

## 📄 License

The existing project code is distributed under the [MIT License](LICENSE). Dataset columns and identifiers are described in [datasets/README.md](datasets/README.md).
