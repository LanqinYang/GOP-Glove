# BSL Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A controlled six-participant engineering feasibility study of an isolated 11-class British Sign Language (BSL) vocabulary using a five-channel graphite-on-paper glove. This repository contains the pseudonymised sensor recordings, training and diagnostic sources, and archived embedded source files.

## Manuscript (under review, 2026)
*Domain-Adversarial Light Gradient Boosting Machine for On-Device Recognition of Isolated Gestures from a Constrained BSL Vocabulary Using Graphite-on-Paper Sensors*
Contact: ml23597@qmul.ac.uk

## 🚀 Features

- **Hybrid model (DA-LGBM):** ADANN (domain-adversarial, user-invariant representation) + LightGBM (high-accuracy classifier)
- **Confidence-gated fusion:** prediction-margin gating to mitigate sensor drift and inter-subject variability
- **Hardware closed-loop:** DIY 5-channel GoP glove + readout circuit + 50 Hz acquisition firmware
- **Edge deployment:** model translated to pure C (m2cgen) and deployed on Arduino Nano 33 BLE (256 KB SRAM)
- **Cross-subject evaluation:** six held-out-subject folds, with original diagnostic results and provenance recorded separately.
- **Deployment timing:** approximately 2.5 s per full recognition cycle, including two-second acquisition and less than 0.5 s of post-acquisition processing in the archived indoor, USB-connected setup.

## 🛠️ Quick Start

### Installation

```bash
git clone https://github.com/LanqinYang/GOP-Glove.git
cd GOP-Glove
python -m pip install -r requirements.txt
```

### Data Collection

1. **Upload Arduino Firmware**:
   ```bash
   # Upload to Arduino Nano 33 BLE Sense Rev2
   # File: arduino/data_collection/sensor_data_collector/sensor_data_collector.ino
   ```

2. **Collect Gesture Data**:
   ```bash
   # Test sensor (15s)
   python -m src.data.data_collector test --port /dev/cu.usbmodemXXXX --duration 15
   
   # Full dataset collection
   python -m src.data.data_collector auto --port /dev/cu.usbmodemXXXX
   ```

### Training

```bash
# Basic training
python run.py --model_type 1D_CNN --epochs 100 --n_trials 50

# LOSO cross-validation
python run.py --model_type ADANN_LightGBM --loso --epochs 100 --n_trials 50

```

### Recorded result sources

| Source | Mean Macro-F1 | Mean accuracy | Status |
|---|---:|---:|---|
| Archived DA-LGBM confusion matrices | 0.8366 | 0.8530 | Archived counts; exact training-checkpoint correspondence remains open |
| Five-seed offline diagnostic | 0.7931 | 0.8139 | Raw summary arithmetic verified |
| Separate seed-810 gate audit | 0.7717 | 0.7909 | Independent diagnostic; includes the Static audit |

These are separate result sources. They are not one reproduced training run.
See [reproduction status](reproducibility/REPRODUCTION_STATUS.md).


## 📁 Project Structure

```
├── src/
│   ├── training/          # Model training scripts
│   ├── data/             # Data collection and processing
│   └── test/             # Testing and evaluation
├── arduino/
│   ├── data_collection/  # Arduino firmware
│   └── tinyml_inference/ # Edge deployment code
├── datasets/
│   └── gesture_csv/      # Training data
├── configs/              # Configuration files
├── models/               # Trained models
├── outputs/              # Evaluation results
└── run.py               # Main entry point
```

## 🔧 Key Technologies

- **Machine Learning**: TensorFlow, XGBoost, LightGBM, Optuna
- **Hardware**: Arduino Nano 33 BLE Sense Rev2
- **Edge Computing**: TensorFlow Lite, TinyML
- **Data Processing**: NumPy, Pandas, Scikit-learn

## 🚀 Deployment

### Arduino Deployment

```bash
# Generate Arduino-optimized model
python run.py --model_type 1D_CNN --arduino --epochs 100 --n_trials 50

# Upload inference code to Arduino
# File: arduino/tinyml_inference/1D_CNN_inference/Latency_standard/Latency_*.ino
```

### Latency Testing

```bash
# Arduino latency test
# In Serial Monitor: latency 200 10

# Colab CPU benchmarking
# Open: src/test/Latency_test_CPU.ipynb
```

## 📈 Advanced Features

### Hyperparameter Optimization

- **Optuna Integration**: Automated search with pruning
- **Early Convergence**: Efficient trial management
- **Reproducible Results**: Fixed random seeds

### Data Processing Pipeline

1. **Resampling**: Fixed 100 timesteps per sequence
2. **Augmentation**: Jittering, scaling, time warping
3. **Normalization**: StandardScaler for consistent features

### Model Deployment

- **Code Generation**: C/C++ code for ALL models (Transformer cannot infer)
- **Convert Tool**: TensorFlowLite, PyTorch, m2cgen, micromlgen

## 🤝 Contributing

I welcome contributions! This project demonstrates:
- Quantified instability (shift, drift, channel-specific) and its impact on cross-user genelization.
- DA-LGBM hybrid: ADANN (user-invariant features) + LightGBM (threshold-like cues) -> best LOSO result
- Deployed on Arduino-class MCU; latency bottleneck pinpointed outside the classifier.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Inspect the reproducibility materials

```bash
python -m pip install -r reproducibility/requirements-audit.txt
python scripts/export_reproducibility.py
python scripts/verify_reproducibility.py
```

The check verifies dataset identity, exported split isolation, descriptor
shape and raw diagnostic arithmetic. Training and physical-device profiling
were not rerun by this material audit. Current split manifests use sorted
filenames and are not claimed as recovered historical primary-run splits.
The legacy hardware demo differs from the offline margin/Static gate; read
[reproduction status](reproducibility/REPRODUCTION_STATUS.md) before reuse.
