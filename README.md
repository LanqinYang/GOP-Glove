# BSL Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive gesture recognition system for British Sign Language (BSL) digits 0-9 and static gestures using a DIY flexible sensor glove. This repo contains: sensor data collection firmware, training/evaluation code, and MCU-class on-device deployment code.

## Manuscript (under review, 2026)
*Domain-Adversarial Light Gradient Boosting Machine for On-Device Sign Recognition with Graphite-on-Paper Sensors*  
Contact: ml23597@qmul.ac.uk

## 🚀 Features

- **Hybrid model (DA-LGBM):** ADANN (domain-adversarial, user-invariant representation) + LightGBM (high-accuracy classifier)
- **Confidence-gated fusion:** prediction-margin gating to mitigate sensor drift and inter-subject variability
- **Hardware closed-loop:** DIY 5-channel GoP glove + readout circuit + 50 Hz acquisition firmware
- **Edge deployment:** model translated to pure C (m2cgen) and deployed on Arduino Nano 33 BLE (256 KB SRAM)
- **Strict cross-subject evaluation:** LOSO Macro-F1 = 83.66%, ablation shows ~+6% gain from domain-adversarial module
- **Real-time on-device inference:** <0.5 s end-to-end pipeline latency on Arduino-class MCU

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

### Supported Models

| Model            | Best LOSO Macro-F1 | Key Features                                   |
|------------------|-------------------|------------------------------------------------|
| **ADANN_LightGBM** | **83.66%**        | Combines domain adaptation and boosting for top accuracy |
| **ADANN**          | **77.12%**        | Domain adaptation for strong cross-subject generalization |
| **Transformer**    | **74.99%**        | Fast training, suitable for Arduino deployment  |
| **LightGBM**       | **74.17%**        | Lightweight, efficient, ideal for edge devices  |
| **XGBoost**        | **74.79%**        | Gradient boosting, robust to overfitting        |
| **1D_CNN**         | **73.45%**        | Deep learning baseline, effective for time series |

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
