# BSL Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive gesture recognition system for British Sign Language (BSL) digits 0-9 and static gestures using a DIY flexible sensor glove. Features multiple ML architectures, automated hyperparameter optimization, and edge deployment capabilities.

## 🚀 Features

- **🤖 Multi-Model Support**: 1D CNN, Transformer, XGBoost, LightGBM, ADANN, and hybrid ADANN_LightGBM
- **⚡ Automated Optimization**: Optuna-based hyperparameter tuning with pruning
- **🔬 Rigorous Evaluation**: Standard train-test (Polled) and Leave-One-Subject-Out (LOSO) cross-validation strategy
- **🧠 Domain Adaptation**: ADANN for superior cross-subject generalization
- **📱 Edge Deployment**: Arduino models with C header generation
- **📊 Comprehensive Analysis**: Detailed evaluation metrics and visualizations

## 🛠️ Quick Start

### Installation

```bash
git clone https://github.com/yourusername/bsl-gesture-recognition.git
cd bsl-gesture-recognition
pip install -r requirements.txt
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

| Model            | Best LOSO Accuracy | Key Features                                   |
|------------------|-------------------|------------------------------------------------|
| **ADANN_LightGBM** | **85.30%**        | Combines domain adaptation and boosting for top accuracy |
| **ADANN**          | **80.30%**        | Domain adaptation for strong cross-subject generalization |
| **Transformer**    | **77.58%**        | Fast training, suitable for Arduino deployment  |
| **LightGBM**       | **77.12%**        | Lightweight, efficient, ideal for edge devices  |
| **XGBoost**        | **76.82%**        | Gradient boosting, robust to overfitting        |
| **1D_CNN**         | **75.91%**        | Deep learning baseline, effective for time series |

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

## 📊 Performance Highlights

- **Cross-Subject Generalization**: 85.30% accuracy with DA_LGBM
- **Real-time Inference**: <10ms latency on Arduino
- **Model Size**: <1MB for edge deployment
- **Robust Evaluation**: LOSO cross-validation across 6 subjects

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

### Model Optimization

- **Pruning**: Architectural and weight pruning for edge devices
- **Quantization**: INT8 quantization for reduced memory
- **Code Generation**: C/C++ code for traditional ML models

## 🤝 Contributing

We welcome contributions! This project demonstrates:
- Quantified instability (shift, drift, channel-specific) and its impact on cross-user genelization.
- DA-LGBM hybrid: ADANN (user-invariant features) + LightGBM (threshold-like cues) -> best LOSO result
- Deployed on Arduino-class MCU; latency bottleneck pinpointed outside the classifier.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.