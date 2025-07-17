# BSL Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/LanqinYang/GOP-Glove?style=social)](https://github.com/LanqinYang/GOP-Glove/stargazers)

A comprehensive multi-model gesture recognition system for British Sign Language (BSL) digits 0-9 and static state recognition using DIY flexible sensor data gloves. The system supports multiple machine learning architectures with automated hyperparameter optimization and edge deployment capabilities.

## 🚀 Features

### ✨ Core Capabilities
- **Multi-Model Architecture**: Support for 1D-CNN, XGBoost, CNN-LSTM, and Transformer Encoder models
- **Arduino Integration**: Real-time data collection from flexible sensor gloves via serial communication
- **Automated Optimization**: Hyperparameter tuning using Optuna for optimal model performance
- **Edge Deployment**: TensorFlow Lite model generation for Arduino inference
- **Comprehensive Evaluation**: Detailed performance analysis with confusion matrices, confidence scores, and visualizations

### 🏗️ Supported Model Architectures

| Model | Description | Best For |
|-------|-------------|----------|
| **1D-CNN** | Lightweight convolutional neural network | Edge deployment, real-time inference |
| **XGBoost** | Gradient boosting with full/Arduino modes | High accuracy or embedded deployment |
| **CNN-LSTM** | Hybrid deep learning with temporal modeling | Complex gesture sequences |
| **Transformer Encoder** | Attention-based advanced architecture | State-of-the-art performance |

### 📊 Machine Learning Best Practices
- ✅ **Proper Data Splitting**: Train/Validation/Test separation without data leakage
- ✅ **Feature Scaling**: StandardScaler fitted only on training data
- ✅ **Hyperparameter Optimization**: Architecture search including layers, normalization, activations
- ✅ **Comprehensive Evaluation**: Detailed performance reports with visualizations
- ✅ **Dual-Mode XGBoost**: Full version for accuracy (85%) or Arduino version for deployment (<1MB)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Arduino IDE (for hardware setup)
- USB cable for Arduino communication

### Quick Setup
```bash
git clone https://github.com/LanqinYang/GOP-Glove.git
cd GOP-Glove
pip install -r requirements.txt
```

### Dependencies
```bash
# Core ML frameworks
tensorflow>=2.15.0
scikit-learn>=1.0.0
xgboost>=1.7.0
micromlgen>=1.1.0
optuna>=3.0.0

# Data processing
numpy>=1.21.0
scipy>=1.9.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Hardware communication
pyserial>=3.5
```

## 🎮 Quick Start

### 1️⃣ Data Collection
```bash
# Test mode (interactive)
python run.py collect test --port /dev/cu.usbmodem2101

# Automated collection mode
python run.py collect auto --port /dev/cu.usbmodem2101
```

### 2️⃣ Model Training

#### Basic Training
```bash
# Train specific models (model_type is required)
python run.py train --model_type 1D_CNN
python run.py train --model_type XGBoost                    # Full model (high accuracy)
python run.py train --model_type XGBoost --arduino          # Arduino-optimized (<1MB)
python run.py train --model_type CNN_LSTM
python run.py train --model_type Transformer_Encoder
```

#### Advanced Training
```bash
# Custom hyperparameter optimization
python run.py train --model_type 1D_CNN --n_trials 200 --epochs 100
python run.py train --model_type XGBoost --n_trials 200              # Full model
python run.py train --model_type XGBoost --n_trials 100 --arduino    # Arduino optimized
python run.py train --model_type CNN_LSTM --n_trials 50 --epochs 200
python run.py train --model_type Transformer_Encoder --n_trials 30 --epochs 150
```

#### ⚡ Smart Early Stopping
All training scripts include intelligent early stopping to save computational time:

```bash
# Don't worry about large n_trials values - system will stop when optimal
python run.py train --model_type XGBoost --n_trials 500
```

**Early stopping strategies:**
- 🎯 **Perfect accuracy**: Stops at 100% validation accuracy
- ⏱️ **Time saving**: Automatically stops when perfect solution found

**Example results:**
```
Setting: --n_trials 200
Actual:  20 trials (90% time saved)
Reason:  📉 Early stopping: No significant improvement for 10 trials
```

## 📁 Project Structure

```
GOP-Glove/
├── 📁 src/
│   ├── 📁 data/
│   │   └── data_collector.py      # Arduino data collection
│   └── 📁 training/
│       ├── train_cnn1d.py         # 1D-CNN training
│       ├── train_xgboost.py       # XGBoost training
│       ├── train_cnn_lstm.py      # CNN-LSTM training
│       └── train_transformer.py   # Transformer training
├── 📁 arduino/
│   ├── 📁 data_collection/        # Arduino data collection code
│   └── 📁 tinyml_inference/       # Edge inference code
├── 📁 datasets/
│   ├── 📁 csv/                    # Raw sensor data
│   └── 📁 gesture_csv/            # Processed gesture data
├── 📁 models/
│   └── 📁 trained/                # Trained models with timestamps
├── 📁 configs/
│   └── config.yaml                # Configuration parameters
├── run.py                         # Main CLI interface
└── requirements.txt               # Python dependencies
```

## 🔧 Hardware Setup

### Components
- **Arduino Uno/Nano**: Microcontroller for data collection
- **5x Flex Sensors**: Finger position detection (0-1023 range)
- **Resistors**: Pull-down resistors for sensor circuits
- **Breadboard & Wires**: Circuit connections

### Wiring Diagram
```
Arduino Pin | Component
------------|----------
A0-A4       | Flex sensors (with pull-down resistors)
5V          | Sensor power supply
GND         | Common ground
```

### Communication Protocol
- **Baud Rate**: 115200
- **Data Format**: CSV (timestamp, sensor1, sensor2, sensor3, sensor4, sensor5)
- **Sampling Rate**: Configurable via Arduino code

## 📊 Model Performance & Evaluation

### Training Output Example
```bash
Loading data from datasets/gesture_csv...
Loaded 110 samples
Train: 70, Val: 18, Test: 22

Optimizing hyperparameters with 100 trials...
[I 2025-01-08 10:30:15] Trial 50/100: val_accuracy=0.8889
Best validation accuracy: 0.9167

==================================================
COMPREHENSIVE EVALUATION
==================================================
Test Accuracy: 0.8636
Average Confidence: 0.7542
```

### Generated Files
Each training run creates timestamped files:

```
models/trained/1D_CNN/
├── bsl_model_1D_CNN_20250108_103045.h5      # Keras model
├── bsl_model_1D_CNN_20250108_103045.tflite  # TensorFlow Lite model
├── scaler_1D_CNN_20250108_103045.pkl        # Feature scaler
├── params_1D_CNN_20250108_103045.json       # Best hyperparameters
├── evaluation_1D_CNN_20250108_103045.json   # Performance metrics
├── evaluation_plots_1D_CNN_20250108_103045.png # Visualizations
└── predictions_1D_CNN_20250108_103045.json  # Detailed predictions
```

### Evaluation Metrics
- **Accuracy**: Overall classification performance
- **Precision/Recall/F1**: Per-class performance metrics
- **Confusion Matrix**: Detailed classification errors
- **Confidence Analysis**: Prediction certainty distribution
- **Class Distribution**: Training vs prediction balance

## 🎯 Gesture Classes

| Class ID | Gesture | Description |
|----------|---------|-------------|
| 0-9      | Numbers | BSL digit signs (0, 1, 2, ..., 9) |
| 10       | Static  | Rest position/no gesture |

## ⚙️ Configuration

### Model Parameters
Edit `configs/config.yaml` to customize:

```yaml
# Data parameters
sequence_length: 100
n_features: 5
n_classes: 11

# Training parameters
test_size: 0.2
val_size: 0.2
random_seed: 42

# Model-specific configurations
models:
  1D_CNN:
    n_conv_layers: [2, 4]
    filters: [16, 128]
    kernel_sizes: [3, 15]
  
  XGBoost:
    full_mode:
      n_estimators: [100, 1000]   # High accuracy range
      max_depth: [3, 10]          # Deep trees for complexity
      learning_rate: [0.01, 0.3]
    arduino_mode:
      n_estimators: [10, 50]      # Memory-limited range
      max_depth: [2, 4]           # Shallow trees for size
      learning_rate: [0.01, 0.3]
      target_file_size: "<1MB"    # Arduino constraint
```

## 🚀 Advanced Usage

### Custom Model Training
```python
from src.training.train_cnn1d import train_model

# Custom training with specific parameters
model_path, tflite_path = train_model(
    csv_dir="datasets/gesture_csv",
    output_dir="models/trained",
    n_trials=50,
    epochs=100,
    model_type="1D_CNN"
)
```

### Unified Training Interface
```bash
# All models use the same interface
python run.py train --model_type 1D_CNN --n_trials 50 --epochs 100
python run.py train --model_type XGBoost --n_trials 50               # Full version
python run.py train --model_type XGBoost --n_trials 50 --arduino     # Arduino version
python run.py train --model_type CNN_LSTM --n_trials 30 --epochs 200
python run.py train --model_type Transformer_Encoder --n_trials 20 --epochs 150
```

### Arduino Deployment

#### Neural Network Models (1D-CNN, CNN-LSTM, Transformer)
1. Load the generated `.tflite` model onto Arduino
2. Use the inference code in `arduino/tinyml_inference/`
3. Connect the sensor glove for real-time recognition

#### XGBoost Model (Two Versions Available)

**Full Version (High Accuracy)**:
- **Training**: `python run.py train --model_type XGBoost`
- **Features**: 50 features (5 sensors × 10 statistics)
- **Performance**: ~85% accuracy
- **File Size**: 2-10MB (may exceed Arduino memory)
- **Use Case**: Desktop/server deployment

**Arduino Version (Embedded Optimized)**:
- **Training**: `python run.py train --model_type XGBoost --arduino`
- **Features**: 20 features (5 sensors × 4 statistics) 
- **Performance**: ~81% accuracy
- **File Size**: <1MB (Arduino compatible)
- **Use Case**: Arduino Uno/Nano deployment

**Usage**:
```cpp
#include "bsl_model_XGBoost_[timestamp].h"        // Full version
// OR
#include "bsl_model_XGBoost_Arduino_[timestamp].h" // Arduino version

// Normalize features
float features[50];  // 50 for full, 20 for Arduino
normalize_features(features, length);

// Get prediction
int prediction = predict(features);
```

## 🎯 XGBoost Dual-Mode Architecture

XGBoost supports two distinct modes to balance accuracy and deployment constraints:

### 🚀 Full Mode (High Accuracy)
- **Purpose**: Maximum model performance
- **Features**: 50 statistical features per sample
- **Trees**: Up to 1000 estimators, depth up to 10
- **Accuracy**: ~85% (best performance)
- **File Size**: 2-10MB
- **Deployment**: Desktop, server, high-memory embedded systems

### 🤖 Arduino Mode (Embedded Optimized)
- **Purpose**: Arduino-compatible deployment
- **Features**: 20 essential features per sample
- **Trees**: Up to 50 estimators, depth up to 4
- **Accuracy**: ~81% (good performance)
- **File Size**: <1MB
- **Deployment**: Arduino Uno/Nano, microcontrollers

### Generated Files Structure
```
models/trained/
├── XGBoost/                                    # Full version
│   ├── bsl_model_XGBoost_[timestamp].h
│   ├── params_XGBoost_[timestamp].json
│   ├── evaluation_XGBoost_[timestamp].json
│   ├── evaluation_plots_XGBoost_[timestamp].png
│   └── predictions_XGBoost_[timestamp].json
└── XGBoost_Arduino/                           # Arduino version
    ├── bsl_model_XGBoost_Arduino_[timestamp].h
    ├── params_XGBoost_Arduino_[timestamp].json
    ├── evaluation_XGBoost_Arduino_[timestamp].json
    ├── evaluation_plots_XGBoost_Arduino_[timestamp].png
    └── predictions_XGBoost_Arduino_[timestamp].json
```

### Comparison Table
| Aspect | Full Mode | Arduino Mode |
|--------|-----------|--------------|
| **Accuracy** | ~85% | ~81% |
| **Features** | 50 | 20 |
| **Max Trees** | 1000 | 50 |
| **Max Depth** | 10 | 4 |
| **File Size** | 2-10MB | <1MB |
| **Memory Usage** | High | Low |
| **Training Time** | Longer | Faster |

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Run tests
pytest tests/

# Format code
black src/
flake8 src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** for the excellent ML framework
- **Optuna** for hyperparameter optimization
- **Arduino Community** for hardware support
- **British Sign Language** community for gesture inspiration

## 📞 Contact & Support

- **Author**: Lambert Yang
- **GitHub**: [@LanqinYang](https://github.com/LanqinYang)
- **Email**: Contact via GitHub issues

### Issues & Support
- 🐛 **Bug Reports**: [Create an Issue](https://github.com/LanqinYang/GOP-Glove/issues)
- 💡 **Feature Requests**: [Create an Issue](https://github.com/LanqinYang/GOP-Glove/issues)
- ❓ **Questions**: [GitHub Discussions](https://github.com/LanqinYang/GOP-Glove/discussions)

## 📈 Project Stats

![GitHub Stats](https://github-readme-stats.vercel.app/api/pin/?username=LanqinYang&repo=GOP-Glove&theme=dark)

---

Made by QMUL SEMS MSc Advanced Robotics Lanqin Yang