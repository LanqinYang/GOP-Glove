# BSL Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg)](https://xgboost.readthedocs.io/)

A comprehensive multi-model gesture recognition system for British Sign Language (BSL) digits 0-9 and static state recognition using DIY flexible sensor data gloves. The system supports multiple machine learning architectures with automated hyperparameter optimization and edge deployment capabilities.

## 🚀 Key Features

- **🤖 Multi-Model Support**: 1D-CNN, CNN-LSTM, Transformer, and XGBoost architectures
- **⚡ Smart Training**: Unified interface with intelligent early stopping
- **📱 Edge Deployment**: Arduino-optimized models with pruning+quantization for 256KB limit
- **📊 Comprehensive Evaluation**: Detailed performance analysis with visualizations
- **🔧 Hardware Integration**: Real-time data collection from flexible sensor gloves

## 🏗️ Supported Model Architectures

| Model | Description | Accuracy | Deploy Size | Arduino Mode |
|-------|-------------|----------|-------------|--------------|
| **1D-CNN** | Lightweight CNN | ~80% | 0.05MB | ✅ Pruned+Quantized |
| **XGBoost** | Gradient boosting | ~85% | 0.3MB | ✅ Arduino-optimized |
| **CNN-LSTM** | Hybrid temporal model | ~85% | 0.2MB | ✅ Compressed |
| **Transformer** | Attention mechanism | ~85% | 0.2MB | ✅ Lightweight |

## 🛠️ Quick Start

### Installation
```bash
git clone https://github.com/LanqinYang/GOP-Glove.git
cd GOP-Glove
pip install -r requirements.txt
```

### Training Models
```bash
# Basic training with smart defaults
python src/training/train.py 1D_CNN
python src/training/train.py XGBoost

# Arduino-optimized models (256KB limit)
python src/training/train.py 1D_CNN --arduino
python src/training/train.py XGBoost --arduino

# Advanced hyperparameter tuning
python src/training/train.py CNN_LSTM --n_trials 200 --epochs 100
python src/training/train.py Transformer_Encoder --n_trials 100 --epochs 50
```

### Smart Early Stopping
All models include intelligent early stopping:
- 🎯 **Automatic termination** when 100% validation accuracy is reached
- ⏱️ **Time saving** - no wasted computation on perfect solutions
- 📊 **Large n_trials safe** - set high values without worry

## 🔬 Arduino Optimization Techniques

### Pruning + Quantization Pipeline
Our Arduino mode applies aggressive model compression:

1. **Pruning**: Removes redundant neural connections
2. **Quantization**: Converts float32 → float16 precision
3. **Architecture Optimization**: Reduces layer sizes for edge constraints
4. **TensorFlow Lite**: Standard conversion with hardware-specific ops

### Hardware Constraints (Arduino Nano 33 BLE Sense Rev2)
- **Flash Memory**: 1MB total, **256KB limit** for model storage
- **SRAM**: 256KB for runtime operations
- **Clock**: 64MHz ARM Cortex-M4F processor
- **AI Acceleration**: Built-in DSP instructions for efficient inference

## 📁 Project Structure

```
BSL-Gesture-Recognition/
├── 📁 src/
│   ├── 📁 training/
│   │   ├── train.py              # 🚀 Unified training interface
│   │   ├── train_cnn1d.py        # 1D-CNN implementation
│   │   ├── train_cnn_lstm.py     # CNN-LSTM implementation  
│   │   ├── train_transformer.py  # Transformer implementation
│   │   └── train_xgboost.py      # XGBoost implementation
│   └── 📁 data/
│       └── data_collector.py     # Arduino data collection
├── 📁 arduino/
│   ├── 📁 data_collection/       # Arduino firmware
│   └── 📁 tinyml_inference/      # Edge inference examples
├── 📁 datasets/
│   └── 📁 gesture_csv/           # Gesture data (CSV format)
├── 📁 models/
│   └── 📁 trained/               # Trained models by type
│       ├── 📁 1D_CNN/           # Regular CNN models
│       ├── 📁 1D_CNN_Arduino/   # Arduino-optimized CNN
│       ├── 📁 XGBoost/          # Regular XGBoost models
│       └── 📁 XGBoost_Arduino/  # Arduino-optimized XGBoost
└── requirements.txt
```

## 🔧 Hardware Components

- **Microcontroller**: Arduino Nano 33 BLE Sense Rev2 (nRF52840)
- **Memory**: 1MB Flash / 256KB SRAM
- **Sensors**: 5x flexible bend sensors (thumb to pinky)
- **Communication**: Serial over USB (115200 baud)
- **Sampling Rate**: 50Hz stable data acquisition
- **AI Features**: ARM Cortex-M4F with DSP extensions

## 📊 Performance Results

### Model Comparison (Arduino Nano 33 BLE Sense Rev2)
```
Model                Accuracy    File Size    Training Time    Arduino Compatible
1D_CNN (Arduino)     ~80%        50KB        ~2 min          ✅ Pruned+Quantized
XGBoost (Arduino)    ~85%        240KB       ~1 min          ✅ Optimized
CNN_LSTM (Arduino)   ~85%        180KB       ~5 min          ✅ Compressed  
Transformer (Arduino) ~85%       200KB       ~8 min          ✅ Lightweight
```

### Gesture Classes
- **Digits**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Static**: Rest/neutral position
- **Total**: 11 distinct classes

## 💡 Usage Examples

### Data Collection
```bash
# Test sensor connectivity
python src/data/data_collector.py test --port /dev/cu.usbmodem2101

# Collect gesture data
python src/data/data_collector.py auto --port /dev/cu.usbmodem2101
```

### Model Training & Evaluation
```bash
# Quick training with evaluation plots
python src/training/train.py 1D_CNN --arduino

# Custom hyperparameter search
python src/training/train.py XGBoost --n_trials 500

# Full training with detailed evaluation
python src/training/train.py CNN_LSTM --epochs 100 --n_trials 200
```

### Generated Outputs
Each training run produces:
- ✅ **Model files**: `.keras`, `.tflite`, `.h` (Arduino header)
- ✅ **Evaluation**: JSON metrics + PNG visualization plots
- ✅ **Parameters**: Best hyperparameters in JSON format
- ✅ **Predictions**: Detailed prediction results

## 🤝 Contributing

Contributions are welcome! This project demonstrates:
- Modern ML pipeline design
- Edge deployment optimization with pruning+quantization
- Multi-model architecture comparison
- Arduino Nano 33 BLE/TinyML integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

Built for British Sign Language gesture recognition research and practical deployment in Arduino Nano 33 BLE Sense Rev2 embedded systems with 256KB memory constraints.