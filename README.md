# BSL Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg)](https://xgboost.readthedocs.io/)

A comprehensive multi-model gesture recognition system for British Sign Language (BSL) digits 0-9 and static state recognition using DIY flexible sensor data gloves. The system supports multiple machine learning architectures with automated hyperparameter optimization and edge deployment capabilities.

## 🚀 Key Features

- **🤖 Multi-Model Support**: 1D-CNN, CNN-LSTM, Transformer, and XGBoost architectures
- **⚡ Smart Training**: Unified interface with intelligent early stopping and Optuna-based pruners
- **🔬 Robust Validation**: Standard train/test split and Leave-One-Subject-Out (LOSO) cross-validation
- **📱 Edge Deployment**: Arduino-optimized models with pruning+quantization for 256KB limit
- **📊 Comprehensive Evaluation**: Detailed performance analysis with visualizations
- **🔧 Hardware Integration**: Real-time data collection from flexible sensor gloves

## 🏗️ Supported Model Architectures

| Model | Description | Accuracy | Deploy Size | Arduino Mode |
|-------|-------------|----------|-------------|--------------|
| **1D-CNN** | Lightweight CNN | ~80% | 0.05MB | ✅ Pruned + Quantized |
| **XGBoost** | Gradient boosting | ~85% | 0.3MB | ✅ Optimized (Features + Params) |
| **CNN-LSTM** | Hybrid temporal model | ~85% | 0.2MB | ✅ Pruned + Quantized |
| **Transformer** | Attention mechanism | ~85% | 0.2MB | ✅ Pruned + Quantized |

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

# Leave-One-Subject-Out (LOSO) cross-validation
python src/training/train.py Transformer_Encoder --loso --n_trials 50

# Advanced hyperparameter tuning
python src/training/train.py CNN_LSTM --n_trials 200 --epochs 100
python src/training/train.py Transformer_Encoder --n_trials 100 --epochs 50
```

### Smart Early Stopping & Pruning
All models include intelligent hyperparameter search and early stopping:
- 🎯 **Optuna Integration**: Automated hyperparameter tuning to find the best model configurations.
- **Pruning**: All Optuna trials use a `Pruner` (`MedianPruner` for TF models, `XGBoostPruningCallback` for XGBoost) to automatically stop unpromising trials early.
- ⏱️ **Time Saving**: No wasted computation on poor hyperparameter combinations.
- 📊 **Large `n_trials` Safe**: Set high trial counts without worry, as bad runs are terminated quickly.

## 🔬 Arduino Optimization Techniques

Our Arduino mode applies aggressive model compression to meet the **256KB flash memory limit** of the Arduino Nano 33 BLE Sense Rev2.

### Pruning + Quantization (TensorFlow Models)
For `1D-CNN`, `CNN-LSTM`, and `Transformer` models, the pipeline is as follows:

1.  **Structural Pruning**: A simplified, fixed model architecture is used, which reduces the number of layers and parameters compared to the full version. This is a manual form of architectural pruning.
2.  **Quantization**: Model weights are converted from `float32` to `float16` precision, halving the model size with minimal impact on accuracy.
3.  **TensorFlow Lite Conversion**: The model is converted into the standard `.tflite` format for edge devices.

### Feature & Parameter Pruning (XGBoost)
The `XGBoost` model is optimized differently, as it is not a neural network:

1.  **Feature Pruning**: A reduced set of statistical features is extracted from the raw sensor data, making the input to the model smaller.
2.  **Parameter Pruning**: The hyperparameter search space is constrained to produce a simpler model (e.g., fewer trees, shallower depth).
3.  **Arduino C-Code Generation**: The final model is converted directly into C++ code using the `micromlgen` library, which is highly efficient for deployment on microcontrollers.

### Hardware Constraints (Arduino Nano 33 BLE Sense Rev2)
- **Flash Memory**: 1MB total, **256KB limit** for model storage
- **SRAM**: 256KB for runtime operations
- **Clock**: 64MHz ARM Cortex-M4F processor
- **AI Acceleration**: Built-in DSP instructions for efficient inference

## 🧪 Validation Strategies

This project supports two primary validation strategies to ensure model robustness.

### 1. Standard Train-Validation-Test Split
- **Usage**: Default training mode (`python src/training/train.py <MODEL_TYPE>`).
- **Process**: The dataset is randomly split into three sets:
    - **Training Set**: Used to train the model.
    - **Validation Set**: Used during hyperparameter tuning (Optuna) to evaluate trial performance.
    - **Test Set**: A completely held-out set used only once at the end to report the final model accuracy.
- **Best For**: Quick training and getting a general performance baseline.

### 2. Leave-One-Subject-Out (LOSO) Cross-Validation
- **Usage**: Activated with the `--loso` flag (`python src/training/train.py <MODEL_TYPE> --loso`).
- **Process**: This is a more rigorous, user-independent validation method.
    - For each subject (user) in the dataset, a model is trained on all *other* subjects' data and then tested on that one subject's data.
    - This process is repeated for every subject.
    - The final reported accuracy is the average accuracy across all subjects.
- **Best For**: Assessing the model's ability to generalize to new, unseen users, which is critical for real-world applications.

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