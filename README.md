# BSL Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive gesture recognition system for British Sign Language (BSL) digits 0-9 and static gestures using a DIY flexible sensor glove. Features multiple ML architectures, automated hyperparameter optimization, and edge deployment capabilities.

## 🚀 Features

- **🤖 Multi-Model Support**: 1D CNN, Transformer, XGBoost, LightGBM, ADANN, and hybrid ADANN_LightGBM
- **⚡ Automated Optimization**: Optuna-based hyperparameter tuning with pruning
- **🔬 Rigorous Evaluation**: Standard train-test and Leave-One-Subject-Out (LOSO) cross-validation
- **🧠 Domain Adaptation**: ADANN for superior cross-subject generalization
- **📱 Edge Deployment**: Arduino-optimized models with quantization and pruning
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

# Arduino-optimized model
python run.py --model_type LightGBM --arduino --epochs 100 --n_trials 50
```

### Supported Models

| Model | Best LOSO Accuracy | Key Features |
|-------|-------------------|--------------|
| **ADANN_LightGBM** | **80.30%** | 🥇 Hybrid approach, best performance |
| **ADANN** | **76.97%** | 🧠 Domain adaptation |
| **1D_CNN** | **75.00%** | ⚡ Fast training, Arduino-ready |
| **LightGBM** | **74.00%** | 🛡️ Classic ML, edge-friendly |
| **Transformer** | **72.30%** | 🔍 Attention mechanisms |
| **XGBoost** | **68.50%** | 📊 Traditional approach |

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

- **Cross-Subject Generalization**: 80.30% accuracy with ADANN_LightGBM
- **Real-time Inference**: <10ms latency on Arduino
- **Model Size**: <50KB for edge deployment
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

# CPU/A100 benchmarking
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
- Modern ML pipeline design
- Edge computing optimization
- Multi-architecture comparison
- Hardware-software integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{bsl_gesture_recognition_2024,
  title={BSL Gesture Recognition with Multi-Model Edge Deployment},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 🔗 Related Links

- [Arduino Documentation](https://docs.arduino.cc/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Optuna Documentation](https://optuna.org/)
- [TinyML Book](https://tinymlbook.com/)