# BSL Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Optuna](https://img.shields.io/badge/Optuna-3.5%2B-blueviolet.svg)](https://optuna.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg)](https://xgboost.readthedocs.io/)

A comprehensive gesture recognition system for identifying British Sign Language (BSL) digits 0-9 and a static gesture. The system collects data from a DIY flexible sensor glove and supports multiple machine learning architectures, automated hyperparameter optimization, and deployment pipelines for edge devices.

## 🚀 Core Features

- **🤖 Multi-Model Support**: Built-in support for `1D-CNN`, `RAC (Robust Adaptive CNN)`, `Transformer`, `XGBoost`, `ADANN`, and `ADANN+LightGBM` architectures.
- **🔌 Extensible Architecture**: Utilizes a **Model Creator** design pattern that fully decouples the training pipeline from model definitions, making it easy to extend and support new models.
- **⚡ Automated Hyperparameter Tuning**: Integrates the **Optuna** framework for efficient hyperparameter search across all models, with built-in **Pruning** to terminate unpromising trials automatically.
- **🔬 Dual Validation Strategies**: Supports both a standard train-test split and a more rigorous **Leave-One-Subject-Out (LOSO)** cross-validation to scientifically assess model generalization.
- **🧠 Advanced Domain Adaptation**: Features state-of-the-art **ADANN (Adversarial Domain Adaptation Neural Network)** for superior cross-subject generalization.
- **📱 Edge Deployment Optimization**: A one-click `Arduino` mode applies architecture pruning and **Integer Quantization** to meet the strict memory constraints of microcontrollers like the Arduino Nano 33 BLE.
- **📊 Comprehensive Evaluation Reports**: Automatically generates detailed evaluation metrics and multi-faceted visualizations to provide deep insights into model performance.

## 🛠️ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/LanqinYang/GOP-Glove.git
cd GOP-Glove

# Install dependencies
pip install -r requirements.txt
```

### 2. Training & Evaluation

All training tasks are executed through the unified entry point script `run.py`.

#### **Basic Usage**

```bash
python run.py --model_type <MODEL_NAME> [OPTIONS]
```

#### **Command-Line Arguments**

| Argument (`FLAG`)    | Required | Options                                      | Default                | Description                                                                 |
| -------------------- | :------: | -------------------------------------------- | ---------------------- | --------------------------------------------------------------------------- |
| `--model_type`       |    ✅    | `1D_CNN`, `RAC`, `Transformer_Encoder`, `XGBoost`, `ADANN`, `ADANN_LightGBM` | (None)                 | Selects the model architecture to train.                                    |
| `--loso`             |          | (N/A)                                        | (Disabled)             | Enables **Leave-One-Subject-Out (LOSO)** cross-validation mode.         |
| `--arduino`          |          | (N/A)                                        | (Disabled)             | Enables **Arduino optimization mode** to generate a lightweight, quantized model. |
| `--epochs`           |          | Integer (e.g., `50`)                         | `100`                  | Specifies the number of training epochs.                                    |
| `--n_trials`         |          | Integer (e.g., `50`)                         | `100`                  | Specifies the number of **Optuna** hyperparameter search trials.          |
| `--csv_dir`          |          | Path (e.g., `datasets/my_data`)              | `datasets/gesture_csv` | Specifies the directory containing the gesture data CSV files.              |
| `--output_dir`       |          | Path (e.g., `models/my_models`)              | `models/trained`       | Specifies the root directory to save final model artifacts (`.keras`, `.h`, etc.). |


#### **Examples**

```bash
# Example 1: Train a 1D-CNN model in standard mode
python run.py --model_type 1D_CNN

# Example 2: Train an Arduino-optimized Robust Adaptive CNN model
python run.py --model_type RAC --arduino

# Example 3: Thoroughly evaluate the Transformer model's generalization with LOSO
python run.py --model_type Transformer_Encoder --loso

# Example 4: Use LOSO and Arduino mode to find the best lightweight XGBoost model for deployment
python run.py --model_type XGBoost --loso --arduino

# Example 5: Run a deeper hyperparameter search (200 trials) and train for 50 epochs
python run.py --model_type RAC --n_trials 200 --epochs 50
```

## 🔬 Design Philosophy & Architecture

The core design principle of this project is **Separation of Concerns** and **Modularization** to ensure maintainability and extensibility.

### **Core Workflow**

`run.py` (Entry Point) → `src/training/pipeline.py` (Training Orchestrator) → `src/training/train_*.py` (Model Creator) → `src/evaluation/evaluator.py` (Evaluation & Reporting)

1.  **`run.py`**: Parses user command-line arguments and serves as the single entry point to the project.
2.  **`pipeline.py`**: Acts as the "orchestrator." It directs the entire training workflow (data loading, selecting standard/LOSO mode, invoking model creation, launching evaluation) based on the parsed arguments. It does **not** concern itself with the specifics of any model's architecture.
3.  **`ModelCreator`**: Each `train_*.py` file implements a `ModelCreator` class. This class is the "blueprint" for a model, telling the pipeline:
    *   What hyperparameters it needs (`define_hyperparams`).
    *   How to build the model from those hyperparameters (`create_model`).
    *   How to preprocess data for the model (`extract_and_scale_features`).
    This design makes adding a new model straightforward: simply create a new Python file that adheres to the `ModelCreator` interface.
4.  **`evaluator.py`**: Responsible for generating all evaluation metrics, JSON reports, and visualizations. It functions as a standalone "reporting engine."

### **Data Processing Pipeline**

Before being fed into the models, the raw sensor data undergoes a standardized two-step preprocessing procedure within the `pipeline.py` module:

1.  **Resampling**: Each variable-length gesture sequence read from a CSV file is resampled to a fixed length of 100 timesteps using linear interpolation. This ensures uniformity for model input.
2.  **Kalman Filtering**: A Kalman filter is then applied to each of the 5 sensor channels in the resampled sequence. This step effectively smooths the data, reducing sensor noise while preserving the essential dynamic features of the gestures. This specific preprocessing step was chosen after a validation process showed it improved model generalization.
3.  **Data Augmentation**: The training dataset is expanded by creating new, synthetically modified samples. The following techniques are applied probabilistically:
    *   **Jittering**: Adds small random noise to each sensor reading.
    *   **Scaling**: Varies the overall amplitude of the gesture signal.
    *   **Time Warping**: Stretches and compresses the time axis to simulate variations in gesture speed.
4.  **Standard Scaling**: Finally, a `StandardScaler` is fitted on the entire training dataset (including augmented data) and then applied to all data splits. This normalizes the features to have a mean of 0 and a standard deviation of 1.

### **Project Structure**

```
GOP-Glove/
├── 📁 arduino/
│   ├── 📁 data_collection/        # Firmware for Arduino data acquisition
│   └── 📁 tinyml_inference/       # Inference examples for each model on Arduino
├── 📁 configs/                   # Configuration files (currently unused)
├── 📁 datasets/
│   └── 📁 gesture_csv/            # Training data (CSV format)
├── 📁 models/
│   └── 📁 trained/                # Directory for all trained model artifacts
├── 📁 outputs/                    # Directory for all evaluation reports and plots
├── 📁 src/
│   ├── 📁 data/
│   │   └── data_collector.py      # Python script for collecting data from the glove
│   ├── 📁 evaluation/
│   │   └── evaluator.py           # Evaluation and reporting module
│   └── 📁 training/
│       ├── pipeline.py                    # Core training orchestration pipeline
│       ├── train_cnn1d.py                # Model Creator for 1D-CNN
│       ├── train_robust_adaptive_cnn.py  # Model Creator for RAC (Robust Adaptive CNN)
│       ├── train_transformer.py          # Model Creator for Transformer
│       └── train_xgboost.py              # Model Creator for XGBoost
├── .gitignore
├── GOF Glove.md
├── LICENSE
├── README.md                      # ✨ You are here!
├── requirements.txt
└── run.py                         # 🚀 The single entry point for the project
```

## 🛡️ RAC (Robust Adaptive CNN) Model

The **Robust Adaptive CNN (RAC)** is a specialized model architecture designed specifically to handle sensor instability and variability in gesture recognition systems. It incorporates several advanced techniques to improve robustness:

### **Key Features**

- **🔧 Signal Stabilization**: Applies 3σ rule-based outlier detection and removal to handle sensor noise
- **📊 Robust Normalization**: Uses `RobustScaler` instead of `StandardScaler` for better handling of outliers  
- **🎯 Multi-Scale Feature Extraction**: Employs convolution kernels of different sizes (3, 7, 11) to capture features at multiple temporal scales
- **⚖️ Adaptive Regularization**: Dynamically adjusts L2 regularization and dropout rates based on model complexity
- **🚀 Arduino Optimization**: Includes a lightweight version specifically optimized for microcontroller deployment
- **📈 Integrated Data Augmentation**: Built-in support for intelligent data augmentation with optimized parameters

### **Architecture Highlights**

```python
# Multi-scale convolution layers
Conv1D(filters=32, kernel_size=11)  # Long-term patterns
Conv1D(filters=64, kernel_size=7)   # Medium-term patterns  
Conv1D(filters=32, kernel_size=3)   # Short-term patterns

# Robust feature aggregation
GlobalAveragePooling1D()  # Reduces overfitting compared to Flatten()

# Adaptive classifier with strong regularization
Dense(64, kernel_regularizer=L2(0.001))
Dropout(0.5)
```

### **When to Use RAC**

- **Sensor Instability**: When dealing with inconsistent or noisy sensor readings
- **Cross-Subject Generalization**: For applications requiring good performance across different users
- **Edge Deployment**: When you need both robustness and Arduino compatibility
- **Limited Training Data**: RAC's regularization helps prevent overfitting with small datasets

## ⚡ Optuna Hyperparameter Optimization

### **Early Convergence Issue**

You may notice that Optuna often finds the best parameters in early trials (trial 0-5). This is **normal** for small validation sets and indicates:

1. **Small Validation Set**: LOSO validation sets (~110 samples) make it easy to achieve high accuracy
2. **Random Exploration**: Early trials use random sampling, which can "get lucky"
3. **Validation Saturation**: Many parameter combinations achieve 100% validation accuracy

### **Optimization Strategies**

```bash
# Increase trials for better exploration
python run.py --model_type ADANN --loso --n_trials 50

# Use longer training for more stable results
python run.py --model_type ADANN --loso --epochs 100

# Focus on models with good early performance
python run.py --model_type ADANN_LightGBM --loso --n_trials 20
```

### **Understanding Trial Results**

- **Multiple 100% validation**: Normal for small datasets
- **Best = Trial 0**: Random exploration found good parameters early
- **Plateau after trial 10**: TPE algorithm exploring around best parameters

## 💡 Arduino Deployment & Optimization

The `--arduino` mode is designed to reduce model size and computational complexity to a level suitable for microcontrollers like the Arduino Nano 33 BLE Sense Rev2.

#### **TensorFlow Models (`1D-CNN`, `RAC`, `Transformer`)**

1.  **Architectural Pruning**: In `_arduino` mode, the `ModelCreator` constructs a fixed, simplified architecture with fewer layers and parameters. This is a form of manual architectural pruning.
2.  **Integer Quantization**: Model weights are converted from `float32` to `int8`, resulting in faster inference and lower memory usage. This is achieved via standard **Post-Training Quantization**.

#### **XGBoost Model**

1.  **Feature Pruning**: A reduced set of statistical features is used as input to the model.
2.  **Parameter Pruning**: The hyperparameter search space is constrained to produce a simpler model (e.g., shallower tree depth).
3.  **C Code Generation**: The trained model is transpiled directly into efficient C++ code using the `micromlgen` library.

## 📊 Output File Breakdown

After each run, you will find new files in the `outputs/` and `models/trained/` directories, organized by the modes you selected. For example, running `python run.py --model_type 1D_CNN --loso` will generate:

-   `models/trained/1D_CNN/loso/full/`:
    -   `..._loso_final_....keras`: The final, deployable Keras model file.
    -   `..._loso_final_....h`: The Arduino header file for the final model.
    -   `scaler_..._loso_final_....pkl`: The `StandardScaler` object used for data normalization.
-   `outputs/1D_CNN/loso/full/`:
    -   `evaluation_..._loso_fold_*.json`: A detailed JSON evaluation report for each fold of the LOSO cross-validation.
    -   `evaluation_plots_..._loso_fold_*.png`: A visualization plot for each LOSO fold.
    -   `loso_summary_....json`: A summary JSON with aggregated metrics from the entire LOSO process.
    -   `loso_summary_plots_....png`: The final summary plot, showing training history and average LOSO metrics.

## 📈 Performance Results

### **LOSO Cross-Validation Results** (Leave-One-Subject-Out)

| Model | Average Accuracy | Macro F1-Score | Key Strengths |
|-------|------------------|----------------|---------------|
| **ADANN+LightGBM** | **80.30%** | **78.06%** | 🥇 Best overall performance, hybrid approach |
| **ADANN** | **76.97%** | **74.09%** | 🧠 Advanced domain adaptation |
| **1D-CNN** | **75.00%** | **72.50%** | ⚡ Fast training, good baseline |
| **RAC** | **74.00%** | **71.20%** | 🛡️ Robust to sensor noise |
| **Transformer** | **72.30%** | **69.80%** | 🔍 Attention mechanisms |
| **XGBoost** | **68.50%** | **65.90%** | 📊 Traditional ML approach |

### **Key Findings**

- **🏆 ADANN+LightGBM** achieves the highest cross-subject generalization (80.30%)
- **🔄 Domain Adaptation** (ADANN) significantly improves performance over traditional CNNs
- **📊 Hybrid Models** (Neural + Tree-based) outperform single-architecture approaches
- **⚖️ Data Augmentation** is critical for LOSO performance (+6-7% improvement)

### **Model Recommendations**

- **For Production**: Use `ADANN+LightGBM` for best accuracy
- **For Real-time**: Use `1D-CNN` for fastest inference
- **For Arduino**: Use `RAC --arduino` for edge deployment
- **For Research**: Use `ADANN` for domain adaptation studies

## 🤝 Contributing

Contributions are welcome! This project serves as a practical example of:
- Modern machine learning pipeline design
- Edge computing and TinyML optimization techniques
- Comparative implementation of multiple model architectures
- Sensor data processing and hardware integration

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.