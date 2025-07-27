# BSL Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Optuna](https://img.shields.io/badge/Optuna-3.5%2B-blueviolet.svg)](https://optuna.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg)](https://xgboost.readthedocs.io/)

A comprehensive gesture recognition system for identifying British Sign Language (BSL) digits 0-9 and a static gesture. The system collects data from a DIY flexible sensor glove and supports multiple machine learning architectures, automated hyperparameter optimization, and deployment pipelines for edge devices.

## 🚀 Core Features

- **🤖 Multi-Model Support**: Built-in support for `1D-CNN`, `CNN-LSTM`, `Transformer`, and `XGBoost` architectures.
- **🔌 Extensible Architecture**: Utilizes a **Model Creator** design pattern that fully decouples the training pipeline from model definitions, making it easy to extend and support new models.
- **⚡ Automated Hyperparameter Tuning**: Integrates the **Optuna** framework for efficient hyperparameter search across all models, with built-in **Pruning** to terminate unpromising trials automatically.
- **🔬 Dual Validation Strategies**: Supports both a standard train-test split and a more rigorous **Leave-One-Subject-Out (LOSO)** cross-validation to scientifically assess model generalization.
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
| `--model_type`       |    ✅    | `1D_CNN`, `CNN_LSTM`, `Transformer_Encoder`, `XGBoost` | (None)                 | Selects the model architecture to train.                                    |
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

# Example 2: Train an Arduino-optimized CNN-LSTM model
python run.py --model_type CNN_LSTM --arduino

# Example 3: Thoroughly evaluate the Transformer model's generalization with LOSO
python run.py --model_type Transformer_Encoder --loso

# Example 4: Use LOSO and Arduino mode to find the best lightweight XGBoost model for deployment
python run.py --model_type XGBoost --loso --arduino

# Example 5: Run a deeper hyperparameter search (200 trials) and train for 50 epochs
python run.py --model_type CNN_LSTM --n_trials 200 --epochs 50
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
│       ├── pipeline.py            # Core training orchestration pipeline
│       ├── train_cnn1d.py         # Model Creator for 1D-CNN
│       ├── train_cnn_lstm.py      # Model Creator for CNN-LSTM
│       ├── train_transformer.py   # Model Creator for Transformer
│       └── train_xgboost.py       # Model Creator for XGBoost
├── .gitignore
├── GOF Glove.md
├── LICENSE
├── README.md                      # ✨ You are here!
├── requirements.txt
└── run.py                         # 🚀 The single entry point for the project
```

## 💡 Arduino Deployment & Optimization

The `--arduino` mode is designed to reduce model size and computational complexity to a level suitable for microcontrollers like the Arduino Nano 33 BLE Sense Rev2.

#### **TensorFlow Models (`1D-CNN`, `CNN-LSTM`, `Transformer`)**

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

## 🤝 Contributing

Contributions are welcome! This project serves as a practical example of:
- Modern machine learning pipeline design
- Edge computing and TinyML optimization techniques
- Comparative implementation of multiple model architectures
- Sensor data processing and hardware integration

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.