# BSL Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Optuna](https://img.shields.io/badge/Optuna-3.5%2B-blueviolet.svg)](https://optuna.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green.svg)](https://xgboost.readthedocs.io/)

A comprehensive gesture recognition system for identifying British Sign Language (BSL) digits 0-9 and a static gesture. The system collects data from a DIY flexible sensor glove and supports multiple machine learning architectures, automated hyperparameter optimization, and deployment pipelines for edge devices.

## 🚀 Core Features

 - **🤖 Multi-Model Support**: Built-in support for `1D_CNN`, `Transformer_Encoder`, `XGBoost`, `LightGBM`, `ADANN`, and `ADANN_LightGBM` architectures.
- **🔌 Extensible Architecture**: Utilizes a **Model Creator** design pattern that fully decouples the training pipeline from model definitions, making it easy to extend and support new models.
- **⚡ Automated Hyperparameter Tuning**: Integrates the **Optuna** framework for efficient hyperparameter search across all models, with built-in **Pruning** to terminate unpromising trials automatically.
- **🔬 Dual Validation Strategies**: Supports both a standard train-test split and a more rigorous **Leave-One-Subject-Out (LOSO)** cross-validation to scientifically assess model generalization.
- **🧠 Advanced Domain Adaptation**: Features state-of-the-art **ADANN (Adversarial Domain Adaptation Neural Network)** for superior cross-subject generalization.
 - **📱 Edge Deployment Optimization**: A one-click `Arduino` mode applies pruning and integer quantization for microcontrollers like the Arduino Nano 33 BLE. Note: `Transformer_Encoder` is not exportable to Arduino due to current TFLite Micro op limitations.
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

### 2. Upload Arduino data collector firmware

Use Arduino IDE to upload `arduino/data_collection/sensor_data_collector/sensor_data_collector.ino` to Arduino Nano 33 BLE Sense Rev2.

- Board: Arduino Nano 33 BLE Sense Rev2
- Port (macOS): like `/dev/cu.usbmodem*`
- Serial: 115200

After upload, keep the board connected by USB (the Python collector will control start/stop).

### 3. Collect gesture data (CSV)

Use the minimal Python collector to record sensor streams and save CSVs under `datasets/gesture_csv/` with the required naming convention used by the pipeline.

```bash
# 3.1 Quick sensor test (15s, saves to datasets/csv/)
python -m src.data.data_collector test --port /dev/cu.usbmodemXXXX --duration 15

# 3.2 Interactive per-class recording for full dataset (saves to datasets/gesture_csv/)
# Program will prompt for user ID once, then for each gesture 0..10 and each sample, press Enter to record 2s
python -m src.data.data_collector auto --port /dev/cu.usbmodemXXXX

# Naming produced by the collector (required by pipeline):
# datasets/gesture_csv/user_<ID>_gesture_<0..10>_<Name>_sample_<N>_<timestamp>.csv
```

Notes:
- The Arduino firmware streams 5 analog channels at ~50 Hz. The Python tool sends 'S'/'X' to control capture.
- The training pipeline automatically resamples every sequence to 100×5 and reads gesture/subject IDs from file names.

### 4. Training & Evaluation

All training tasks are executed through the unified entry point script `run.py`.

#### **Basic Usage**

```bash
python run.py --model_type <MODEL_NAME> [OPTIONS]
```

#### **Command-Line Arguments**

| Argument (`FLAG`)    | Required | Options                                      | Default                | Description                                                                 |
| -------------------- | :------: | -------------------------------------------- | ---------------------- | --------------------------------------------------------------------------- |
| `--model_type`       |    ✅    | `1D_CNN`, `Transformer_Encoder`, `XGBoost`, `LightGBM`, `ADANN`, `ADANN_LightGBM` | (None)                 | Selects the model architecture to train.                                    |
| `--loso`             |          | (N/A)                                        | (Disabled)             | Enables **Leave-One-Subject-Out (LOSO)** cross-validation mode.         |
| `--arduino`          |          | (N/A)                                        | (Disabled)             | Enables **Arduino optimization mode** to generate a lightweight, quantized model. |
| `--epochs`           |          | Integer (e.g., `50`)                         | `100`                  | Specifies the number of training epochs.                                    |
| `--n_trials`         |          | Integer (e.g., `50`)                         | `50`                   | Specifies the number of **Optuna** hyperparameter search trials.           |
| `--csv_dir`          |          | Path (e.g., `datasets/my_data`)              | `datasets/gesture_csv` | Specifies the directory containing the gesture data CSV files.              |
| `--output_dir`       |          | Path (e.g., `models/my_models`)              | `models/trained`       | Specifies the root directory to save final model artifacts (`.keras`, `.h`, etc.). |


#### **Examples (six models)**

```bash
# 1) 1D_CNN (standard)
python run.py --model_type 1D_CNN --epochs 100 --n_trials 50

# 2) LightGBM (fast and Arduino-friendly)
python run.py --model_type LightGBM --epochs 100 --n_trials 50

# 3) Transformer_Encoder with LOSO
python run.py --model_type Transformer_Encoder --loso --epochs 100 --n_trials 50

# 4) XGBoost with LOSO and Arduino mode
python run.py --model_type XGBoost --loso --arduino --epochs 100 --n_trials 50

# 5) ADANN deeper search
python run.py --model_type ADANN --epochs 50 --n_trials 200

# 6) ADANN_LightGBM (hybrid)
python run.py --model_type ADANN_LightGBM --epochs 100 --n_trials 50

```

### 5. Arduino Deployment (auto export + copy)

Use `--arduino` to trigger Arduino-optimized export and header copy. Headers are copied to the corresponding inference sketch folders under `arduino/tinyml_inference/*/Latency_*`.

```bash
# Example: 1D_CNN Arduino export (standard mode)
python run.py --model_type 1D_CNN --arduino --epochs 100 --n_trials 50

# Header locations after export (auto):
# arduino/tinyml_inference/1D_CNN_inference/Latency_standard/bsl_model_1D_CNN.h
# models/trained/1D_CNN/standard/arduino/bsl_model_1D_CNN_*.h (kept as artifacts)

# LightGBM also generates C header via m2cgen and is copied similarly.

Notes:
- `Transformer_Encoder` 暂不支持 Arduino 导出（缺少 TFLM BATCH_MATMUL）
- XGBoost/LightGBM 在 Arduino 模式下采用 20 维统计特征，保证与训练协议一致
```

Open Arduino IDE → `arduino/tinyml_inference/<Model>_inference/Latency_<standard|loso>/Latency_<...>.ino` → Upload → open Serial Monitor at 115200.

### 6. Latency benchmarking (Arduino / CPU / A100)

- Arduino: In Serial Monitor, run

```
record
latency 200 10
```

It prints JSON with latency and throughput for easy logging. Increase to `latency 1000 50` for smoother averages.

- CPU/A100 (Colab): Open notebooks and Run All
  - `src/test/latency_benchmark_test/Latency_Test_Arduino_CPU.ipynb` (desktop/Colab CPU)
  - `src/test/latency_benchmark_test/Latency_Test_A100.ipynb` (NVIDIA A100)

These notebooks load the exported models and measure end-to-end/inference latency depending on the cell selection.
```

<!-- Removed high-level architecture to keep README practitioner-focused -->

### **Data Processing Pipeline**

Before being fed into the models, the raw sensor data undergoes a standardized preprocessing procedure within the `pipeline.py` module:

1.  **Resampling**: Each variable-length gesture sequence read from a CSV file is resampled to a fixed length of 100 timesteps using linear interpolation. This ensures uniformity for model input.
2.  **Data Augmentation**: The training dataset is expanded by creating new, synthetically modified samples. The following techniques are applied probabilistically:
    *   **Jittering**: Adds small random noise to each sensor reading.
    *   **Scaling**: Varies the overall amplitude of the gesture signal.
    *   **Time Warping**: Stretches and compresses the time axis to simulate variations in gesture speed.
3.  **Standard Scaling**: Finally, a `StandardScaler` is fitted on the training data (including augmented data where applicable) and then applied to all data splits. This normalizes the features to have a mean of 0 and a standard deviation of 1.

### **Project Structure**

```
GOP-Glove/
├── 📁 arduino/
│   ├── 📁 data_collection/        # Firmware for Arduino data acquisition
│   └── 📁 tinyml_inference/       # Inference examples for each model on Arduino
├── 📁 configs/                   # Configuration files (e.g., data augmentation)
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
│       ├── train_cnn1d.py                # Model Creator for 1D_CNN
│       ├── train_transformer.py          # Model Creator for Transformer_Encoder
│       ├── train_xgboost.py              # Model Creator for XGBoost
│       ├── train_lightgbm.py             # Model Creator for LightGBM
│       ├── train_adann.py                # Model Creator for ADANN
│       └── train_adann_lightgbm.py       # Model Creator for ADANN_LightGBM (hybrid)
├── .gitignore
├── ADANN_LightGBM_算法详解.md
├── LICENSE
├── README.md                      # ✨ You are here!
├── requirements.txt
└── run.py                         # 🚀 The single entry point for the project
```

## 🧠 ADANN and Hybrid ADANN_LightGBM

ADANN provides domain-adaptive representation learning aimed at stronger cross-subject generalization. The hybrid `ADANN_LightGBM` combines a neural feature extractor with a LightGBM classifier to balance accuracy and deployability (LightGBM branch can be transpiled to C for Arduino).

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

#### **TensorFlow Models (`1D_CNN`, `Transformer_Encoder`)**

1.  **Architectural Pruning**: In `_arduino` mode, the `ModelCreator` constructs a fixed, simplified architecture with fewer layers and parameters. This is a form of manual architectural pruning.
2.  **Integer Quantization**: Model weights are converted from `float32` to `int8`, resulting in faster inference and lower memory usage. This is achieved via standard **Post-Training Quantization**. Note that `Transformer_Encoder` is currently not exported to Arduino due to missing TFLite Micro ops.

#### **XGBoost Model**

1.  **Feature Pruning**: A reduced set of statistical features is used as input to the model.
2.  **Parameter Pruning**: The hyperparameter search space is constrained to produce a simpler model (e.g., shallower tree depth).
3.  **C Code Generation**: The trained model is transpiled directly into efficient C++ code using the `micromlgen` library.

### 🔧 技术要点（论文可引用）

- 数据处理：
  - 线性重采样到 100×5（每序列固定长度）
  - 轻量数据增强：AddNoise、TimeWarp、按概率的幅度缩放（范围见配置）
  - 训练集拟合 `StandardScaler`，统一归一化所有切分
- 复现性设置：
  - 统一设定 Python/NumPy/TensorFlow 随机种子；启用 TensorFlow 确定性算子
  - Optuna 使用 TPESampler(seed=42) 与 MedianPruner 保证搜索可复现
- 超参数优化（Optuna）：
  - TensorFlow 系列：批大小、学习率、卷积/全连接单元、Dropout 等超参；早停与剪枝回调（TFKerasPruningCallback）
  - XGBoost/LightGBM：树深、学习率、子采样、叶子数等；集成 `XGBoostPruningCallback` 或内部验证
  - LOSO 下每折独立优化并汇总最佳试验
- 量化与剪枝：
  - Arduino 模式：
    - Keras 模型启用幅度剪枝（30%→60% 多项式衰减）；训练后 strip pruning
    - TFLite 全整数量化（INT8），使用代表性数据集确保量化校准
  - 生成 `.h` 头文件并自动复制到 `arduino/tinyml_inference/*/Latency_*`
- 传统模型部署：
  - LightGBM：`m2cgen` 生成 C 代码；设备端使用 20 维统计特征（每通道均值/方差/最小/最大）
  - XGBoost：`micromlgen` 生成 C 代码；同上特征集
- 混合模型 ADANN_LightGBM：
  - ADANN 提供域鲁棒特征；LightGBM 负责分类
  - 导出：ADANN 纯 C FP32（利用 M4F FPU），LightGBM 通过 m2cgen 生成 C 预测器
- 延迟评测方法：
  - Arduino：只计推理阶段（预处理/打包一次性准备），运行 `record` 后 `latency R W`，报告均值与 IPS
  - CPU/A100：提供 Notebook 进行端到端与推理阶段计时的可复现实验

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
| **ADANN_LightGBM** | **80.30%** | **78.06%** | 🥇 Best overall performance, hybrid approach |
| **ADANN** | **76.97%** | **74.09%** | 🧠 Advanced domain adaptation |
| **1D_CNN** | **75.00%** | **72.50%** | ⚡ Fast training, good baseline |
| **LightGBM** | **74.00%** | **71.20%** | 🛡️ Strong classic model, Arduino-friendly |
| **Transformer_Encoder** | **72.30%** | **69.80%** | 🔍 Attention mechanisms |
| **XGBoost** | **68.50%** | **65.90%** | 📊 Traditional ML approach |

### **Key Findings**

- **🏆 ADANN_LightGBM** achieves the highest cross-subject generalization (80.30%)
- **🔄 Domain Adaptation** (ADANN) significantly improves performance over traditional CNNs
- **📊 Hybrid Models** (Neural + Tree-based) outperform single-architecture approaches
- **⚖️ Data Augmentation** is critical for LOSO performance (+6-7% improvement)

### **Model Recommendations**

- **For Production**: Use `ADANN_LightGBM` for best accuracy
- **For Real-time**: Use `1D_CNN` for fastest inference
- **For Arduino**: Use `1D_CNN --arduino` or `LightGBM` for edge deployment
- **For Research**: Use `ADANN` for domain adaptation studies

## 🤝 Contributing

Contributions are welcome! This project serves as a practical example of:
- Modern machine learning pipeline design
- Edge computing and TinyML optimization techniques
- Comparative implementation of multiple model architectures
- Sensor data processing and hardware integration

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.