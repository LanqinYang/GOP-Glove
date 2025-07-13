# BSL手势识别系统 - 多模型架构设计

## 概述

BSL手势识别系统支持多种机器学习模型架构，从传统的机器学习到最新的深度学习技术，为不同的应用场景和性能需求提供最适合的解决方案。

## 支持的模型架构

### 1. 1D-CNN (默认)
**描述**: 一维卷积神经网络，专门设计用于时序传感器数据处理

**优势**:
- 计算效率高，适合边缘部署
- 对硬件要求低
- 支持TensorFlow Lite转换
- 推理速度快

**适用场景**:
- Arduino等嵌入式设备部署
- 实时手势识别
- 资源受限环境

**配置文件**: `configs/config.yaml` - `1D_CNN`

### 2. XGBoost
**描述**: 基于梯度提升的传统机器学习模型

**优势**:
- 训练速度快
- 特征工程灵活
- 模型可解释性强
- 数据量要求相对较少

**特征工程**:
- 统计特征：均值、标准差、最大值、最小值等
- 频域特征：FFT变换特征
- 时域特征：时间序列统计特征
- 窗口特征：滑动窗口统计

**适用场景**:
- 性能基准测试
- 特征重要性分析
- 快速原型验证

**配置文件**: `configs/config.yaml` - `xgboost`

### 3. CNN-LSTM
**描述**: 结合卷积神经网络和长短期记忆网络的混合架构

**优势**:
- 同时捕获空间和时间特征
- 对长序列依赖性建模能力强
- 适合复杂手势模式识别

**架构设计**:
- CNN层：提取局部特征
- LSTM层：捕获时序依赖
- 全连接层：最终分类

**适用场景**:
- 复杂手势序列识别
- 高精度要求的应用
- 充足计算资源环境

**配置文件**: `configs/config.yaml` - `cnn_lstm`

### 4. Transformer Encoder
**描述**: 基于自注意力机制的Transformer编码器架构

**优势**:
- 并行计算能力强
- 长距离依赖建模优秀
- 注意力机制可解释性好
- 最新的深度学习技术

**架构特点**:
- 多头自注意力机制
- 位置编码
- 层归一化
- 前馈神经网络

**适用场景**:
- 研究和实验
- 高精度手势识别
- 注意力分析和可视化

**配置文件**: `configs/config.yaml` - `transformer_encoder`

## 文件夹结构

```
models/trained/
├── 1D_CNN/              # 1D-CNN模型文件
├── XGBoost/             # XGBoost模型文件  
├── CNN_LSTM/            # CNN-LSTM模型文件
└── Transformer_Encoder/ # Transformer模型文件
```

每个模型文件夹包含：
- **模型文件**: `.h5`、`.pkl`、`.tflite`
- **预处理器**: `scaler_*.pkl`
- **超参数**: `params_*.json`
- **评估结果**: `evaluation_*.json`
- **可视化图表**: `evaluation_plots_*.png`
- **详细预测**: `predictions_*.json`

## 使用方法

### 训练不同类型的模型

```bash
# 1D-CNN（默认）
python run.py train --model_type 1D_CNN

# XGBoost
python run.py train --model_type XGBoost

# CNN-LSTM
python run.py train --model_type CNN_LSTM  

# Transformer Encoder
python run.py train --model_type Transformer_Encoder
```

### 自定义训练参数

```bash
# 指定训练轮次和优化试验次数
python run.py train --model_type 1D_CNN --epochs 100 --n_trials 200

# 指定数据路径和输出路径
python run.py train --csv_dir datasets/gesture_csv --output_dir models/trained --model_type CNN_LSTM
```

## 模型比较和评估

系统为每种模型提供统一的评估框架：

### 评估指标
- **基础指标**: 准确率、损失
- **分类指标**: 精确率、召回率、F1分数
- **可视化**: 混淆矩阵、类别分布、置信度分析
- **错误分析**: 错误统计、低置信度预测

### 性能对比
不同模型的预期性能特点：

| 模型类型 | 训练速度 | 推理速度 | 精度 | 内存占用 | 边缘部署 |
|---------|---------|---------|------|---------|---------|
| 1D-CNN | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |
| XGBoost | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ |
| CNN-LSTM | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❓ |
| Transformer | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ |

## 扩展开发

### 添加新模型类型

1. **配置文件**: 在`configs/config.yaml`中添加新模型配置
2. **训练代码**: 扩展`src/training/train.py`中的模型创建函数
3. **文件夹**: 创建对应的模型输出文件夹
4. **文档**: 更新本文档和README

### 自定义评估指标

系统支持扩展评估指标，可在`src/training/train.py`的`comprehensive_evaluation`函数中添加新的评估逻辑。

## 最佳实践

1. **模型选择**: 根据部署环境和性能要求选择合适的模型
2. **数据预处理**: 所有模型共享相同的数据预处理流程
3. **超参数优化**: 使用Optuna进行自动超参数调优
4. **结果比较**: 使用统一的评估框架比较不同模型性能
5. **版本管理**: 通过时间戳管理不同训练的模型版本 