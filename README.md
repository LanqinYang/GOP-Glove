# BSL 手势识别系统

一个基于CNN的手势识别系统，使用弯曲传感器采集手势数据，支持0-9数字手势识别以及静止状态识别。

## 功能特性

### 核心功能
- **数据采集**: 支持Arduino串口数据采集
- **智能训练**: 基于Optuna的超参数优化
- **全面评估**: 包含混淆矩阵、分类报告、置信度分析等
- **边缘部署**: 生成TensorFlow Lite模型用于Arduino推理

### 数据科学最佳实践
- **正确的数据分割**: 训练/验证/测试集分离
- **标准化**: 在训练集上拟合缩放器，避免数据泄漏
- **超参数优化**: 架构搜索包括层数、BN、激活函数、Dropout等
- **详细评估**: 生成完整的模型性能报告和可视化

## 系统架构

### 硬件组件
- Arduino微控制器
- 5个弯曲传感器 (0-1023数值范围)
- 串口通信 (115200波特率)

### 软件架构
- **数据采集**: `src/data/data_collector.py`
- **模型训练**: `src/training/train.py`
- **主程序**: `run.py`

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 数据采集
```bash
# 测试模式
python run.py collect test --port /dev/cu.usbmodem2101

# 自动采集模式
python run.py collect auto --port /dev/cu.usbmodem2101
```

### 3. 模型训练
```bash
# 基础训练
python run.py train

# 自定义参数
python run.py train --n_trials 100 --epochs 50
```

## 评估系统

### 生成的文件
训练完成后，系统会生成以下文件（带时间戳）：

#### 模型文件
- `bsl_model_TIMESTAMP.h5` - Keras模型
- `bsl_model_TIMESTAMP.tflite` - TensorFlow Lite模型
- `scaler_TIMESTAMP.pkl` - 数据缩放器
- `params_TIMESTAMP.json` - 最优超参数

#### 评估文件
- `evaluation_TIMESTAMP.json` - 完整评估报告
- `evaluation_plots_TIMESTAMP.png` - 可视化图表
- `predictions_TIMESTAMP.json` - 详细预测结果

### 评估指标

#### 1. 基础评估
- **测试损失**: 模型在测试集上的损失值
- **测试准确率**: 整体分类准确率

#### 2. 类别分布分析
- **真实分布**: 各类别在测试集中的样本数
- **预测分布**: 模型预测的各类别样本数
- **分布偏差**: 识别模型的预测偏好

#### 3. 混淆矩阵分析
- **混淆矩阵**: 详细的分类错误统计
- **最易混淆组合**: 找出最容易混淆的手势对
- **可视化热力图**: 直观显示分类结果

#### 4. 分类报告
- **精确率 (Precision)**: 预测为正例中实际为正例的比例
- **召回率 (Recall)**: 实际正例中被预测为正例的比例
- **F1分数**: 精确率和召回率的调和平均
- **支持度**: 各类别的样本数量

#### 5. 单类别准确率
- **每个手势的准确率**: 0-9各个数字的识别准确率
- **性能差异**: 识别哪些手势表现最好/最差

#### 6. 置信度分析
- **平均置信度**: 模型预测的平均置信度
- **置信度分布**: 预测置信度的分布情况
- **低置信度预测**: 识别不确定的预测

#### 7. 错误分析
- **错误总数**: 分类错误的样本数
- **错误率**: 错误样本占总样本的比例
- **错误示例**: 具体的错误预测案例

#### 8. 可视化图表
生成6个子图的综合评估图表：
- **混淆矩阵热力图**: 显示分类结果
- **单类别准确率柱状图**: 各手势识别性能
- **类别分布对比**: 真实vs预测分布
- **置信度分布直方图**: 预测置信度统计
- **精确率/召回率/F1对比**: 多指标性能比较
- **各类别错误数统计**: 错误分析可视化

## 技术特点

### 超参数优化
- **架构搜索**: 2-4层卷积网络
- **正则化**: 批归一化、Dropout可选
- **激活函数**: ReLU、Tanh、Swish
- **卷积核大小**: 覆盖序列长度的10% (3-15)
- **优化器**: Adam优化器，学习率自适应

### 数据处理
- **序列长度标准化**: 重采样到100个时间步
- **特征缩放**: StandardScaler标准化
- **数据分割**: 训练64%，验证16%，测试20%
- **无数据泄漏**: 缩放器只在训练集上拟合

### 模型评估
- **交叉验证**: 基于验证集的早停机制
- **多指标评估**: 准确率、精确率、召回率、F1分数
- **可视化分析**: 混淆矩阵、置信度分布等
- **结果保存**: JSON格式，便于后续分析

## 示例输出

### 训练过程
```bash
Loading data from datasets/gesture_csv...
Loaded 100 samples
Train: 64, Val: 16, Test: 20
Fitting scaler on training data...
Optimizing hyperparameters with 100 trials...
Best validation accuracy: 0.9375
```

### 评估结果
```bash
==================================================
COMPREHENSIVE EVALUATION
==================================================

1. Basic Evaluation:
   Test Loss: 0.6911
   Test Accuracy: 0.7500

4. Confusion Matrix Analysis:
   Top confusions:
   Gesture_8 → Gesture_9: 2 times
   Gesture_1 → Gesture_7: 1 times

7. Confidence Analysis:
   Average confidence: 0.5700
   Low confidence predictions (<0.5): 6

8. Error Analysis:
   Total errors: 5
   Error rate: 0.25
```

### 生成文件
```
models/trained/
├── bsl_model_20250706_030330.h5 (154KB)
├── bsl_model_20250706_030330.tflite (20KB)
├── scaler_20250706_030330.pkl
├── params_20250706_030330.json
├── evaluation_20250706_030330.json (完整评估报告)
├── evaluation_plots_20250706_030330.png (可视化图表)
└── predictions_20250706_030330.json (详细预测结果)
```

## 系统要求
- Python 3.8+
- TensorFlow 2.15+
- Arduino IDE (用于数据采集)
- 串口支持

## 许可证
MIT License 