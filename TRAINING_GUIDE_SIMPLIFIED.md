# BSL手势识别 - 简化训练指南

## 🚀 快速开始

### 基础训练
```bash
# 训练1D CNN模型（默认参数）
python train.py 1D_CNN

# 训练XGBoost模型（默认参数）  
python train.py XGBoost

# 训练CNN-LSTM模型（默认参数）
python train.py CNN_LSTM

# 训练Transformer编码器模型（默认参数）
python train.py Transformer_Encoder
```

### Arduino优化模式 🤖
```bash
# Arduino优化：生成<1MB的Arduino兼容.h文件
python train.py 1D_CNN --arduino
python train.py XGBoost --arduino
python train.py CNN_LSTM --arduino
python train.py Transformer_Encoder --arduino
```

### 自定义参数 ⚙️
```bash
# 自定义训练轮数和超参数优化试验次数
python train.py 1D_CNN --epochs 30 --n_trials 50

# Arduino模式 + 自定义参数
python train.py XGBoost --arduino --epochs 20 --n_trials 30
```

## ✨ 主要功能

### 🎯 统一训练接口
- **简单dispatcher设计**：一个命令调用所有模型
- **智能参数转发**：自动转发参数给对应的训练脚本
- **清晰的帮助信息**：`python train.py` 查看完整用法

### 🤖 Arduino模式优化
- **所有模型支持**：1D_CNN、CNN_LSTM、Transformer_Encoder、XGBoost
- **架构简化**：
  - 1D_CNN: 16→32 卷积滤波器，32个密集单元
  - CNN_LSTM: 简化卷积层+32单元LSTM
  - Transformer: 单层，32维嵌入，2个注意力头
  - XGBoost: 最多50棵树，最大深度4
- **自动量化**：TFLite量化优化，减小文件大小
- **文件大小验证**：自动检查是否<1MB

### 📁 输出文件
每次训练生成：
- **Arduino头文件**：`.h`格式，包含模型和归一化参数
- **参数文件**：最佳超参数和性能指标
- **详细评估**：混淆矩阵、分类报告、可视化图表
- **预测结果**：完整的预测数据

## 📊 性能参考

基于测试结果：

| 模型类型 | 标准模式精度 | Arduino模式精度 | 文件大小 |
|---------|------------|--------------|----------|
| XGBoost | ~77% | ~77% | 0.3MB ✅ |
| 1D_CNN | ~72% | ~45% | <1MB ✅ |
| CNN_LSTM | ~70% | ~53% | <1MB ✅ |
| Transformer | ~65% | ~59% | <1MB ✅ |

## 🛠️ 高级选项

### 参数说明
- `--epochs N`: 训练轮数（默认：50）
- `--n_trials N`: 超参数优化试验次数（默认：100）
- `--arduino`: 启用Arduino优化模式

### 文件结构
```
models/trained/
├── 1D_CNN/                    # 标准模式输出
│   ├── bsl_model_*.h         # Arduino头文件
│   ├── params_*.json         # 训练参数
│   └── evaluation_*.json     # 评估结果
└── 1D_CNN_Arduino/           # Arduino模式输出
    ├── bsl_model_*.h         # 优化的Arduino头文件
    ├── params_*.json         # Arduino训练参数
    └── evaluation_*.json     # Arduino评估结果
```

## 🎯 推荐使用

### 最佳精度
```bash
python train.py XGBoost --epochs 50 --n_trials 100
```

### Arduino项目
```bash
python train.py XGBoost --arduino --epochs 30 --n_trials 50
```

### 快速测试
```bash
python train.py 1D_CNN --arduino --epochs 10 --n_trials 10
```

## 🔧 技术说明

- **随机种子固定**：保证实验可重现
- **数据分割**：训练60%，验证20%，测试20%
- **早停机制**：防止过拟合
- **自动评估**：生成详细的性能报告和可视化

## 💡 提示

1. **Arduino模式**适合资源受限的嵌入式项目
2. **XGBoost**在Arduino模式下表现最佳
3. **标准模式**追求最高精度
4. 所有模式都会生成Arduino兼容的`.h`头文件 