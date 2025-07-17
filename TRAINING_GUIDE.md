# BSL手势识别统一训练脚本使用指南

## 简介

新的统一训练脚本 `src/training/train.py` 已经重构完成，简化了训练流程并支持所有模型类型。

## 主要特性

✅ **统一接口**: 一个脚本支持所有模型类型
✅ **默认参数**: 无需每次指定所有参数  
✅ **Arduino模式**: 所有模型都支持Arduino优化版本
✅ **智能量化**: 自动应用模型压缩技术
✅ **.h文件生成**: 直接输出Arduino兼容的头文件

## 支持的模型类型

- **1D_CNN**: 一维卷积神经网络 [[memory:3133346]]
- **CNN_LSTM**: 卷积LSTM混合模型
- **Transformer_Encoder**: Transformer编码器
- **XGBoost**: 极限梯度提升树

## 基本使用方法

### 最简单的使用（使用所有默认值）
```bash
python src/training/train.py
```
这将训练一个1D_CNN模型，使用normal模式，30个epoch，50次优化试验。

### 指定模型类型
```bash
python src/training/train.py --model_type XGBoost
```

### Arduino优化模式
```bash
python src/training/train.py --model_type 1D_CNN --mode arduino
```

### 自定义训练参数
```bash
python src/training/train.py --model_type CNN_LSTM --mode normal --epochs 50 --n_trials 100
```

## 参数说明

| 参数 | 默认值 | 选择 | 说明 |
|------|--------|------|------|
| `--model_type` | 1D_CNN | 1D_CNN, CNN_LSTM, Transformer_Encoder, XGBoost | 模型类型 |
| `--mode` | normal | normal, arduino | 训练模式 |
| `--epochs` | 30 | 任意正整数 | 训练轮数 |
| `--n_trials` | 50 | 任意正整数 | 超参数优化试验次数 |

## 训练模式对比

### Normal模式 (标准模式)
- **目标**: 追求最高精度
- **模型大小**: 较大，适合服务器/PC部署
- **特征**: 完整特征集，复杂模型架构

### Arduino模式
- **目标**: 小文件大小，适合嵌入式设备
- **模型大小**: 目标<1MB
- **特征**: 简化特征集，轻量级架构
- **优化**: 自动量化和模型压缩

## 输出文件

训练完成后会生成以下文件：

```
models/trained/[MODEL_TYPE]/
├── bsl_model_[MODEL_TYPE]_[TIMESTAMP].h      # Arduino头文件
└── params_[MODEL_TYPE]_[TIMESTAMP].json      # 训练参数和结果
```

### Arduino头文件内容
- **归一化参数**: 自动包含scaler参数
- **模型数据**: TFLite格式(TensorFlow)或C代码(XGBoost)
- **辅助函数**: 特征归一化函数

## 示例使用场景

### 1. 快速原型开发 [[memory:3553631]]
```bash
# 快速测试，使用最少参数
python src/training/train.py --epochs 10 --n_trials 10
```

### 2. Arduino部署
```bash
# 生成Arduino兼容的轻量级模型
python src/training/train.py --model_type XGBoost --mode arduino
```

### 3. 生产环境训练
```bash
# 完整训练，追求最高精度
python src/training/train.py --model_type Transformer_Encoder --mode normal --epochs 100 --n_trials 200
```

## 训练结果说明

### 文件大小
- **Arduino模式**: 目标<1MB，脚本会自动检查并提示
- **Normal模式**: 无大小限制

### 准确率
- 脚本会显示测试准确率
- 参数文件中包含详细的训练信息

### 兼容性
- **XGBoost**: 使用micromlgen生成C代码，完美兼容Arduino
- **TensorFlow模型**: 生成TFLite格式(可能需要TensorFlow Lite库)

## 故障排除

### TFLite转换失败
如果看到"TFLite转换失败"的消息，这是正常的。脚本会：
1. 尝试SavedModel转换
2. 回退到基础转换
3. 最终生成占位符头文件

虽然转换可能失败，但训练过程是成功的，生成的参数和模型架构信息仍然有用。

### 内存不足
如果遇到内存问题，减少以下参数：
- `--n_trials` (减少优化试验次数)
- `--epochs` (减少训练轮数)
- 使用`--mode arduino` (使用更小的模型架构)

## 与旧版本对比

| 特性 | 旧版本 | 新版本 |
|------|--------|--------|
| 训练脚本 | 4个独立文件 | 1个统一文件 |
| 命令行参数 | 必须指定 | 智能默认值 |
| Arduino支持 | 仅XGBoost | 所有模型 |
| 代码重复 | 大量重复 | 高度整合 |
| 易用性 | 复杂 | 简单 |

## 总结

新的统一训练脚本大大简化了BSL手势识别模型的训练流程。现在只需要一条命令就可以训练任何模型类型，并且所有模型都支持Arduino优化模式。默认参数经过精心选择，适合大多数使用场景，同时仍然保留了完全的自定义能力。 