# BSL手势识别系统 - 基于DIY传感器与端侧AI

![项目状态](https://img.shields.io/badge/status-development-yellow)
![Python版本](https://img.shields.io/badge/python-3.8+-blue)
![Arduino](https://img.shields.io/badge/arduino-nano%2033%20ble-green)

## 📖 项目简介

本项目实现了一个基于DIY传感器与端侧AI的英国手语（BSL）数字手势识别系统。系统能够识别BSL数字0-9以及静止状态，具有以下创新特点：

- 🔧 **DIY硬件**: 使用石墨-纸基柔性传感器阵列，成本低廉
- 🧠 **双层AI架构**: PC端Transformer高精度 + Arduino端1D-CNN实时部署
- ⚡ **端侧推理**: 基于TinyML技术，实现低延迟、低功耗识别
- 📡 **实时处理**: 滑动窗口机制，智能触发机制

## 🎯 功能特性

### 核心功能
- [x] 11分类手势识别（BSL数字0-9 + 静止状态）
- [x] 实时数据采集与预处理
- [x] 双层模型架构（Transformer + 1D-CNN）
- [x] PC端实时推理系统
- [x] Arduino端TinyML部署
- [x] 模型性能评估与比较

### 技术亮点
- **分段式传感器设计**: 消除信号串扰
- **EMA滤波**: 硬件级噪声抑制
- **动态手势采集**: 2秒时间窗口，50Hz采样
- **数据增强**: 抖动和缩放技术提升鲁棒性
- **模型量化**: INT8量化优化部署性能

## 🛠 技术栈

### 硬件
- **微控制器**: Arduino Nano 33 BLE Sense Rev2
- **传感器**: DIY石墨-纸基柔性传感器 × 5
- **电路**: 分压电路设计，独立信号通道

### 软件
- **开发语言**: Python 3.8+, Arduino C++
- **机器学习**: PyTorch, TensorFlow/Keras
- **端侧部署**: TensorFlow Lite for Microcontrollers
- **数据处理**: NumPy, Pandas, Scikit-learn
- **可视化**: Matplotlib, Seaborn

## 📁 项目结构

```
BSL-Gesture-Recognition/
├── arduino/                    # Arduino相关代码
│   ├── data_collection/        # 数据采集固件
│   ├── tinyml_inference/       # TinyML推理固件
│   └── libraries/              # 自定义库文件
├── src/                        # Python源代码
│   ├── data/                   # 数据处理模块
│   ├── models/                 # 模型定义
│   ├── training/               # 训练脚本
│   ├── inference/              # 推理系统
│   └── utils/                  # 工具函数
├── datasets/                   # 数据集存储
├── models/                     # 训练好的模型
├── docs/                       # 项目文档
├── configs/                    # 配置文件
├── tests/                      # 测试代码
├── requirements.txt            # Python依赖
└── README.md                   # 项目说明
```

## 🚀 快速开始

### 环境准备

1. **克隆项目**
```bash
git clone https://github.com/yourusername/BSL-Gesture-Recognition.git
cd BSL-Gesture-Recognition
```

2. **安装Python依赖**
```bash
pip install -r requirements.txt
```

3. **准备Arduino环境**
- 安装Arduino IDE
- 安装Arduino_TensorFlowLite库
- 配置Arduino Nano 33 BLE Sense Rev2

### 使用流程

1. **硬件制作**: 按照文档制作DIY传感器手套

2. **数据采集**: 运行数据采集脚本收集训练数据
   ```bash
   # 传感器数据可视化测试
   python run.py collect --visualize --duration 30
   
   # CSV格式数据采集
   python run.py collect --csv --duration 60 --filename test_data
   
   # 自动手势分类采集（推荐）
   python run.py collect --auto --user_id 001 --gestures 0 1 2 3 4
   ```

3. **数据预处理**: 清理和增强采集的数据
   ```bash
   python run.py preprocess --input_prefix raw_data
   ```

4. **模型训练**: 训练Transformer和1D-CNN模型
   ```bash
   python run.py train --data_prefix processed_data --model_type both
   ```

5. **性能评估**: 评估模型性能并进行对比
   ```bash
   python run.py evaluate --model_path models/trained/model.pth
   ```

6. **端侧部署**: 将1D-CNN部署到Arduino
   ```bash
   python run.py deploy --model_path models/trained/cnn_model.pth
   ```

## 📊 性能指标

| 模型 | 准确率 | F1分数 | 内存占用 | 推理延迟 |
|------|--------|--------|----------|----------|
| Transformer (PC) | TBD | TBD | ~50MB | ~10ms |
| 1D-CNN (Arduino) | TBD | TBD | ~32KB | ~5ms |

## 📚 文档

- [硬件制作指南](docs/hardware_guide.md)
- [数据采集说明](docs/data_collection.md)
- [模型训练教程](docs/model_training.md)
- [部署说明](docs/deployment.md)
- [API参考](docs/api_reference.md)

## 🤝 贡献

欢迎提交Issue和Pull Request来帮助改进项目！

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 👨‍💼 作者

**Lambert Yang** - 英国硕士学位论文项目

## 📝 版本历史

### v1.1 (2024-12-30) - 数据采集系统增强
- ✨ **原始数据采集**: 移除硬件EMA滤波，采集原始ADC值保证数据完整性
- ⏱️ **精确时间戳**: 使用相对时间戳（毫秒），精确记录数据采集间隔
- 🤖 **自动手势采集**: 新增按手势分类的自动采集程序，支持用户ID和交互式操作
- 📊 **增强CSV格式**: 包含元数据头部，便于数据回顾和分析
- 🔧 **命令行优化**: 新增 `--auto`, `--csv`, `--gesture_duration` 参数
- 🛠️ **兼容性改进**: int/float数据格式自动兼容，文件名字符清理

### v1.0 (2024-12-29) - 初始版本
- 🎯 完整的项目架构搭建
- 📡 基础数据采集和可视化功能
- 🧠 Transformer和1D-CNN模型框架
- ⚡ Arduino TinyML部署准备

---

*本项目是硕士学位论文的一部分，探索了低成本硬件与先进AI技术结合的可能性。* 