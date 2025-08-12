# BSL手势识别TinyML推理固件（Arduino版本）

## 🎯 概述

本目录包含4个不同机器学习模型的Arduino推理固件，全部采用**ArduTFLite.h库**来大幅简化TensorFlow Lite模型的部署和使用。

## ✨ 核心优势

### 🚀 使用ArduTFLite.h库的优势

- **API大幅简化**：从复杂的TensorFlow Lite C++ API简化为直观的Arduino风格API
- **内存管理自动化**：无需手动管理tensor arena和解释器
- **代码量减少70%**：告别繁琐的tensor操作和内存分配
- **更好的稳定性**：减少内存泄漏和初始化错误
- **Arduino友好**：完全符合Arduino编程习惯

### 📊 API对比

**传统TensorFlow Lite方式**：
```cpp
// 复杂的初始化
tflite::MicroInterpreter* interpreter;
tflite::MicroMutableOpResolver<6> resolver;
constexpr int tensor_arena_size = 28*1024;
alignas(16) uint8_t tensor_arena[tensor_arena_size];

// 复杂的推理
TfLiteTensor* input = interpreter->input(0);
input->data.f[i] = normalized_value;
interpreter->Invoke();
TfLiteTensor* output = interpreter->output(0);
float result = output->data.f[i];
```

**ArduTFLite.h方式**：
```cpp
// 简单的初始化
ArduTFLite tfModel;
tfModel.begin(model);

// 简单的推理
tfModel.input(i) = normalized_value;
tfModel.run();
float result = tfModel.output(i);
```

## 🔧 硬件要求

- **Arduino Nano 33 BLE Sense Rev2** 
- **5个柔性传感器**（连接到A0-A4引脚）
- **可选LED指示灯**（用于结果显示）

## 🧠 支持的模型

### 1. 1D_CNN模型 ⚡
- **文件**: `1D_CNN_inference/`
- **内存占用**: ~28KB
- **推理速度**: 最快（~8-15ms）
- **适用场景**: 简单手势识别，低延迟要求
- **特点**: 轻量级一维卷积，适合实时应用

### 2. Transformer_Encoder模型 🎯
- **文件**: `Transformer_Encoder_inference/`
- **内存占用**: ~48KB
- **推理速度**: 最慢（~40-60ms）
- **适用场景**: 最高准确率，复杂模式识别
- **特点**: 基于注意力机制，处理长序列依赖

### 4. XGBoost_Arduino模型 🚀
- **文件**: `XGBoost_Arduino_inference/`
- **内存占用**: 可变
- **推理速度**: 极快（~2-5ms）
- **适用场景**: 超低延迟，基于特征的分类
- **特点**: 决策树集成，无需TensorFlow

## 📁 项目结构

```
arduino/tinyml_inference/
├── 1D_CNN_inference/              # 轻量级CNN模型
│   ├── 1D_CNN_inference.ino       # 主程序
│   └── bsl_model_1D_CNN.h         # 模型权重
├── Transformer_Encoder_inference/  # Transformer模型
│   ├── Transformer_Encoder_inference.ino
│   └── bsl_model_Transformer_Encoder.h
├── XGBoost_Arduino_inference/      # XGBoost决策树模型
│   ├── XGBoost_Arduino_inference.ino
│   └── bsl_model_XGBoost_Arduino.h
└── README.md                      # 本文件
```

## 🚀 技术特性

### 核心功能
- **使用ArduTFLite.h库简化API**
- **实时传感器数据采集**（50Hz采样率）
- **滑动窗口数据处理**（100时间步序列）
- **自动数据归一化**（使用训练时的scaler参数）
- **置信度阈值控制**
- **LED指示和串口输出**

### 数据处理管道
1. **传感器读取** → 模拟读取A0-A4引脚
2. **EMA滤波** → 平滑传感器噪声
3. **数据归一化** → 使用训练时的均值和标准差
4. **滑动窗口** → 维护100个时间步的历史数据
5. **模型推理** → 使用ArduTFLite.h执行推理
6. **结果后处理** → 置信度检查和稳定性过滤

## 📋 使用说明

### 1. 安装依赖
```bash
# 在Arduino IDE中安装以下库：
# - ArduTFLite (通过库管理器)
```

### 2. 硬件连接
```
Arduino Nano 33 BLE Sense Rev2:
- A0 → 柔性传感器1
- A1 → 柔性传感器2  
- A2 → 柔性传感器3
- A3 → 柔性传感器4
- A4 → 柔性传感器5
- D2-D6 → LED指示灯（可选）
```

### 3. 上传代码
1. 选择想要使用的模型（1D_CNN推荐开始）
2. 在Arduino IDE中打开对应的.ino文件
3. 选择板卡：Arduino Nano 33 BLE
4. 选择正确的串口
5. 点击上传

### 4. 串口监控
```
波特率：115200
输出格式：
- 实时概率显示
- JSON格式结果
- 系统状态信息
```

## 🎮 串口命令

所有模型都支持以下串口命令：

| 命令 | 功能 |
|------|------|
| `status` | 显示系统状态和缓冲区信息 |
| `reset` | 重置系统和清除缓冲区 |
| `buffer` | 显示传感器数据缓冲区 |
| `sensors` | 显示实时传感器值 |
| `help` | 显示可用命令列表 |

## 📊 性能对比

| 模型 | 内存占用 | 推理时间 | 准确率 | 适用场景 |
|------|----------|----------|--------|----------|
| 1D_CNN | 28KB | 8-15ms | 85-90% | 实时应用 |
| Transformer | 48KB | 40-60ms | 95-98% | 最高精度 |
| XGBoost | 可变 | 2-5ms | 80-85% | 超低延迟 |

## 🔧 自定义配置

### 调整采样参数
```cpp
const int SAMPLE_RATE = 50;           // 采样频率(Hz)
const int SEQUENCE_LENGTH = 100;      // 序列长度
const float CONFIDENCE_THRESHOLD = 0.8; // 置信度阈值
```

### 修改滤波参数
```cpp
const float EMA_ALPHA = 0.1;         // EMA滤波系数
```

## 🆚 模型选择指南

### 选择1D_CNN如果你需要：
- ✅ 最快的推理速度
- ✅ 最小的内存占用
- ✅ 简单的手势识别
- ✅ 实时响应

### 选择Transformer如果你需要：
- ✅ 最高的识别精度
- ✅ 复杂的注意力机制
- ✅ 长期时序依赖建模
- ✅ 不介意较慢的推理速度

### 选择XGBoost如果你需要：
- ✅ 超快的推理速度
- ✅ 无需TensorFlow依赖
- ✅ 基于特征的分类
- ✅ 简单的部署

## 🔍 故障排除

### 常见问题

**Q: 编译错误 "ArduTFLite.h not found"**
A: 请在Arduino IDE库管理器中安装ArduTFLite库

**Q: 模型初始化失败**
A: 检查模型文件是否正确包含，确认Arduino有足够内存

**Q: 传感器读数异常**
A: 检查传感器连接，确认A0-A4引脚连接正确

**Q: 识别准确率低**
A: 确保传感器校准正确，尝试调整置信度阈值

## 📈 未来改进

- [ ] 支持更多传感器类型
- [ ] 添加在线学习功能
- [ ] 实现模型自动选择
- [ ] 增加蓝牙数据传输
- [ ] 添加手势录制功能

## 👥 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证 - 详见LICENSE文件

---

**作者**: Lambert Yang  
**版本**: 2.0 (ArduTFLite.h版本)  
**更新日期**: 2024年

**注意**: 使用ArduTFLite.h库大大简化了代码复杂度，提高了可维护性和稳定性。推荐所有TensorFlow Lite Arduino项目使用此库！ 