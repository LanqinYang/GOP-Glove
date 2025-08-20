# Arduino Real-time Gesture Recognition Test Guide

## 📋 概述

本文档指导如何运行Arduino上的实时手势识别测试，这是论文中重要的应用验证部分。

## 🎯 测试目标

验证ADANN_LightGBM模型在Arduino上的实时性能：
- **在线准确率**: 实时手势识别的准确性
- **平均延迟**: 从手势检测到分类结果的时间
- **抖动与误报率**: 系统稳定性和可靠性
- **姿态保持稳定性**: 长时间使用的稳定性

## 🔧 硬件要求

- **Arduino Nano 33 BLE Sense Rev2**
- **5个DIY柔性传感器** (连接到A0-A4)
- **USB连接线**
- **传感器手套** (已制作完成)

## 📁 文件结构

```
arduino/tinyml_inference/
├── 1D_CNN_inference/
│   └── Real_time_test/
│       ├── Real_time_test.ino          # 实时测试固件
│       └── bsl_model_1D_CNN.h          # 模型头文件
├── ADANN_LightGBM_inference/
│   └── Real_time_test/
│       ├── Real_time_test.ino          # 混合模型实时测试
│       └── bsl_model_ADANN_LightGBM.h  # 混合模型头文件
└── README.md
```

## 🚀 测试步骤

### 第一步：准备Arduino固件

1. **选择模型**: 决定测试哪个模型
   - `1D_CNN`: 基础CNN模型
   - `ADANN_LightGBM`: 核心混合模型 (推荐)

2. **上传固件**:
   ```bash
   # 打开Arduino IDE
   # 打开对应的Real_time_test.ino文件
   # 选择Arduino Nano 33 BLE Sense Rev2
   # 上传到Arduino
   ```

### 第二步：连接硬件

1. **传感器连接**:
   - Thumb → A0
   - Index → A1  
   - Middle → A2
   - Ring → A3
   - Pinky → A4

2. **验证连接**:
   - 打开串口监视器 (115200 baud)
   - 应该看到 "System initialized, waiting for gesture..."

### 第三步：运行实时测试

#### 选项A：快速测试 (推荐)
```bash
python src/evaluation/quick_arduino_test.py
```

#### 选项B：完整测试
```bash
python src/evaluation/arduino_realtime_test.py
```

## 📊 测试协议

### 快速测试 (5-10分钟)
- **测试手势**: 0-5 (Zero, One, Two, Three, Four, Five)
- **每个手势**: 30秒自由表演
- **指标**: 检测次数、准确率、置信度

### 完整测试 (30-45分钟)
- **测试手势**: 0-5
- **每个手势**: 3轮 × 2分钟 = 6分钟
- **总时长**: 6手势 × 6分钟 = 36分钟
- **详细指标**: 准确率、延迟、错误分析

## 🎯 测试手势说明

| 手势ID | 手势名称 | 描述 |
|--------|----------|------|
| 0 | Zero | 握拳 |
| 1 | One | 伸出食指 |
| 2 | Two | 伸出食指和中指 |
| 3 | Three | 伸出食指、中指、无名指 |
| 4 | Four | 伸出除拇指外的四指 |
| 5 | Five | 张开手掌 |

## 📈 预期结果

### 性能指标
- **在线准确率**: >80% (ADANN_LightGBM)
- **平均延迟**: <5ms
- **检测率**: 2-5次/分钟
- **置信度**: >70%

### 系统行为
- **自动检测**: 手势变化时自动触发
- **实时反馈**: 立即显示识别结果
- **冷却期**: 2秒冷却防止重复触发
- **错误处理**: 低置信度时显示"未识别"

## 🔍 故障排除

### 常见问题

1. **Arduino未连接**
   ```
   ❌ No Arduino found!
   ```
   **解决方案**: 检查USB连接，确认端口

2. **固件未上传**
   ```
   ❌ Arduino not responding with expected messages
   ```
   **解决方案**: 重新上传Real_time_test.ino

3. **传感器无响应**
   ```
   ⚠️ No gesture recognized
   ```
   **解决方案**: 检查传感器连接，调整阈值

4. **识别错误**
   ```
   ❌ Wrong gesture detected
   ```
   **解决方案**: 检查手势标准，调整模型

### 调试技巧

1. **启用调试输出**:
   ```cpp
   // 在Real_time_test.ino中取消注释
   Serial.println(difference);  // 查看传感器变化值
   ```

2. **调整阈值**:
   ```cpp
   const float GESTURE_TRIGGER_THRESHOLD = 50.0;  // 调整触发阈值
   const float STATIC_THRESHOLD = 30.0;           // 调整静态阈值
   ```

3. **检查传感器值**:
   ```cpp
   // 在setup()中添加
   for (int i = 0; i < NUM_SENSORS; i++) {
     Serial.print(analogRead(SENSOR_PINS[i]));
     Serial.print("\t");
   }
   Serial.println();
   ```

## 📊 结果分析

### 输出文件
测试完成后，结果保存在 `outputs/arduino_realtime_test/`:
- `realtime_test_results_*.json` - 详细结果
- `realtime_test_summary_*.csv` - 汇总表格
- `realtime_performance_plots_*.png` - 性能图表

### 关键指标
1. **整体准确率**: 所有检测的正确率
2. **手势准确率**: 每个手势的单独准确率
3. **检测率**: 每分钟检测次数
4. **错误类型**: 未识别、通信错误等
5. **置信度分布**: 检测置信度的统计

## 🎯 论文应用

### 实时演示结果
这些测试结果将用于论文的"实时演示"部分，展示：
- ADANN_LightGBM在真实环境中的性能
- 与基线模型的对比
- 系统稳定性和可靠性
- 实际应用可行性

### 数据收集
建议收集以下数据：
- 3个用户的测试结果
- 每个用户3轮测试
- 不同环境条件下的表现
- 长时间使用的稳定性

## 📝 注意事项

1. **环境一致性**: 保持测试环境稳定
2. **手势标准化**: 严格按照手势定义执行
3. **时间控制**: 按测试协议控制时间
4. **数据记录**: 详细记录所有异常情况
5. **重复验证**: 多次测试确保结果可靠性

## 🔗 相关文件

- `src/evaluation/arduino_realtime_test.py` - 完整测试脚本
- `src/evaluation/quick_arduino_test.py` - 快速测试脚本
- `arduino/tinyml_inference/*/Real_time_test/Real_time_test.ino` - 测试固件
- `docs/Dissertation_Structure.md` - 论文结构要求
