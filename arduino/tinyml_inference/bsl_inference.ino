/*
  BSL手势识别系统 - TinyML推理固件
  
  功能：
  - 实时传感器数据采集
  - 滑动窗口数据缓存
  - TensorFlow Lite模型推理
  - 手势识别结果输出
  - LED指示和串口通信
  
  硬件要求：
  - Arduino Nano 33 BLE Sense Rev2
  - 5个DIY柔性传感器
  - LED指示灯（可选）
  
  作者: Lambert Yang
  版本: 1.0
*/

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// 包含模型数据（需要先生成）
// #include "bsl_model_data.h"

// 配置参数
const int SENSOR_PINS[] = {A0, A1, A2, A3, A4};
const int NUM_SENSORS = 5;
const int SEQUENCE_LENGTH = 100;
const int SAMPLE_RATE = 50;
const int SAMPLE_INTERVAL = 1000 / SAMPLE_RATE;
const float EMA_ALPHA = 0.1;

// LED指示
const int LED_PIN = LED_BUILTIN;
const int RESULT_LED_PINS[] = {2, 3, 4, 5, 6};  // 可选的结果指示LED

// 推理参数
const float CONFIDENCE_THRESHOLD = 0.8;
const int MIN_STABLE_PREDICTIONS = 3;
const int INFERENCE_INTERVAL = 500;  // ms

// 手势类别名称
const char* GESTURE_NAMES[] = {
  "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"
};

// 全局变量
float sensorBuffer[SEQUENCE_LENGTH][NUM_SENSORS];
int bufferIndex = 0;
bool bufferFull = false;
unsigned long lastSampleTime = 0;
unsigned long lastInferenceTime = 0;

// EMA滤波器状态
float emaValues[NUM_SENSORS];
bool emaInitialized = false;

// TensorFlow Lite相关
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// 为TensorFlow Lite分配内存 (调整大小根据模型需要)
constexpr int kTensorArenaSize = 32 * 1024;  // 32KB
uint8_t tensor_arena[kTensorArenaSize];

// 预测稳定性跟踪
int lastPredictions[MIN_STABLE_PREDICTIONS];
int predictionIndex = 0;
int stablePrediction = -1;

void setup() {
  // 初始化串口
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }
  
  Serial.println("BSL手势识别系统 - TinyML推理模式");
  
  // 初始化传感器引脚
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(SENSOR_PINS[i], INPUT);
    emaValues[i] = 0.0;
  }
  
  // 初始化LED
  pinMode(LED_PIN, OUTPUT);
  for (int i = 0; i < 5; i++) {
    pinMode(RESULT_LED_PINS[i], OUTPUT);
    digitalWrite(RESULT_LED_PINS[i], LOW);
  }
  
  // 初始化TensorFlow Lite
  setupTensorFlowLite();
  
  // 初始化预测历史
  for (int i = 0; i < MIN_STABLE_PREDICTIONS; i++) {
    lastPredictions[i] = -1;
  }
  
  Serial.println("系统初始化完成，开始手势识别...");
  Serial.println("手势类别: Zero, One, Two, Three, Four, Five, Six, Seven, Eight, Nine");
  
  lastSampleTime = millis();
  lastInferenceTime = millis();
}

void loop() {
  unsigned long currentTime = millis();
  
  // 数据采集
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = currentTime;
    collectSensorData();
  }
  
  // 推理
  if (bufferFull && (currentTime - lastInferenceTime >= INFERENCE_INTERVAL)) {
    lastInferenceTime = currentTime;
    performInference();
  }
  
  // 处理串口命令
  handleSerialCommands();
}

void setupTensorFlowLite() {
  Serial.println("初始化TensorFlow Lite...");
  
  // 设置错误报告器
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  /* 注意：这里需要在生成模型数据后取消注释
  // 加载模型
  model = tflite::GetModel(bsl_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                          "Model version %d not equal to supported version %d.",
                          model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  
  // 创建操作解析器
  static tflite::AllOpsResolver resolver;
  
  // 创建解释器
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  
  // 分配内存
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  
  // 获取输入输出张量
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.print("输入维度: ");
  Serial.print(input->dims->size);
  Serial.print(" [");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print(", ");
  }
  Serial.println("]");
  
  Serial.print("输出维度: ");
  Serial.print(output->dims->size);
  Serial.print(" [");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1) Serial.print(", ");
  }
  Serial.println("]");
  */
  
  Serial.println("TensorFlow Lite初始化完成");
  Serial.println("注意: 需要包含实际的模型数据文件");
}

void collectSensorData() {
  // 读取原始传感器值
  int rawValues[NUM_SENSORS];
  for (int i = 0; i < NUM_SENSORS; i++) {
    rawValues[i] = analogRead(SENSOR_PINS[i]);
  }
  
  // 应用EMA滤波
  if (!emaInitialized) {
    for (int i = 0; i < NUM_SENSORS; i++) {
      emaValues[i] = rawValues[i];
    }
    emaInitialized = true;
  } else {
    for (int i = 0; i < NUM_SENSORS; i++) {
      emaValues[i] = EMA_ALPHA * rawValues[i] + (1.0 - EMA_ALPHA) * emaValues[i];
    }
  }
  
  // 存储到缓冲区（归一化到0-1范围）
  for (int i = 0; i < NUM_SENSORS; i++) {
    sensorBuffer[bufferIndex][i] = emaValues[i] / 1023.0;  // Arduino ADC是10位
  }
  
  bufferIndex++;
  if (bufferIndex >= SEQUENCE_LENGTH) {
    bufferIndex = 0;
    bufferFull = true;
  }
}

void performInference() {
  /* 注意：这里需要在包含模型数据后取消注释
  if (!model || !interpreter) {
    return;
  }
  
  // 准备输入数据
  int startIdx = bufferFull ? bufferIndex : 0;
  int inputIdx = 0;
  
  for (int t = 0; t < SEQUENCE_LENGTH; t++) {
    int dataIdx = (startIdx + t) % SEQUENCE_LENGTH;
    for (int s = 0; s < NUM_SENSORS; s++) {
      input->data.f[inputIdx++] = sensorBuffer[dataIdx][s];
    }
  }
  
  // 执行推理
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }
  
  // 获取预测结果
  float maxProb = 0.0;
  int predictedClass = 0;
  
  for (int i = 0; i < 11; i++) {  // 11个类别
    float prob = output->data.f[i];
    if (prob > maxProb) {
      maxProb = prob;
      predictedClass = i;
    }
  }
  
  // 检查置信度
  if (maxProb >= CONFIDENCE_THRESHOLD) {
    updatePredictionHistory(predictedClass);
    int stable = getStablePrediction();
    
    if (stable != -1 && stable != stablePrediction) {
      stablePrediction = stable;
      outputGestureResult(stable, maxProb);
      updateLEDIndicators(stable);
    }
  }
  */
  
  // 临时模拟推理（用于测试硬件）
  simulateInference();
}

void simulateInference() {
  // 简单的基于传感器值的模拟分类
  float avgValues[NUM_SENSORS];
  float totalActivity = 0.0;
  
  // 计算当前窗口的平均值
  for (int s = 0; s < NUM_SENSORS; s++) {
    avgValues[s] = 0.0;
    for (int t = 0; t < SEQUENCE_LENGTH; t++) {
      avgValues[s] += sensorBuffer[t][s];
    }
    avgValues[s] /= SEQUENCE_LENGTH;
    totalActivity += avgValues[s];
  }
  
  // 简单分类逻辑
  int predictedClass = 0;  // 默认为Rest
  float confidence = 0.5;
  
  if (totalActivity > 2.0) {  // 有明显活动
    // 基于最活跃的传感器进行简单分类
    int maxSensor = 0;
    float maxValue = avgValues[0];
    for (int i = 1; i < NUM_SENSORS; i++) {
      if (avgValues[i] > maxValue) {
        maxValue = avgValues[i];
        maxSensor = i;
      }
    }
    
    predictedClass = maxSensor + 1;  // 传感器0->手势1, 传感器1->手势2, 等等
    confidence = min(0.95, totalActivity / 3.0);
  }
  
  if (confidence >= CONFIDENCE_THRESHOLD) {
    updatePredictionHistory(predictedClass);
    int stable = getStablePrediction();
    
    if (stable != -1 && stable != stablePrediction) {
      stablePrediction = stable;
      outputGestureResult(stable, confidence);
      updateLEDIndicators(stable);
    }
  }
}

void updatePredictionHistory(int prediction) {
  lastPredictions[predictionIndex] = prediction;
  predictionIndex = (predictionIndex + 1) % MIN_STABLE_PREDICTIONS;
}

int getStablePrediction() {
  // 检查最近的预测是否一致
  int firstPrediction = lastPredictions[0];
  if (firstPrediction == -1) return -1;
  
  for (int i = 1; i < MIN_STABLE_PREDICTIONS; i++) {
    if (lastPredictions[i] != firstPrediction) {
      return -1;  // 不一致
    }
  }
  
  return firstPrediction;  // 一致的预测
}

void outputGestureResult(int gestureClass, float confidence) {
  Serial.print("手势识别: ");
  Serial.print(GESTURE_NAMES[gestureClass]);
  Serial.print(" (置信度: ");
  Serial.print(confidence * 100, 1);
  Serial.println("%)");
  
  // 可选：发送JSON格式的结果
  Serial.print("{\"gesture\":\"");
  Serial.print(GESTURE_NAMES[gestureClass]);
  Serial.print("\",\"class\":");
  Serial.print(gestureClass);
  Serial.print(",\"confidence\":");
  Serial.print(confidence, 3);
  Serial.print(",\"timestamp\":");
  Serial.print(millis());
  Serial.println("}");
}

void updateLEDIndicators(int gestureClass) {
  // 清除所有LED
  for (int i = 0; i < 5; i++) {
    digitalWrite(RESULT_LED_PINS[i], LOW);
  }
  
  // 点亮对应的LED（如果在范围内）
  if (gestureClass > 0 && gestureClass <= 5) {
    digitalWrite(RESULT_LED_PINS[gestureClass - 1], HIGH);
  }
  
  // 内置LED闪烁表示识别到手势
  digitalWrite(LED_PIN, HIGH);
  delay(100);
  digitalWrite(LED_PIN, LOW);
}

void handleSerialCommands() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "status") {
      printStatus();
    } else if (command == "reset") {
      resetSystem();
    } else if (command == "buffer") {
      printBuffer();
    } else if (command == "sensors") {
      printSensorValues();
    } else if (command == "help") {
      printHelp();
    }
  }
}

void printStatus() {
  Serial.println("=== 系统状态 ===");
  Serial.print("缓冲区状态: ");
  Serial.print(bufferFull ? "满" : "填充中");
  Serial.print(" (");
  Serial.print(bufferIndex);
  Serial.print("/");
  Serial.print(SEQUENCE_LENGTH);
  Serial.println(")");
  
  Serial.print("当前预测: ");
  if (stablePrediction != -1) {
    Serial.println(GESTURE_NAMES[stablePrediction]);
  } else {
    Serial.println("无");
  }
  
  Serial.print("内存使用: ");
  Serial.print(kTensorArenaSize);
  Serial.println(" bytes");
  
  Serial.print("运行时间: ");
  Serial.print(millis() / 1000);
  Serial.println(" 秒");
}

void resetSystem() {
  Serial.println("重置系统...");
  
  // 重置缓冲区
  bufferIndex = 0;
  bufferFull = false;
  
  // 重置EMA
  emaInitialized = false;
  
  // 重置预测历史
  for (int i = 0; i < MIN_STABLE_PREDICTIONS; i++) {
    lastPredictions[i] = -1;
  }
  stablePrediction = -1;
  
  // 关闭所有LED
  for (int i = 0; i < 5; i++) {
    digitalWrite(RESULT_LED_PINS[i], LOW);
  }
  
  Serial.println("系统已重置");
}

void printBuffer() {
  Serial.println("=== 传感器缓冲区 ===");
  Serial.println("时间步, S1, S2, S3, S4, S5");
  
  int startIdx = bufferFull ? bufferIndex : 0;
  for (int t = 0; t < min(10, SEQUENCE_LENGTH); t++) {  // 只打印前10行
    int dataIdx = (startIdx + t) % SEQUENCE_LENGTH;
    Serial.print(t);
    for (int s = 0; s < NUM_SENSORS; s++) {
      Serial.print(", ");
      Serial.print(sensorBuffer[dataIdx][s], 3);
    }
    Serial.println();
  }
  
  if (SEQUENCE_LENGTH > 10) {
    Serial.println("... (显示前10行)");
  }
}

void printSensorValues() {
  Serial.println("=== 实时传感器值 ===");
  Serial.print("原始值: ");
  for (int i = 0; i < NUM_SENSORS; i++) {
    Serial.print(analogRead(SENSOR_PINS[i]));
    if (i < NUM_SENSORS - 1) Serial.print(", ");
  }
  Serial.println();
  
  Serial.print("EMA值: ");
  for (int i = 0; i < NUM_SENSORS; i++) {
    Serial.print(emaValues[i], 2);
    if (i < NUM_SENSORS - 1) Serial.print(", ");
  }
  Serial.println();
}

void printHelp() {
  Serial.println("=== 可用命令 ===");
  Serial.println("status  - 显示系统状态");
  Serial.println("reset   - 重置系统");
  Serial.println("buffer  - 显示数据缓冲区");
  Serial.println("sensors - 显示实时传感器值");
  Serial.println("help    - 显示此帮助信息");
} 