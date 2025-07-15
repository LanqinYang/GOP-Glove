/*
  BSL手势识别系统 - 1D_CNN模型智能推理固件
  
  功能：
  - 实时传感器数据采集和滑动窗口缓存
  - 智能手势检测：静态预检测+动态完整推理
  - 1D_CNN TensorFlow Lite模型推理
  - LED指示和串口通信
  
  硬件要求：
  - Arduino Nano 33 BLE Sense Rev2
  - 5个DIY柔性传感器连接到A0-A4
  - LED指示灯（可选）
  
  智能推理机制：
  - 持续低成本静态检测（检测是否有手势发生）
  - 检测到手势变化时自动触发完整模型推理
  - 节省计算资源，提高响应速度
  
  模型类型: 1D_CNN
  作者: Lambert Yang
  版本: 3.0 (智能推理版本)
*/

#include <ArduTFLite.h>
#include "bsl_model_1D_CNN.h"

// ===== 硬件配置 =====
const int SENSOR_PINS[] = {A0, A1, A2, A3, A4};
const int NUM_SENSORS = 5;
const int SEQUENCE_LENGTH = 100;
const int SAMPLE_RATE = 50;  // Hz
const int SAMPLE_INTERVAL = 1000 / SAMPLE_RATE;  // 20ms

// ===== LED指示配置 =====
const int LED_PIN = LED_BUILTIN;
const int RESULT_LED_PINS[] = {2, 3, 4, 5, 6};  // 可选外部LED

// ===== 智能推理配置 =====
const float STATIC_THRESHOLD = 30.0;        // 静态检测阈值
const float GESTURE_TRIGGER_THRESHOLD = 50.0; // 手势触发阈值
const int STATIC_CHECK_SAMPLES = 10;        // 静态检测样本数
const int MIN_GESTURE_DURATION = 30;        // 最小手势持续时间（样本数）
const float CONFIDENCE_THRESHOLD = 0.7;     // 置信度阈值
const int STABLE_PREDICTION_COUNT = 3;      // 稳定预测需要的次数

// ===== TensorFlow Lite配置 =====
constexpr int tensorArenaSize = 28 * 1024;  // 1D_CNN需要约28KB
alignas(16) byte tensorArena[tensorArenaSize];

// ===== 数据缓存和状态 =====
float sensorBuffer[SEQUENCE_LENGTH][NUM_SENSORS];
int bufferIndex = 0;
bool bufferFilled = false;

// 智能检测状态
enum DetectionState {
  STATE_STATIC,        // 静态状态
  STATE_GESTURE_START, // 手势开始
  STATE_GESTURE_ACTIVE,// 手势进行中
  STATE_COOLDOWN       // 冷却期
};

DetectionState currentState = STATE_STATIC;
unsigned long stateStartTime = 0;
int gestureFrameCount = 0;
float recentVariance[NUM_SENSORS] = {0};

// 预测结果管理
int lastPredictions[STABLE_PREDICTION_COUNT];
int predictionIndex = 0;
int currentGestureClass = -1;
float currentConfidence = 0.0;

// 时间管理
unsigned long lastSampleTime = 0;
unsigned long lastInferenceTime = 0;
unsigned long lastStatusTime = 0;

// 手势类别名称
const char* GESTURE_NAMES[] = {
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "静态"
};
const int NUM_GESTURES = 11;

// ===== 工具函数（最底层，被其他函数调用）=====

void initializeBuffer() {
  for (int i = 0; i < SEQUENCE_LENGTH; i++) {
    for (int j = 0; j < NUM_SENSORS; j++) {
      sensorBuffer[i][j] = 0.0;
    }
  }
}

void clearLEDIndicators() {
  for (int i = 0; i < 5; i++) {
    digitalWrite(RESULT_LED_PINS[i], LOW);
  }
}

void updateLEDIndicators(int gestureClass) {
  clearLEDIndicators();
  
  if (gestureClass >= 0 && gestureClass < 5) {
    digitalWrite(RESULT_LED_PINS[gestureClass], HIGH);
  }
  
  if (gestureClass == 10) {  // 静态
    digitalWrite(LED_PIN, LOW);
  } else {
    digitalWrite(LED_PIN, HIGH);
  }
}

void updatePredictionHistory(int prediction) {
  lastPredictions[predictionIndex] = prediction;
  predictionIndex = (predictionIndex + 1) % STABLE_PREDICTION_COUNT;
}

int getStablePrediction() {
  // 检查是否有稳定的预测（多数相同）
  for (int target = 0; target < NUM_GESTURES; target++) {
    int count = 0;
    for (int i = 0; i < STABLE_PREDICTION_COUNT; i++) {
      if (lastPredictions[i] == target) count++;
    }
    if (count >= (STABLE_PREDICTION_COUNT + 1) / 2) {  // 过半数
      return target;
    }
  }
  return -1;  // 无稳定预测
}

void outputGestureResult(int gestureClass, float confidence) {
  static unsigned long lastOutputTime = 0;
  unsigned long currentTime = millis();
  
  // 限制输出频率
  if (currentTime - lastOutputTime < 500) return;
  
  Serial.print("🤲 识别结果: ");
  Serial.print(GESTURE_NAMES[gestureClass]);
  Serial.print(" (置信度: ");
  Serial.print(confidence * 100, 1);
  Serial.println("%)");
  
  lastOutputTime = currentTime;
}

void calculateRecentVariance() {
  // 计算最近STATIC_CHECK_SAMPLES个样本的方差
  for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
    float sum = 0.0;
    float sumSquared = 0.0;
    
    for (int i = 0; i < STATIC_CHECK_SAMPLES; i++) {
      int idx = (bufferIndex - 1 - i + SEQUENCE_LENGTH) % SEQUENCE_LENGTH;
      float value = sensorBuffer[idx][sensor];
      sum += value;
      sumSquared += value * value;
    }
    
    float mean = sum / STATIC_CHECK_SAMPLES;
    recentVariance[sensor] = (sumSquared / STATIC_CHECK_SAMPLES) - (mean * mean);
  }
}

// ===== 状态管理函数 =====

void checkForGestureStart() {
  float totalVariance = 0.0;
  for (int i = 0; i < NUM_SENSORS; i++) {
    totalVariance += recentVariance[i];
  }
  
  if (totalVariance > GESTURE_TRIGGER_THRESHOLD) {
    currentState = STATE_GESTURE_START;
    stateStartTime = millis();
    gestureFrameCount = 0;
    Serial.println("🎯 检测到可能的手势开始");
    digitalWrite(LED_PIN, HIGH);
  }
}

void confirmGestureStart() {
  gestureFrameCount++;
  
  if (gestureFrameCount >= MIN_GESTURE_DURATION) {
    currentState = STATE_GESTURE_ACTIVE;
    Serial.println("✅ 手势确认，开始完整推理");
    lastInferenceTime = millis();
  } else if (millis() - stateStartTime > 1000) {  // 1秒超时
    currentState = STATE_STATIC;
    Serial.println("❌ 手势确认超时，返回静态状态");
    digitalWrite(LED_PIN, LOW);
  }
}

void checkGestureEnd() {
  float totalVariance = 0.0;
  for (int i = 0; i < NUM_SENSORS; i++) {
    totalVariance += recentVariance[i];
  }
  
  if (totalVariance < STATIC_THRESHOLD) {
    currentState = STATE_COOLDOWN;
    stateStartTime = millis();
    Serial.println("📉 手势结束，进入冷却期");
  }
}

void checkCooldownEnd() {
  if (millis() - stateStartTime > 1000) {  // 1秒冷却期
    currentState = STATE_STATIC;
    currentGestureClass = -1;
    currentConfidence = 0.0;
    Serial.println("🔄 返回静态监测状态");
    digitalWrite(LED_PIN, LOW);
    clearLEDIndicators();
  }
}

// ===== 推理函数 =====

void performCompleteInference() {
  unsigned long currentTime = millis();
  
  // 限制推理频率（避免过于频繁）
  if (currentTime - lastInferenceTime < 200) return;  // 最多5Hz
  
  // 准备输入数据
  int inputIdx = 0;
  int startIdx = (bufferIndex - SEQUENCE_LENGTH + SEQUENCE_LENGTH) % SEQUENCE_LENGTH;
  
  for (int t = 0; t < SEQUENCE_LENGTH; t++) {
    int dataIdx = (startIdx + t) % SEQUENCE_LENGTH;
    for (int s = 0; s < NUM_SENSORS; s++) {
      // 应用归一化：(x - mean) / scale
      float normalized = (sensorBuffer[dataIdx][s] - scaler_mean[s]) / scaler_scale[s];
      modelSetInput(normalized, inputIdx++);
    }
  }
  
  // 执行推理
  if (!modelRunInference()) {
    Serial.println("❌ 1D_CNN推理失败!");
    return;
  }
  
  // 处理推理结果
  float maxProb = 0.0;
  int predictedClass = 0;
  
  for (int i = 0; i < NUM_GESTURES; i++) {
    float prob = modelGetOutput(i);
    if (prob > maxProb) {
      maxProb = prob;
      predictedClass = i;
    }
  }
  
  // 更新预测历史
  updatePredictionHistory(predictedClass);
  
  // 检查稳定预测
  int stableClass = getStablePrediction();
  if (stableClass >= 0 && maxProb > CONFIDENCE_THRESHOLD) {
    currentGestureClass = stableClass;
    currentConfidence = maxProb;
    
    outputGestureResult(stableClass, maxProb);
    updateLEDIndicators(stableClass);
  }
  
  lastInferenceTime = currentTime;
}

void performIntelligentInference() {
  if (!bufferFilled) return;
  
  // 计算最近样本的方差（用于手势检测）
  calculateRecentVariance();
  
  // 根据当前状态执行相应的检测逻辑
  switch (currentState) {
    case STATE_STATIC:
      checkForGestureStart();
      break;
      
    case STATE_GESTURE_START:
      confirmGestureStart();
      break;
      
    case STATE_GESTURE_ACTIVE:
      performCompleteInference();
      checkGestureEnd();
      break;
      
    case STATE_COOLDOWN:
      checkCooldownEnd();
      break;
  }
}

// ===== 数据采集 =====

void collectSensorData() {
  // 读取当前传感器值并应用EMA滤波
  for (int i = 0; i < NUM_SENSORS; i++) {
    float rawValue = analogRead(SENSOR_PINS[i]);
    
    // 简单的EMA滤波
    if (bufferFilled) {
      int prevIdx = (bufferIndex - 1 + SEQUENCE_LENGTH) % SEQUENCE_LENGTH;
      float alpha = 0.3;  // EMA系数
      sensorBuffer[bufferIndex][i] = alpha * rawValue + (1 - alpha) * sensorBuffer[prevIdx][i];
    } else {
      sensorBuffer[bufferIndex][i] = rawValue;
    }
  }
  
  // 更新缓冲区索引
  bufferIndex = (bufferIndex + 1) % SEQUENCE_LENGTH;
  if (!bufferFilled && bufferIndex == 0) {
    bufferFilled = true;
    Serial.println("📊 数据缓冲区已填充完成");
  }
}

// ===== 串口和显示函数 =====

void printSystemStatus() {
  Serial.print("📊 状态: ");
  switch (currentState) {
    case STATE_STATIC:
      Serial.print("静态监测");
      break;
    case STATE_GESTURE_START:
      Serial.print("手势检测中");
      break;
    case STATE_GESTURE_ACTIVE:
      Serial.print("手势识别中");
      break;
    case STATE_COOLDOWN:
      Serial.print("冷却期");
      break;
  }
  
  if (currentGestureClass >= 0) {
    Serial.print(" | 当前手势: ");
    Serial.print(GESTURE_NAMES[currentGestureClass]);
    Serial.print(" (");
    Serial.print(currentConfidence * 100, 1);
    Serial.print("%)");
  }
  
  Serial.println();
}

void printDetailedStatus() {
  Serial.println("=== 1D_CNN系统详细状态 ===");
  Serial.print("模型状态: 已加载 | 内存使用: ");
  Serial.print(tensorArenaSize / 1024);
  Serial.println("KB");
  
  Serial.print("检测状态: ");
  printSystemStatus();
  
  Serial.print("传感器方差: ");
  for (int i = 0; i < NUM_SENSORS; i++) {
    Serial.print(recentVariance[i], 1);
    Serial.print(" ");
  }
  Serial.println();
  
  Serial.print("缓冲区: ");
  Serial.print(bufferFilled ? "已填充" : "填充中");
  Serial.print(" | 索引: ");
  Serial.println(bufferIndex);
}

void resetSystem() {
  Serial.println("🔄 重置系统...");
  
  currentState = STATE_STATIC;
  currentGestureClass = -1;
  currentConfidence = 0.0;
  bufferFilled = false;
  bufferIndex = 0;
  
  for (int i = 0; i < STABLE_PREDICTION_COUNT; i++) {
    lastPredictions[i] = -1;
  }
  
  clearLEDIndicators();
  digitalWrite(LED_PIN, LOW);
  
  Serial.println("✅ 系统重置完成");
}

void printBufferInfo() {
  Serial.println("=== 数据缓冲区信息 ===");
  Serial.print("大小: ");
  Serial.print(SEQUENCE_LENGTH);
  Serial.print(" x ");
  Serial.print(NUM_SENSORS);
  Serial.println(" 样本");
  
  Serial.print("状态: ");
  Serial.println(bufferFilled ? "已填充" : "填充中");
  
  Serial.print("当前索引: ");
  Serial.println(bufferIndex);
  
  Serial.println("最近10个样本:");
  for (int i = 0; i < 10 && i < SEQUENCE_LENGTH; i++) {
    int idx = (bufferIndex - 1 - i + SEQUENCE_LENGTH) % SEQUENCE_LENGTH;
    Serial.print("样本");
    Serial.print(i);
    Serial.print(": ");
    for (int s = 0; s < NUM_SENSORS; s++) {
      Serial.print(sensorBuffer[idx][s], 1);
      Serial.print(" ");
    }
    Serial.println();
  }
}

void printSensorValues() {
  Serial.print("实时传感器值: ");
  for (int i = 0; i < NUM_SENSORS; i++) {
    Serial.print("A");
    Serial.print(i);
    Serial.print("=");
    Serial.print(analogRead(SENSOR_PINS[i]));
    Serial.print(" ");
  }
  Serial.println();
}

void printHelp() {
  Serial.println("=== BSL手势识别系统命令帮助 ===");
  Serial.println("s/S - 显示详细系统状态");
  Serial.println("r/R - 重置系统");
  Serial.println("b/B - 显示数据缓冲区信息");
  Serial.println("d/D - 显示实时传感器值");
  Serial.println("h/H - 显示此帮助信息");
  Serial.println();
  Serial.println("智能推理流程:");
  Serial.println("1. 静态监测 - 低功耗监测传感器变化");
  Serial.println("2. 手势检测 - 检测到变化时确认手势开始");
  Serial.println("3. 完整推理 - 执行1D_CNN模型推理");
  Serial.println("4. 冷却期 - 手势结束后的稳定期");
}

void handleSerialCommands() {
  if (Serial.available()) {
    char command = Serial.read();
    
    switch (command) {
      case 's':
      case 'S':
        printDetailedStatus();
        break;
      case 'r':
      case 'R':
        resetSystem();
        break;
      case 'b':
      case 'B':
        printBufferInfo();
        break;
      case 'd':
      case 'D':
        printSensorValues();
        break;
      case 'h':
      case 'H':
        printHelp();
        break;
      default:
        Serial.println("❓ 未知命令，发送 'h' 查看帮助");
        break;
    }
  }
}

// ===== 主程序 =====

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);  // 等待3秒或串口连接
  
  Serial.println("=== BSL手势识别系统 - 1D_CNN智能推理版本 ===");
  Serial.println("正在初始化...");
  
  // 初始化传感器引脚
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(SENSOR_PINS[i], INPUT);
  }
  
  // 初始化LED
  pinMode(LED_PIN, OUTPUT);
  for (int i = 0; i < 5; i++) {
    pinMode(RESULT_LED_PINS[i], OUTPUT);
    digitalWrite(RESULT_LED_PINS[i], LOW);
  }
  
  // 初始化TensorFlow Lite模型
  Serial.println("正在初始化1D_CNN模型...");
  if (!modelInit(model, tensorArena, tensorArenaSize)) {
    Serial.println("❌ 1D_CNN模型初始化失败!");
    while (true) {
      digitalWrite(LED_PIN, HIGH);
      delay(200);
      digitalWrite(LED_PIN, LOW);
      delay(200);
    }
  }
  Serial.println("✅ 1D_CNN模型初始化成功");
  
  // 初始化预测历史
  for (int i = 0; i < STABLE_PREDICTION_COUNT; i++) {
    lastPredictions[i] = -1;
  }
  
  // 初始化缓冲区
  initializeBuffer();
  
  Serial.println("✅ 系统初始化完成!");
  Serial.println("智能推理模式：静态监测 -> 手势检测 -> 完整推理");
  Serial.println("发送 'h' 查看帮助信息");
  Serial.println();
  
  digitalWrite(LED_PIN, HIGH);
  delay(500);
  digitalWrite(LED_PIN, LOW);
}

void loop() {
  unsigned long currentTime = millis();
  
  // 定时采集传感器数据
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
    collectSensorData();
    lastSampleTime = currentTime;
    
    // 智能推理决策
    performIntelligentInference();
  }
  
  // 处理串口命令
  handleSerialCommands();
  
  // 定时状态输出
  if (currentTime - lastStatusTime >= 5000) {  // 每5秒输出一次状态
    printSystemStatus();
    lastStatusTime = currentTime;
  }
} 