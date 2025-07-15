/*
  BSL手势识别系统 - Transformer_Encoder模型智能推理固件
  
  智能推理机制：持续静态检测+动态完整推理
  模型类型: Transformer_Encoder (最高精度)
  作者: Lambert Yang
  版本: 3.0 (智能推理版本)
*/

#include <ArduTFLite.h>
#include "bsl_model_Transformer_Encoder.h"

// 硬件配置
const int SENSOR_PINS[] = {A0, A1, A2, A3, A4};
const int NUM_SENSORS = 5;
const int SEQUENCE_LENGTH = 100;
const int SAMPLE_RATE = 50;
const int SAMPLE_INTERVAL = 20;  // 20ms

// 智能推理配置
const float STATIC_THRESHOLD = 30.0;
const float GESTURE_TRIGGER_THRESHOLD = 50.0;
const int STATIC_CHECK_SAMPLES = 10;
const int MIN_GESTURE_DURATION = 30;
const float CONFIDENCE_THRESHOLD = 0.8;   // Transformer最高阈值
const int STABLE_PREDICTION_COUNT = 5;    // Transformer需要最多稳定性

// TensorFlow Lite配置
constexpr int tensorArenaSize = 48 * 1024;  // Transformer需要约48KB
alignas(16) byte tensorArena[tensorArenaSize];

// 数据缓存和状态
float sensorBuffer[SEQUENCE_LENGTH][NUM_SENSORS];
int bufferIndex = 0;
bool bufferFilled = false;

// 智能检测状态
enum DetectionState {
  STATE_STATIC, STATE_GESTURE_START, STATE_GESTURE_ACTIVE, STATE_COOLDOWN
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

// LED配置
const int LED_PIN = LED_BUILTIN;
const int RESULT_LED_PINS[] = {2, 3, 4, 5, 6};

// 手势类别名称
const char* GESTURE_NAMES[] = {
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "静态"
};
const int NUM_GESTURES = 11;

// ============================================================================
// 工具函数（最低依赖级别）
// ============================================================================

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
  
  if (gestureClass == 10) {
    digitalWrite(LED_PIN, LOW);
  } else {
    digitalWrite(LED_PIN, HIGH);
  }
}

void calculateRecentVariance() {
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

void updatePredictionHistory(int prediction) {
  lastPredictions[predictionIndex] = prediction;
  predictionIndex = (predictionIndex + 1) % STABLE_PREDICTION_COUNT;
}

int getStablePrediction() {
  for (int target = 0; target < NUM_GESTURES; target++) {
    int count = 0;
    for (int i = 0; i < STABLE_PREDICTION_COUNT; i++) {
      if (lastPredictions[i] == target) count++;
    }
    if (count >= (STABLE_PREDICTION_COUNT + 1) / 2) {
      return target;
    }
  }
  return -1;
}

// ============================================================================
// 状态管理函数
// ============================================================================

void checkForGestureStart() {
  float totalVariance = 0.0;
  for (int i = 0; i < NUM_SENSORS; i++) {
    totalVariance += recentVariance[i];
  }
  
  if (totalVariance > GESTURE_TRIGGER_THRESHOLD) {
    currentState = STATE_GESTURE_START;
    stateStartTime = millis();
    gestureFrameCount = 0;
    Serial.println("🎯 Transformer检测到可能的手势开始");
    digitalWrite(LED_PIN, HIGH);
  }
}

void confirmGestureStart() {
  gestureFrameCount++;
  
  if (gestureFrameCount >= MIN_GESTURE_DURATION) {
    currentState = STATE_GESTURE_ACTIVE;
    Serial.println("✅ 手势确认，开始Transformer完整推理");
    lastInferenceTime = millis();
  } else if (millis() - stateStartTime > 1000) {
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
    Serial.println("📉 Transformer手势结束，进入冷却期");
  }
}

void checkCooldownEnd() {
  if (millis() - stateStartTime > 2000) {  // Transformer冷却期最长
    currentState = STATE_STATIC;
    currentGestureClass = -1;
    currentConfidence = 0.0;
    Serial.println("🔄 返回静态监测状态");
    digitalWrite(LED_PIN, LOW);
    clearLEDIndicators();
  }
}

// ============================================================================
// 推理函数
// ============================================================================

void outputGestureResult(int gestureClass, float confidence) {
  static unsigned long lastOutputTime = 0;
  unsigned long currentTime = millis();
  
  if (currentTime - lastOutputTime < 800) return;  // Transformer输出频率最低
  
  Serial.print("🤲 Transformer识别结果: ");
  Serial.print(GESTURE_NAMES[gestureClass]);
  Serial.print(" (置信度: ");
  Serial.print(confidence * 100, 1);
  Serial.println("%)");
  
  lastOutputTime = currentTime;
}

void performCompleteInference() {
  unsigned long currentTime = millis();
  
  // Transformer需要最多计算时间，频率最低
  if (currentTime - lastInferenceTime < 500) return;  // 约2Hz
  
  int inputIdx = 0;
  int startIdx = (bufferIndex - SEQUENCE_LENGTH + SEQUENCE_LENGTH) % SEQUENCE_LENGTH;
  
  for (int t = 0; t < SEQUENCE_LENGTH; t++) {
    int dataIdx = (startIdx + t) % SEQUENCE_LENGTH;
    for (int s = 0; s < NUM_SENSORS; s++) {
      float normalized = (sensorBuffer[dataIdx][s] - scaler_mean[s]) / scaler_scale[s];
      modelSetInput(normalized, inputIdx++);
    }
  }
  
  if (!modelRunInference()) {
    Serial.println("❌ Transformer推理失败!");
    return;
  }
  
  float maxProb = 0.0;
  int predictedClass = 0;
  
  for (int i = 0; i < NUM_GESTURES; i++) {
    float prob = modelGetOutput(i);
    if (prob > maxProb) {
      maxProb = prob;
      predictedClass = i;
    }
  }
  
  updatePredictionHistory(predictedClass);
  
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
  
  calculateRecentVariance();
  
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

// ============================================================================
// 数据收集函数
// ============================================================================

void collectSensorData() {
  for (int i = 0; i < NUM_SENSORS; i++) {
    float rawValue = analogRead(SENSOR_PINS[i]);
    
    if (bufferFilled) {
      int prevIdx = (bufferIndex - 1 + SEQUENCE_LENGTH) % SEQUENCE_LENGTH;
      float alpha = 0.3;
      sensorBuffer[bufferIndex][i] = alpha * rawValue + (1 - alpha) * sensorBuffer[prevIdx][i];
    } else {
      sensorBuffer[bufferIndex][i] = rawValue;
    }
  }
  
  bufferIndex = (bufferIndex + 1) % SEQUENCE_LENGTH;
  if (!bufferFilled && bufferIndex == 0) {
    bufferFilled = true;
    Serial.println("📊 Transformer数据缓冲区已填充完成");
  }
}

// ============================================================================
// 串口和显示函数
// ============================================================================

void printHelp() {
  Serial.println("=== Transformer_Encoder手势识别系统命令帮助 ===");
  Serial.println("s/S - 显示详细系统状态");
  Serial.println("r/R - 重置系统");
  Serial.println("b/B - 显示数据缓冲区信息");
  Serial.println("d/D - 显示实时传感器值");
  Serial.println("h/H - 显示此帮助信息");
  Serial.println();
  Serial.println("Transformer特点：");
  Serial.println("- 最高的手势识别精度");
  Serial.println("- 强大的注意力机制");
  Serial.println("- 计算时间最长，内存占用最大");
  Serial.println("- 最适合复杂手势模式识别");
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

void printBufferInfo() {
  Serial.println("=== Transformer数据缓冲区信息 ===");
  Serial.print("状态: ");
  Serial.println(bufferFilled ? "已填充" : "填充中");
  Serial.print("当前索引: ");
  Serial.println(bufferIndex);
}

void resetSystem() {
  Serial.println("🔄 重置Transformer系统...");
  
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

void printSystemStatus() {
  Serial.print("📊 Transformer状态: ");
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
  Serial.println("=== Transformer_Encoder系统详细状态 ===");
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

// ============================================================================
// 主程序
// ============================================================================

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  
  Serial.println("=== BSL手势识别系统 - Transformer_Encoder智能推理版本 ===");
  
  // 初始化硬件
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(SENSOR_PINS[i], INPUT);
  }
  pinMode(LED_PIN, OUTPUT);
  for (int i = 0; i < 5; i++) {
    pinMode(RESULT_LED_PINS[i], OUTPUT);
    digitalWrite(RESULT_LED_PINS[i], LOW);
  }
  
  // 初始化TensorFlow Lite模型
  Serial.println("正在初始化Transformer_Encoder模型...");
  if (!modelInit(model, tensorArena, tensorArenaSize)) {
    Serial.println("❌ Transformer_Encoder模型初始化失败!");
    while (true) {
      digitalWrite(LED_PIN, HIGH);
      delay(200);
      digitalWrite(LED_PIN, LOW);
      delay(200);
    }
  }
  Serial.println("✅ Transformer_Encoder模型初始化成功");
  
  // 初始化预测历史
  for (int i = 0; i < STABLE_PREDICTION_COUNT; i++) {
    lastPredictions[i] = -1;
  }
  
  initializeBuffer();
  
  Serial.println("✅ 系统初始化完成!");
  Serial.println("智能推理模式：静态监测 -> 手势检测 -> Transformer推理");
  Serial.println("注意：Transformer模型提供最高精度，但计算时间较长");
  
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
    performIntelligentInference();
  }
  
  // 处理串口命令
  handleSerialCommands();
  
  // 定时状态输出
  if (currentTime - lastStatusTime >= 5000) {
    printSystemStatus();
    lastStatusTime = currentTime;
  }
} 