/*
  BSL手势识别系统 - XGBoost_Arduino模型智能推理固件
  
  智能推理机制：持续静态检测+动态完整推理
  模型类型: XGBoost_Arduino (基于决策树，最快推理)
  作者: Lambert Yang
  版本: 3.0 (智能推理版本)
*/

#include "bsl_model_XGBoost_Arduino.h"

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
const int MIN_GESTURE_DURATION = 20;   // XGBoost响应更快
const float CONFIDENCE_THRESHOLD = 0.6; // XGBoost阈值较低
const int STABLE_PREDICTION_COUNT = 2;  // XGBoost只需少量稳定性

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

// XGBoost分类器对象
Eloquent::ML::Port::XGBClassifier xgb_classifier;

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

void extractStatisticalFeatures(float* features) {
  // 为每个传感器提取5个统计特征：均值、方差、最小值、最大值、范围
  int featureIdx = 0;
  
  for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
    float sum = 0.0, sumSquared = 0.0;
    float minVal = 1023.0, maxVal = 0.0;
    
    // 计算最近50个样本的统计特征
    int samplesCount = min(50, SEQUENCE_LENGTH);
    for (int i = 0; i < samplesCount; i++) {
      int idx = (bufferIndex - 1 - i + SEQUENCE_LENGTH) % SEQUENCE_LENGTH;
      float value = sensorBuffer[idx][sensor];
      
      sum += value;
      sumSquared += value * value;
      minVal = min(minVal, value);
      maxVal = max(maxVal, value);
    }
    
    float mean = sum / samplesCount;
    float variance = (sumSquared / samplesCount) - (mean * mean);
    float range = maxVal - minVal;
    
    // 应用归一化
    features[featureIdx++] = (mean - scaler_mean[sensor]) / scaler_scale[sensor];
    features[featureIdx++] = variance / 1000.0;  // 简单缩放
    features[featureIdx++] = (minVal - scaler_mean[sensor]) / scaler_scale[sensor];
    features[featureIdx++] = (maxVal - scaler_mean[sensor]) / scaler_scale[sensor];
    features[featureIdx++] = range / 1000.0;    // 简单缩放
  }
}

float calculateConfidence(float* features) {
  // 基于特征质量的简单置信度估计
  float totalVariance = 0.0;
  for (int i = 1; i < 25; i += 5) {  // 使用方差特征
    totalVariance += features[i];
  }
  
  // 将方差映射到置信度（高方差=高置信度，表示有明显手势）
  float confidence = min(1.0, totalVariance / 100.0);
  return max(0.0, confidence);
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
    Serial.println("🎯 XGBoost检测到可能的手势开始");
    digitalWrite(LED_PIN, HIGH);
  }
}

void confirmGestureStart() {
  gestureFrameCount++;
  
  if (gestureFrameCount >= MIN_GESTURE_DURATION) {
    currentState = STATE_GESTURE_ACTIVE;
    Serial.println("✅ 手势确认，开始XGBoost完整推理");
    lastInferenceTime = millis();
  } else if (millis() - stateStartTime > 800) {  // XGBoost超时更短
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
    Serial.println("📉 XGBoost手势结束，进入冷却期");
  }
}

void checkCooldownEnd() {
  if (millis() - stateStartTime > 500) {  // XGBoost冷却期最短
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
  
  if (currentTime - lastOutputTime < 200) return;  // XGBoost输出频率较高
  
  Serial.print("🤲 XGBoost识别结果: ");
  Serial.print(GESTURE_NAMES[gestureClass]);
  Serial.print(" (置信度: ");
  Serial.print(confidence * 100, 1);
  Serial.println("%)");
  
  lastOutputTime = currentTime;
}

void performCompleteInference() {
  unsigned long currentTime = millis();
  
  // XGBoost推理非常快，可以高频率执行
  if (currentTime - lastInferenceTime < 100) return;  // 约10Hz
  
  // 提取统计特征（XGBoost使用特征而非原始序列）
  float features[25];  // 5个传感器 * 5个统计特征
  extractStatisticalFeatures(features);
  
  // 执行XGBoost推理
  int predictedClass = xgb_classifier.predict(features);
  
  // 计算置信度（基于特征质量的简单估计）
  float confidence = calculateConfidence(features);
  
  updatePredictionHistory(predictedClass);
  
  int stableClass = getStablePrediction();
  if (stableClass >= 0 && confidence > CONFIDENCE_THRESHOLD) {
    currentGestureClass = stableClass;
    currentConfidence = confidence;
    
    outputGestureResult(stableClass, confidence);
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
    Serial.println("📊 XGBoost数据缓冲区已填充完成");
  }
}

// ============================================================================
// 串口和显示函数
// ============================================================================

void printHelp() {
  Serial.println("=== XGBoost_Arduino手势识别系统命令帮助 ===");
  Serial.println("s/S - 显示详细系统状态");
  Serial.println("r/R - 重置系统");
  Serial.println("b/B - 显示数据缓冲区信息");
  Serial.println("d/D - 显示实时传感器值");
  Serial.println("f/F - 显示特征向量");
  Serial.println("h/H - 显示此帮助信息");
  Serial.println();
  Serial.println("XGBoost特点：");
  Serial.println("- 最快的推理速度");
  Serial.println("- 基于决策树，不使用神经网络");
  Serial.println("- 使用统计特征而非原始序列");
  Serial.println("- 内存占用最小，功耗最低");
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
  Serial.println("=== XGBoost数据缓冲区信息 ===");
  Serial.print("状态: ");
  Serial.println(bufferFilled ? "已填充" : "填充中");
  Serial.print("当前索引: ");
  Serial.println(bufferIndex);
}

void printFeatures() {
  if (!bufferFilled) {
    Serial.println("缓冲区未填充，无法提取特征");
    return;
  }
  
  float features[25];
  extractStatisticalFeatures(features);
  
  Serial.println("=== XGBoost特征向量 ===");
  for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
    Serial.print("传感器");
    Serial.print(sensor);
    Serial.print(": 均值=");
    Serial.print(features[sensor*5], 2);
    Serial.print(" 方差=");
    Serial.print(features[sensor*5+1], 2);
    Serial.print(" 最小=");
    Serial.print(features[sensor*5+2], 2);
    Serial.print(" 最大=");
    Serial.print(features[sensor*5+3], 2);
    Serial.print(" 范围=");
    Serial.println(features[sensor*5+4], 2);
  }
}

void resetSystem() {
  Serial.println("🔄 重置XGBoost系统...");
  
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
  Serial.print("📊 XGBoost状态: ");
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
  Serial.println("=== XGBoost_Arduino系统详细状态 ===");
  Serial.println("模型状态: 已加载 | 算法: 决策树集成");
  
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
      case 'f':
      case 'F':
        printFeatures();
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
  
  Serial.println("=== BSL手势识别系统 - XGBoost_Arduino智能推理版本 ===");
  
  // 初始化硬件
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(SENSOR_PINS[i], INPUT);
  }
  pinMode(LED_PIN, OUTPUT);
  for (int i = 0; i < 5; i++) {
    pinMode(RESULT_LED_PINS[i], OUTPUT);
    digitalWrite(RESULT_LED_PINS[i], LOW);
  }
  
  // XGBoost模型无需特殊初始化
  Serial.println("✅ XGBoost_Arduino模型准备就绪");
  
  // 初始化预测历史
  for (int i = 0; i < STABLE_PREDICTION_COUNT; i++) {
    lastPredictions[i] = -1;
  }
  
  initializeBuffer();
  
  Serial.println("✅ 系统初始化完成!");
  Serial.println("智能推理模式：静态监测 -> 手势检测 -> XGBoost推理");
  Serial.println("特点：超快推理速度，基于决策树算法");
  
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