/*
  BSL Gesture Recognition System - Data Collection Firmware
  
  Purpose:
  - Continuous raw sensor data collection at 50Hz
  - Real-time data streaming via Serial
  
  Author: Lambert Yang
  Version: 1.2 (Simplified)
*/

#include <Arduino.h>

const uint8_t numSensors = 5;
const uint8_t sensorPins[numSensors] = {A0, A1, A2, A3, A4};

// 如果要"倒过来"显示，把 true 改成 false 则显示原始方向
const bool invertReadings = true;

void setup() {
  Serial.begin(115200);
  // 给串口一点时间
  delay(200);
  Serial.println("Ready");
}

void loop() {
  // 依次读取并打印五路数据，中间用 tab 分隔
  for (uint8_t i = 0; i < numSensors; i++) {
    int raw = analogRead(sensorPins[i]); // 0..1023
    int v = invertReadings ? (1023 - raw) : raw;
    Serial.print(v);
    if (i < numSensors - 1) {
      Serial.print('\t');
    }
  }
  Serial.println();
  // 控制刷新率，20ms 大约 50Hz
  delay(20);
} 