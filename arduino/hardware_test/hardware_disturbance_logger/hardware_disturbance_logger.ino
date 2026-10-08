/*
  Hardware disturbance logger for GoP glove measurements.

  Purpose:
  - Stream device timestamps plus five inverted analogue channels at about 50 Hz.
  - Match the existing sensor_data_collector.ino reading convention: 1023 - raw.
  - Support host-controlled start/stop with 'S' and 'X'.
*/

#include <Arduino.h>

const uint8_t numSensors = 5;
const uint8_t sensorPins[numSensors] = {A0, A1, A2, A3, A4};
const bool invertReadings = true;
const unsigned long sampleIntervalUs = 20000UL;

bool sending = false;
unsigned long nextSampleUs = 0;

void setup() {
  Serial.begin(115200);
  analogReadResolution(10);
  delay(200);
  Serial.println("Ready");
  Serial.println("device_us\tA0\tA1\tA2\tA3\tA4");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'S') {
      sending = true;
      nextSampleUs = micros();
    } else if (command == 'X') {
      sending = false;
    }
  }

  if (!sending) {
    return;
  }

  unsigned long now = micros();
  if ((long)(now - nextSampleUs) < 0) {
    return;
  }

  Serial.print(now);
  Serial.print('\t');
  for (uint8_t i = 0; i < numSensors; i++) {
    int raw = analogRead(sensorPins[i]);
    int value = invertReadings ? (1023 - raw) : raw;
    Serial.print(value);
    if (i < numSensors - 1) {
      Serial.print('\t');
    }
  }
  Serial.println();

  nextSampleUs += sampleIntervalUs;
  if ((long)(micros() - nextSampleUs) > (long)sampleIntervalUs) {
    nextSampleUs = micros() + sampleIntervalUs;
  }
}
