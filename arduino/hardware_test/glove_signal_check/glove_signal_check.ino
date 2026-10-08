/*
  Simple glove signal check

  This sketch follows sensor_data_collector.ino:
  - Read five analog sensors on A0-A4
  - Invert readings with 1023 - raw
  - Print tab-separated values at about 50Hz

  Open Serial Monitor at 115200 baud.
*/

#include <Arduino.h>

const uint8_t numSensors = 5;
const uint8_t sensorPins[numSensors] = {A0, A1, A2, A3, A4};

// analogRead() returns 0..1023, so inverted value is 1023 - raw.
const bool invertReadings = true;

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("Ready");
  Serial.println("A0\tA1\tA2\tA3\tA4");
}

void loop() {
  for (uint8_t i = 0; i < numSensors; i++) {
    int raw = analogRead(sensorPins[i]);
    int v = invertReadings ? (1023 - raw) : raw;

    Serial.print(v);
    if (i < numSensors - 1) {
      Serial.print('\t');
    }
  }

  Serial.println();
  delay(20);
}
