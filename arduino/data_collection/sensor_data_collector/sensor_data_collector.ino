/*
  BSL Gesture Recognition System - Data Collection Firmware
  
  Purpose:
  - Continuous raw sensor data collection at 50Hz
  - Real-time data streaming via Serial
  - No filtering applied (raw ADC values)
  - Optimized for Python data collection and post-processing
  
  Hardware:
  - Arduino Nano 33 BLE Sense Rev2
  - 5 DIY flexible sensors connected to A0-A4
  
  Author: Lambert Yang
  Version: 1.1 (Raw Data Collection)
*/

// Configuration
const int SENSOR_PINS[] = {A0, A1, A2, A3, A4};
const int NUM_SENSORS = 5;
const int SAMPLE_RATE = 50;  // Hz
const int SAMPLE_INTERVAL = 1000 / SAMPLE_RATE;  // 20ms

// Global variables
unsigned long lastSampleTime = 0;
unsigned long dataPacketCount = 0;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }
  
  // Initialize sensor pins
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(SENSOR_PINS[i], INPUT);
  }
  
  // Initialize LED
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  
  Serial.println("BSL Data Collection System Initialized");
  Serial.println("Sample Rate: 50Hz");
  Serial.println("Data Format: timestamp,sensor1,sensor2,sensor3,sensor4,sensor5");
  Serial.println("Mode: Raw data collection (no filtering)");
  Serial.println("Starting data collection...");
  
  lastSampleTime = millis();
  delay(1000);  // Give time for Python to connect
}

void loop() {
  unsigned long currentTime = millis();
  
  // Check if it's time to sample
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
    collectAndSendData(currentTime);
    lastSampleTime = currentTime;
    
    // Blink LED to show activity
    if (dataPacketCount % 50 == 0) {
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    }
  }
}

void collectAndSendData(unsigned long timestamp) {
  // Read raw sensor values (no filtering)
  int rawValues[NUM_SENSORS];
  for (int i = 0; i < NUM_SENSORS; i++) {
    rawValues[i] = analogRead(SENSOR_PINS[i]);
  }
  
  // Send raw data via serial
  Serial.print(timestamp);
  for (int i = 0; i < NUM_SENSORS; i++) {
    Serial.print(",");
    Serial.print(rawValues[i]);  // Raw integer values
  }
  Serial.println();
  
  dataPacketCount++;
}

/**
 * Print sensor statistics (for debugging)
 */
void printSensorStats() {
  Serial.println("=== Sensor Status ===");
  for (int i = 0; i < NUM_SENSORS; i++) {
    // Read current raw value
    int rawValue = analogRead(SENSOR_PINS[i]);
    Serial.print("Sensor ");
    Serial.print(i + 1);
    Serial.print(": Raw=");
    Serial.println(rawValue);
  }
  Serial.println("====================");
}

/**
 * Calibrate sensors (optional feature)
 * Record baseline values in static state
 */
void calibrateSensors() {
  Serial.println("Starting sensor calibration...");
  Serial.println("Please keep hand in resting position for 3 seconds");
  
  float baselines[NUM_SENSORS] = {0};
  int numSamples = 150;  // 3 seconds × 50Hz = 150 samples
  
  for (int sample = 0; sample < numSamples; sample++) {
    // Read sensors directly
    for (int i = 0; i < NUM_SENSORS; i++) {
      int rawValue = analogRead(SENSOR_PINS[i]);
      baselines[i] += rawValue;
    }
    
    delay(20);  // 50Hz sampling
  }
  
  // Calculate average baseline values
  Serial.println("Calibration complete! Baseline values:");
  for (int i = 0; i < NUM_SENSORS; i++) {
    baselines[i] /= numSamples;
    Serial.print("Sensor ");
    Serial.print(i + 1);
    Serial.print(": ");
    Serial.println(baselines[i], 2);
  }
}

/**
 * Handle serial commands (optional feature)
 */
void handleSerialCommands() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "stats") {
      printSensorStats();
    } else if (command == "calibrate") {
      calibrateSensors();
    } else if (command == "help") {
      Serial.println("Available commands:");
      Serial.println("  stats     - Show sensor status");
      Serial.println("  calibrate - Calibrate sensors");
      Serial.println("  help      - Show this help");
    }
  }
} 