/*
  BSL Gesture Recognition System - XGBoost Inference Firmware

  Features:
  - Real-time sensor data collection and sliding window buffering
  - Smart gesture detection: Static pre-detection + dynamic full inference
  - XGBoost model inference (using a custom inference engine)
  - Serial communication

  Hardware:
  - Arduino Nano 33 BLE Sense Rev2
  - 5 DIY flex sensors on A0-A4

  Author: Lambert Yang
  Version: 3.1
*/

#include "bsl_model_XGBoost_Arduino.h"

// ===== Hardware Config =====
const int SENSOR_PINS[] = {A0, A1, A2, A3, A4};
const int NUM_SENSORS = 5;
const int SEQUENCE_LENGTH = 100;
const int SAMPLE_RATE = 50;  // Hz
const int SAMPLE_INTERVAL = 1000 / SAMPLE_RATE; // 20ms

// ===== Smart Inference Config =====
const float STATIC_THRESHOLD = 30.0;
const float GESTURE_TRIGGER_THRESHOLD = 50.0;
const int MIN_GESTURE_DURATION = 30;
const float CONFIDENCE_THRESHOLD = 0.7; // XGBoost uses probabilities

// ===== Data Buffer & State =====
float sensor_buffer[SEQUENCE_LENGTH][NUM_SENSORS];
int buffer_index = 0;
unsigned long last_sample_time = 0;

enum SystemState {
  STATE_IDLE,
  STATE_DETECTING,
  STATE_COOLDOWN
};
SystemState current_state = STATE_IDLE;

unsigned long gesture_start_time = 0;
unsigned long cooldown_start_time = 0;

// Gesture Labels
const char* GESTURE_LABELS[] = {
  "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Static"
};
const int NUM_GESTURES = 11;

// Function Prototypes
void perform_inference(float* input_data, float* output_data);
void extract_features(float* sequence, float* features);

// ================================================================================================
//                                       Main Program
// ================================================================================================

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(SENSOR_PINS[i], INPUT);
  }
  
  Serial.println("✅ XGBoost System Initialized, waiting for gesture...");
}

void loop() {
  if (millis() - last_sample_time >= SAMPLE_INTERVAL) {
    last_sample_time = millis();
    
    float current_reading[NUM_SENSORS];
    for (int i = 0; i < NUM_SENSORS; i++) {
      current_reading[i] = analogRead(SENSOR_PINS[i]);
    }
    
    float difference = 0;
    int prev_index = (buffer_index == 0) ? SEQUENCE_LENGTH - 1 : buffer_index - 1;
    for (int i = 0; i < NUM_SENSORS; i++) {
      difference += abs(current_reading[i] - sensor_buffer[prev_index][i]);
    }
    
    for (int i = 0; i < NUM_SENSORS; i++) {
      sensor_buffer[buffer_index][i] = current_reading[i];
    }
    buffer_index = (buffer_index + 1) % SEQUENCE_LENGTH;

    switch (current_state) {
      case STATE_IDLE:
        if (difference > GESTURE_TRIGGER_THRESHOLD) {
          current_state = STATE_DETECTING;
          gesture_start_time = millis();
          Serial.println("Change detected, starting to record...");
        }
        break;
        
      case STATE_DETECTING:
        if (millis() - gesture_start_time > (MIN_GESTURE_DURATION * SAMPLE_INTERVAL)) {
          if (difference < STATIC_THRESHOLD) {
            Serial.println("Movement finished, starting inference...");
            
            float inference_buffer[SEQUENCE_LENGTH * NUM_SENSORS];
            int current_pos = buffer_index;
            for (int i = 0; i < SEQUENCE_LENGTH; i++) {
                for (int j = 0; j < NUM_SENSORS; j++) {
                    inference_buffer[i * NUM_SENSORS + j] = sensor_buffer[current_pos][j];
                }
                current_pos = (current_pos + 1) % SEQUENCE_LENGTH;
            }

            // CRITICAL: Apply StandardScaler normalization
            for (int i = 0; i < SEQUENCE_LENGTH; i++) {
              for (int j = 0; j < NUM_SENSORS; j++) {
                  int index = i * NUM_SENSORS + j;
                  inference_buffer[index] = (inference_buffer[index] - scaler_mean[j]) / scaler_scale[j];
              }
            }
            
            float output_data[NUM_GESTURES];
            perform_inference(inference_buffer, output_data);
            
            int predicted_gesture = -1;
            float max_confidence = 0.0;
            for (int i = 0; i < NUM_GESTURES; i++) {
              if (output_data[i] > max_confidence) {
                max_confidence = output_data[i];
                predicted_gesture = i;
              }
            }
            
            if (max_confidence > CONFIDENCE_THRESHOLD) {
              Serial.print("✅ Detected: ");
              Serial.print(GESTURE_LABELS[predicted_gesture]);
              Serial.print(" (");
              Serial.print(max_confidence * 100);
              Serial.println("%)");
            } else {
              Serial.println("⚠️ No gesture recognized.");
            }

            current_state = STATE_COOLDOWN;
            cooldown_start_time = millis(); 
          }
        }
        break;

      case STATE_COOLDOWN:
        if (millis() - cooldown_start_time > 2000) { // Cooldown for 2 seconds
           Serial.println("Cooldown finished. Waiting for new gesture...");
           current_state = STATE_IDLE;
        }
        break;
    }
  }
}

// ================================================================================================
//                                    XGBoost Inference Functions
// ================================================================================================

void extract_features(float* sequence, float* features) {
    for (int i = 0; i < NUM_SENSORS; i++) {
        float min_val = sequence[i];
        float max_val = sequence[i];
        float sum = 0;
        float sum_sq = 0;

        for (int j = 0; j < SEQUENCE_LENGTH; j++) {
            float val = sequence[j * NUM_SENSORS + i];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum += val;
            sum_sq += val * val;
        }

        float mean = sum / SEQUENCE_LENGTH;
        float std_dev = sqrt(sum_sq / SEQUENCE_LENGTH - mean * mean);

        features[i * 4 + 0] = mean;
        features[i * 4 + 1] = std_dev;
        features[i * 4 + 2] = min_val;
        features[i * 4 + 3] = max_val;
    }
}

void perform_inference(float* input_data, float* output_data) {
    float features[N_FEATURES];
    extract_features(input_data, features);
    
    predict_xgboost(features, output_data); 
} 