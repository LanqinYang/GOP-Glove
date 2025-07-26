/*
  BSL Gesture Recognition System - Transformer Encoder Inference Firmware

  Features:
  - Real-time sensor data collection and sliding window buffering
  - Smart gesture detection: Static pre-detection + dynamic full inference
  - Transformer Encoder TensorFlow Lite model inference
  - Serial communication

  Hardware:
  - Arduino Nano 33 BLE Sense Rev2
  - 5 DIY flex sensors on A0-A4

  Author: Lambert Yang
  Version: 3.1
*/

#include <Chirale_TensorFlowLite.h>
#include "bsl_model_Transformer_Encoder.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

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
const float CONFIDENCE_THRESHOLD = 0.7;

// ===== TensorFlow Lite Config =====
constexpr int kTensorArenaSize = 45 * 1024; // Transformer needs ~40KB, increased to 45KB
alignas(16) byte tensor_arena[kTensorArenaSize];

// TensorFlow Lite globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

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
void setup_tf();
void perform_inference(float* input_data, float* output_data);

// ================================================================================================
//                                       Main Program
// ================================================================================================

void setup() {
  Serial.begin(115200);
  while (!Serial);

  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(SENSOR_PINS[i], INPUT);
  }

  setup_tf();
  
  Serial.println("✅ System initialized, waiting for gesture...");
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
//                                    TensorFlow Lite Functions
// ================================================================================================

void setup_tf() {
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (true);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void perform_inference(float* input_data, float* output_data) {
    if (!input || !output) {
        Serial.println("Input/Output tensor not ready!");
        return;
    }

    for (int i = 0; i < SEQUENCE_LENGTH * NUM_SENSORS; i++) {
        input->data.int8[i] = static_cast<int8_t>((input_data[i] / input->params.scale) + input->params.zero_point);
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed");
        return;
    }

    for (int i = 0; i < NUM_GESTURES; i++) {
        int8_t y_quantized = output->data.int8[i];
        output_data[i] = static_cast<float>(y_quantized - output->params.zero_point) * output->params.scale;
    }
} 