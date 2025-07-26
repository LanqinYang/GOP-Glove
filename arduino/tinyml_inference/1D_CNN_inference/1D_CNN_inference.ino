/*
  BSL手势识别系统 - 1D_CNN模型智能推理固件
  
  功能：
  - 实时传感器数据采集和滑动窗口缓存
  - 智能手势检测：静态预检测+动态完整推理
  - 1D_CNN TensorFlow Lite模型推理
  - 串口通信
  
  硬件要求：
  - Arduino Nano 33 BLE Sense Rev2
  - 5个DIY柔性传感器连接到A0-A4
  
  智能推理机制：
  - 持续低成本静态检测（检测是否有手势发生）
  - 检测到手势变化时自动触发完整模型推理
  - 节省计算资源，提高响应速度
  
  模型类型: 1D_CNN
  作者: Lambert Yang
  版本: 3.1 (修复状态机BUG)
*/

#include <Chirale_TensorFlowLite.h>
#include "bsl_model_1D_CNN.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"


// ===== 硬件配置 =====
const int SENSOR_PINS[] = {A0, A1, A2, A3, A4};
const int NUM_SENSORS = 5;
const int SEQUENCE_LENGTH = 100;
const int SAMPLE_RATE = 50;  // Hz
const int SAMPLE_INTERVAL = 1000 / SAMPLE_RATE;  // 20ms

// ===== 智能推理配置 =====
const float STATIC_THRESHOLD = 30.0;        // 静态检测阈值
const float GESTURE_TRIGGER_THRESHOLD = 50.0; // 手势触发阈值
const int MIN_GESTURE_DURATION = 30;        // 最小手势持续时间（样本数）
const float CONFIDENCE_THRESHOLD = 0.7;     // 置信度阈值
const int STABLE_PREDICTION_COUNT = 3;      // 稳定预测需要的次数

// ===== TensorFlow Lite配置 =====
constexpr int kTensorArenaSize = 30 * 1024;  // 1D_CNN需要约28KB, 增加到30KB
alignas(16) byte tensor_arena[kTensorArenaSize];

// TensorFlow Lite相关全局变量
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// ===== 数据缓存和状态 =====
float sensor_buffer[SEQUENCE_LENGTH][NUM_SENSORS];
int buffer_index = 0;
unsigned long last_sample_time = 0;

enum SystemState {
  STATE_IDLE,           // 等待手势
  STATE_DETECTING,      // 检测到变化，正在记录
  STATE_COOLDOWN        // 推理完成，冷却中
};
SystemState current_state = STATE_IDLE;

unsigned long gesture_start_time = 0;
unsigned long cooldown_start_time = 0; // *** MODIFICATION 1: cooldown_start_time is now a global variable ***

// 手势标签
const char* GESTURE_LABELS[] = {
  "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Static"
};
const int NUM_GESTURES = 11;

// ===== 函数声明 =====
void setup_tf();
void perform_inference(float* input_data, float* output_data);
// void normalize_features(float* features, int num_features); // 假设你在别处定义了此函数


// ================================================================================================
//                                       主程序 (Setup & Loop)
// ================================================================================================

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("DEBUG: Serial connection established.");

  // 初始化传感器
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(SENSOR_PINS[i], INPUT);
  }
  Serial.println("DEBUG: Sensors initialized.");

  // 初始化TensorFlow Lite
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
    // 计算与上一个样本的差值总和
    int prev_index = (buffer_index == 0) ? SEQUENCE_LENGTH - 1 : buffer_index - 1;
    for (int i = 0; i < NUM_SENSORS; i++) {
      difference += abs(current_reading[i] - sensor_buffer[prev_index][i]);
    }
    
    // 更新环形缓冲区
    for (int i = 0; i < NUM_SENSORS; i++) {
      sensor_buffer[buffer_index][i] = current_reading[i];
    }
    buffer_index = (buffer_index + 1) % SEQUENCE_LENGTH;
    
    // *** DEBUGGING: 取消下面这行的注释来观察传感器变化值 ***
    // Serial.println(difference); 

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
          // 当手势持续一段时间后，如果动作变缓（趋于静止），则触发推理
          if (difference < STATIC_THRESHOLD) {
            Serial.println("Movement finished, starting inference...");
            
            // 准备推理数据
            float inference_buffer[SEQUENCE_LENGTH * NUM_SENSORS];
            int current_pos = buffer_index;
            for (int i = 0; i < SEQUENCE_LENGTH; i++) {
                for (int j = 0; j < NUM_SENSORS; j++) {
                    inference_buffer[i * NUM_SENSORS + j] = sensor_buffer[current_pos][j];
                }
                current_pos = (current_pos + 1) % SEQUENCE_LENGTH;
            }

            // CRITICAL: Apply the same StandardScaler normalization used during training.
            // The model will not work correctly without this step.
            // z = (x - mean) / scale
            for (int i = 0; i < SEQUENCE_LENGTH; i++) {
              for (int j = 0; j < NUM_SENSORS; j++) {
                  int index = i * NUM_SENSORS + j;
                  inference_buffer[index] = (inference_buffer[index] - scaler_mean[j]) / scaler_scale[j];
              }
            }

            // 执行推理
            float output_data[NUM_GESTURES];
            perform_inference(inference_buffer, output_data);
            
            // 处理推理结果
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
              Serial.println("️⚠️ No gesture recognized.");
            }

            // *** MODIFICATION 2: Set the cooldown timer when transitioning TO cooldown state ***
            current_state = STATE_COOLDOWN;
            cooldown_start_time = millis(); 
          }
        }
        break;

      case STATE_COOLDOWN:
        // Wait for a fixed cooldown period to prevent immediate re-triggering
        // and allow the user to return to a neutral position.
        if (millis() - cooldown_start_time > 2000) { // Cooldown for 2 seconds
           Serial.println("Cooldown finished. Waiting for new gesture...");
           current_state = STATE_IDLE;
        }
        break;
    }
  }
}

// ================================================================================================
//                                    TensorFlow Lite 函数
// ================================================================================================

void setup_tf() {
  Serial.println("DEBUG: Setting up TensorFlow Lite...");
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided is schema version not equal to supported version!");
    while (true);
  }
  Serial.println("DEBUG: Model loaded and version checked.");

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  Serial.println("DEBUG: Interpreter created.");

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true);
  }
  Serial.println("DEBUG: Tensors allocated.");

  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("DEBUG: Input and output tensors set.");
}

void perform_inference(float* input_data, float* output_data) {
    if (!input || !output) {
        Serial.println("Input/Output tensor not ready!");
        return;
    }

    // 根据你的模型输入类型进行修改，这里假设是 INT8 量化模型
    for (int i = 0; i < SEQUENCE_LENGTH * NUM_SENSORS; i++) {
        input->data.int8[i] = static_cast<int8_t>((input_data[i] / input->params.scale) + input->params.zero_point);
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed");
        return; // 只是返回，而不是让程序卡住
    }

    // 根据你的模型输出类型进行修改
    for (int i = 0; i < NUM_GESTURES; i++) {
        int8_t y_quantized = output->data.int8[i];
        output_data[i] = static_cast<float>(y_quantized - output->params.zero_point) * output->params.scale;
    }
}