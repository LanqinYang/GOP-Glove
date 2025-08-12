#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include "bsl_model_1D_CNN.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Board config
const int SENSOR_PINS[] = {A0, A1, A2, A3, A4};
const int NUM_SENSORS = 5;
const int SEQUENCE_LENGTH = 100;

// Tensor arena (same as main sketch)
constexpr int kTensorArenaSize = 30 * 1024;
alignas(16) byte tensor_arena[kTensorArenaSize];

// TFLM globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Latency config
static const int RUNS_DEFAULT = 200;
static const int WARMUP_DEFAULT = 10;

// Buffers
static float window_fixed[SEQUENCE_LENGTH][NUM_SENSORS];
static bool has_fixed_window = false;

// Prepacked INT8 input buffer
static bool has_packed_input = false;

// EMA filter
const float EMA_ALPHA = 0.1f;

// Utils
static inline void ema_filter(const float in_[SEQUENCE_LENGTH][NUM_SENSORS], float out_[SEQUENCE_LENGTH][NUM_SENSORS]) {
  for (int ch = 0; ch < NUM_SENSORS; ++ch) {
    float ema = in_[0][ch];
    out_[0][ch] = ema;
    for (int t = 1; t < SEQUENCE_LENGTH; ++t) {
      ema = EMA_ALPHA * in_[t][ch] + (1.0f - EMA_ALPHA) * ema;
      out_[t][ch] = ema;
    }
  }
}

static inline void standardize_inplace(float buf[SEQUENCE_LENGTH][NUM_SENSORS]) {
  for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
    for (int ch = 0; ch < NUM_SENSORS; ++ch) {
      float v = buf[t][ch];
      float x = (v - scaler_mean[ch]) / scaler_scale[ch];
      buf[t][ch] = x;
    }
  }
}

// Stats
struct Stats { uint32_t min_v, max_v; double avg_v; };
static void compute_stats(uint32_t* arr, int n, struct Stats* s) {
  if (n <= 0) { s->min_v = s->max_v = 0; s->avg_v = 0; return; }
  uint32_t mn = UINT32_MAX, mx = 0; unsigned long long sum = 0ULL;
  for (int i = 0; i < n; ++i) { uint32_t v = arr[i]; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
  s->min_v = mn; s->max_v = mx; s->avg_v = (double)sum / (double)n;
}

// Commands
static void cmd_help() {
  Serial.println("Commands:");
  Serial.println("  record                - capture one 100x5 window and prepack INT8 input");
  Serial.println("  latency [R W]         - inference-only latency (default 200 10)");
}

static void capture_window(float dst[SEQUENCE_LENGTH][NUM_SENSORS]) {
  for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
    for (int ch = 0; ch < NUM_SENSORS; ++ch) {
      dst[t][ch] = analogRead(SENSOR_PINS[ch]);
    }
    delay(20);
  }
}

static void prepack_input() {
  static float ema_buf[SEQUENCE_LENGTH][NUM_SENSORS];
  ema_filter(window_fixed, ema_buf);
  standardize_inplace(ema_buf);
  int total = input->bytes; int idx = 0;
  for (int t = 0; t < SEQUENCE_LENGTH && idx < total; ++t) {
    for (int ch = 0; ch < NUM_SENSORS && idx < total; ++ch) {
      float f = ema_buf[t][ch];
      int8_t q = (int8_t)roundf(f / input->params.scale) + input->params.zero_point;
      input->data.int8[idx++] = q;
    }
  }
  while (idx < total) input->data.int8[idx++] = input->params.zero_point;
  has_packed_input = true;
}

static void run_latency(int measure_runs, int warmup_runs) {
  if (!has_packed_input) { Serial.println("ERR: No prepacked input. Run 'record' first."); return; }
  measure_runs = constrain(measure_runs, 1, 1024);
  warmup_runs = constrain(warmup_runs, 0, 1024);
  static uint32_t t_infer[1024];

  for (int i = 0; i < warmup_runs; ++i) { interpreter->Invoke(); }
  for (int i = 0; i < measure_runs; ++i) {
    uint32_t ti0 = micros(); interpreter->Invoke(); uint32_t ti1 = micros(); t_infer[i] = ti1 - ti0;
  }

  struct Stats s_infer; compute_stats(t_infer, measure_runs, &s_infer);
  double latency_ms = s_infer.avg_v / 1000.0; double throughput = 1000000.0 / s_infer.avg_v;
  Serial.print("{\"latency_ms\":"); Serial.print(latency_ms, 2);
  Serial.print(",\"throughput_ips\":"); Serial.print(throughput, 2);
  Serial.println("}");
}

// Setup & TFLM init
static void setup_tf() {
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) { Serial.println("Model schema mismatch!"); while (true); }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) { Serial.println("AllocateTensors() failed"); while (true); }
  input = interpreter->input(0); output = interpreter->output(0);
}

void setup() {
  Serial.begin(115200); while (!Serial);
  for (int i = 0; i < NUM_SENSORS; i++) pinMode(SENSOR_PINS[i], INPUT);
  setup_tf();
  Serial.println("1D_CNN Inference-Only Latency Test Ready. Type 'record' then 'latency'.");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n'); line.trim();
    if (line.equalsIgnoreCase("help")) { cmd_help(); }
    else if (line.equalsIgnoreCase("record")) { capture_window(window_fixed); has_fixed_window = true; prepack_input(); Serial.println("OK: recorded & prepacked"); }
    else if (line.startsWith("latency")) {
      int runs = RUNS_DEFAULT, warm = WARMUP_DEFAULT; int sp = line.indexOf(' ');
      if (sp > 0) { int sp2 = line.indexOf(' ', sp + 1); if (sp2 > 0) { runs = line.substring(sp + 1, sp2).toInt(); warm = line.substring(sp2 + 1).toInt(); } else { runs = line.substring(sp + 1).toInt(); } }
      run_latency(runs, warm);
    } else { Serial.println("Unknown command. Type 'help'."); }
  }
}

