#include <Arduino.h>
#include <math.h>
#include "bsl_model_XGBoost_Arduino.h"

const int SENSOR_PINS[] = {A0, A1, A2, A3, A4};
const int NUM_SENSORS = 5;
const int SEQUENCE_LENGTH = 100;
const int NUM_GESTURES = 11;

static const int RUNS_DEFAULT = 200;
static const int WARMUP_DEFAULT = 10;

static float window_fixed[SEQUENCE_LENGTH][NUM_SENSORS];
static bool has_fixed_window = false;

static float features_fixed[NUM_SENSORS * 4];
static bool has_features = false;

const float EMA_ALPHA = 0.1f;

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

// Simple stats
struct Stats { uint32_t min_v, max_v; double avg_v; };
static void compute_stats(uint32_t* arr, int n, struct Stats* s) {
  if (n <= 0) { s->min_v = s->max_v = 0; s->avg_v = 0; return; }
  uint32_t mn = UINT32_MAX, mx = 0; unsigned long long sum = 0ULL;
  for (int i = 0; i < n; ++i) { uint32_t v = arr[i]; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
  s->min_v = mn; s->max_v = mx; s->avg_v = (double)sum / (double)n;
}

static void cmd_help() {
  Serial.println("Commands:");
  Serial.println("  record              - capture one 100x5 window as fixed input and precompute features");
  Serial.println("  latency             - run latency on fixed features (default 200 runs, warmup 10)");
  Serial.println("  mem                 - print RAM composition/estimate (JSON)");
}

static void capture_window(float dst[SEQUENCE_LENGTH][NUM_SENSORS]) {
  for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
    for (int ch = 0; ch < NUM_SENSORS; ++ch) {
      dst[t][ch] = analogRead(SENSOR_PINS[ch]);
    }
    delay(20);
  }
}

static void precompute_features() {
  static float ema_buf[SEQUENCE_LENGTH][NUM_SENSORS];
  ema_filter(window_fixed, ema_buf);
  for (int ch = 0; ch < NUM_SENSORS; ++ch) {
    float minv = ema_buf[0][ch], maxv = ema_buf[0][ch];
    double sum = 0.0, sum2 = 0.0;
    for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
      float v = ema_buf[t][ch];
      if (v < minv) minv = v;
      if (v > maxv) maxv = v;
      sum += v; sum2 += (double)v * (double)v;
    }
    float mean = (float)(sum / SEQUENCE_LENGTH);
    float var = (float)(sum2 / SEQUENCE_LENGTH - (double)mean * (double)mean);
    float stdv = var > 0 ? sqrtf(var) : 0.0f;
    int base = ch * 4;
    features_fixed[base + 0] = mean;
    features_fixed[base + 1] = stdv;
    features_fixed[base + 2] = minv;
    features_fixed[base + 3] = maxv;
  }
  has_features = true;
}

// Memory report (RAM composition for XGBoost)
static void cmd_mem() {
  const int BYTES_PER_FLOAT = (int)sizeof(float);
  const int window_bytes = SEQUENCE_LENGTH * NUM_SENSORS * BYTES_PER_FLOAT;   // window_fixed
  const int features_bytes = (NUM_SENSORS * 4) * BYTES_PER_FLOAT;             // features_fixed
  const int ema_buf_bytes = SEQUENCE_LENGTH * NUM_SENSORS * BYTES_PER_FLOAT;  // ema_buf inside precompute_features
  const int t_pred_bytes = 1024 * (int)sizeof(uint32_t);                      // t_pred in run_latency

  const unsigned long estimated_peak = (unsigned long)window_bytes
    + (unsigned long)features_bytes
    + (unsigned long)ema_buf_bytes
    + (unsigned long)t_pred_bytes;

  Serial.print("{\"buffers_bytes\":{\"window_fixed\":"); Serial.print(window_bytes);
  Serial.print(",\"features_fixed\":"); Serial.print(features_bytes);
  Serial.print(",\"ema_buf\":"); Serial.print(ema_buf_bytes);
  Serial.print(",\"t_pred\":"); Serial.print(t_pred_bytes);
  Serial.print("}");
  Serial.print(",\"estimated_ram_peak_bytes\":"); Serial.print(estimated_peak);
  Serial.println("}");
}

static void run_latency(int runs, int warmup) {
  if (!has_features) { Serial.println("ERR: No fixed features. Run 'record' first."); return; }
  static uint32_t t_pred[1024];
  runs = constrain(runs, 1, 1024); warmup = constrain(warmup, 0, runs - 1);

  static Eloquent::ML::Port::XGBClassifier clf;

  for (int i = 0; i < warmup; ++i) { (void)clf.predict(features_fixed); }

  int eff = 0;
  for (int i = 0; i < runs; ++i) {
    uint32_t t0 = micros();
    int pred = clf.predict(features_fixed);
    (void)pred;
    uint32_t t1 = micros();
    if (i >= warmup) t_pred[eff++] = t1 - t0;
  }

  struct Stats s_pred; compute_stats(t_pred, eff, &s_pred);
  double latency_us = s_pred.avg_v; double throughput = 1000000.0 / s_pred.avg_v;
  Serial.print("{\"latency_us\":"); Serial.print(latency_us, 2);
  Serial.print(",\"throughput_ips\":"); Serial.print(throughput, 2);
  Serial.println("}");
}

void setup() {
  Serial.begin(115200); while (!Serial) {}
  for (int i = 0; i < NUM_SENSORS; ++i) pinMode(SENSOR_PINS[i], INPUT);
  Serial.println("XGBoost LOSO Inference-Only Latency Test Ready. Type 'record' then 'latency'.");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n'); line.trim();
    if (line.equalsIgnoreCase("help")) { cmd_help(); }
    else if (line.equalsIgnoreCase("record")) { capture_window(window_fixed); has_fixed_window = true; precompute_features(); Serial.println("OK: recorded & precomputed features"); }
    else if (line.startsWith("latency")) {
      int runs = RUNS_DEFAULT, warm = WARMUP_DEFAULT; int sp = line.indexOf(' ');
      if (sp > 0) { int sp2 = line.indexOf(' ', sp + 1); if (sp2 > 0) { runs = line.substring(sp + 1, sp2).toInt(); warm = line.substring(sp2 + 1).toInt(); } else { runs = line.substring(sp + 1).toInt(); } }
      run_latency(runs, warm);
    } else if (line.equalsIgnoreCase("mem")) { cmd_mem(); }
    else { Serial.println("Unknown command. Type 'help'."); }
  }
}
