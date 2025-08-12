#include <Arduino.h>
#include <math.h>
#include "bsl_model_LightGBM.h"

// Board config
const int SENSOR_PINS[] = {A0, A1, A2, A3, A4};
const int NUM_SENSORS = 5;
const int SEQUENCE_LENGTH = 100;

// Latency config
static const int RUNS_DEFAULT = 200;
static const int WARMUP_DEFAULT = 10;

// Buffers
static float window_fixed[SEQUENCE_LENGTH][NUM_SENSORS];
static bool has_fixed_window = false;
// Precomputed 20-dim features (mean/std/min/max per channel) for inference-only timing
static float features_fixed[BSL_MODEL_FEATURES];
static bool has_features = false;

// EMA filter
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
  Serial.println("  record                - capture one 100x5 window and precompute features");
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

static void precompute_features() {
  static float ema_buf[SEQUENCE_LENGTH][NUM_SENSORS];
  ema_filter(window_fixed, ema_buf);
  for (int ch = 0; ch < NUM_SENSORS; ++ch) {
    float minv = ema_buf[0][ch], maxv = ema_buf[0][ch];
    double sum = 0.0, sum2 = 0.0;
    for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
      float v = ema_buf[t][ch];
      if (v < minv) minv = v; if (v > maxv) maxv = v;
      sum += v; sum2 += (double)v * (double)v;
    }
    float mean = (float)(sum / SEQUENCE_LENGTH);
    float var = (float)(sum2 / SEQUENCE_LENGTH - (double)mean * (double)mean);
    float stdv = var > 0 ? sqrtf(var) : 0.0f;
    int base = ch * 4; features_fixed[base+0] = mean; features_fixed[base+1] = stdv; features_fixed[base+2] = minv; features_fixed[base+3] = maxv;
  }
  has_features = true;
}

static void run_latency(int measure_runs, int warmup_runs) {
  if (!has_features) { Serial.println("ERR: No fixed features. Run 'record' first."); return; }
  measure_runs = constrain(measure_runs, 1, 1024);
  warmup_runs = constrain(warmup_runs, 0, 1024);

  static uint32_t t_pred[1024];

  // warmups (not recorded)
  for (int i = 0; i < warmup_runs; ++i) { (void)predict_gesture(features_fixed); }

  int eff = 0;
  for (int i = 0; i < measure_runs; ++i) {
    uint32_t ti0 = micros(); (void)predict_gesture(features_fixed); uint32_t ti1 = micros();
    t_pred[eff++] = ti1 - ti0;
  }

  struct Stats s; compute_stats(t_pred, measure_runs, &s);
  double latency_ms = s.avg_v / 1000.0; double throughput = 1000000.0 / s.avg_v;
  Serial.print("{\"latency_ms\":"); Serial.print(latency_ms, 2);
  Serial.print(",\"throughput_ips\":"); Serial.print(throughput, 2);
  Serial.println("}");
}

void setup() {
  Serial.begin(115200); while (!Serial){}
  for (int i = 0; i < NUM_SENSORS; i++) pinMode(SENSOR_PINS[i], INPUT);
  Serial.println("LightGBM Inference-Only Latency Test Ready. Type 'record' then 'latency'.");
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
    } else { Serial.println("Unknown command. Type 'help'."); }
  }
}
