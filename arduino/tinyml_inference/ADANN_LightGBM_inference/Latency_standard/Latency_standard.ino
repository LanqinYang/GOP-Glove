#include <Arduino.h>
#include "bsl_model_ADANN_LightGBM.h"   // LGB branch header (predict_gesture)
#include "bsl_model_ADANN.h"            // ADANN FP32 header (adann_predict)

static const int RUNS_DEFAULT = 200;
static const int WARMUP_DEFAULT = 10;

struct Stats { uint32_t min_v, max_v; double avg_v; };
static void compute_stats(uint32_t* arr, int n, struct Stats* s) {
  if (n <= 0) { s->min_v = s->max_v = 0; s->avg_v = 0; return; }
  uint32_t mn = UINT32_MAX, mx = 0; unsigned long long sum = 0ULL;
  for (int i = 0; i < n; ++i) { uint32_t v = arr[i]; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
  s->min_v = mn; s->max_v = mx; s->avg_v = (double)sum / (double)n;
}

static inline int infer_once() {
  float lgb_features[BSL_MODEL_FEATURES];
  for (int i = 0; i < BSL_MODEL_FEATURES; ++i) lgb_features[i] = 0.0f;
  int pred_lgb = predict_gesture(lgb_features);

  float adann_x[ADANN_INPUT_SIZE];
  for (int i = 0; i < ADANN_INPUT_SIZE; ++i) adann_x[i] = 0.0f;
  int pred_adann = adann_predict(adann_x);

  return (pred_adann == pred_lgb) ? pred_lgb : pred_adann;
}

static void run_latency(int measure_runs, int warmup_runs) {
  measure_runs = constrain(measure_runs, 1, 1024);
  warmup_runs = constrain(warmup_runs, 0, 1024);
  static uint32_t t_infer[1024];

  for (int i = 0; i < warmup_runs; ++i) (void)infer_once();
  for (int i = 0; i < measure_runs; ++i) {
    uint32_t t0 = micros();
    (void)infer_once();
    uint32_t t1 = micros();
    t_infer[i] = t1 - t0;
  }

  struct Stats s; compute_stats(t_infer, measure_runs, &s);
  double latency_ms = s.avg_v / 1000.0;
  double throughput = 1000000.0 / s.avg_v;

  Serial.print("{\"latency_ms\":"); Serial.print(latency_ms, 2);
  Serial.print(",\"throughput_ips\":"); Serial.print(throughput, 2);
  Serial.println("}");
}

void setup() {
  Serial.begin(115200); while(!Serial){}
  Serial.println("ADANN_LightGBM fusion latency tester ready. Type 'latency'.");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n'); line.trim();
    if (line.startsWith("latency")) {
      int runs = RUNS_DEFAULT, warm = WARMUP_DEFAULT; int sp = line.indexOf(' ');
      if (sp > 0) {
        int sp2 = line.indexOf(' ', sp + 1);
        if (sp2 > 0) { runs = line.substring(sp + 1, sp2).toInt(); warm = line.substring(sp2 + 1).toInt(); }
        else { runs = line.substring(sp + 1).toInt(); }
      }
      run_latency(runs, warm);
    } else {
      Serial.println("Type: latency [runs warmup]");
    }
  }
}
