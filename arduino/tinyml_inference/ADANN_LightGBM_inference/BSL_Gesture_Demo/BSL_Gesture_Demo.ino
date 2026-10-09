// === BSL Fusion Demo + Instrumentation ===
// Board: Arduino Nano 33 BLE Sense Rev2
// Window: 2s @ 50Hz -> 100x5 samples -> 190D features

#include <Arduino.h>
#include <math.h>
#include "bsl_model_ADANN.h"
#include "bsl_model_LightGBM.h"
#include "confidence_gate.h"

// ===== Config =====
#define FE_CH   5
#define FE_FS   50.0f
#define FE_WIN  100

#define USE_WAVELET 1          // 1=启用小波; 0=禁用
#define DEBUG_Z 0              // Z-score汇报
#define DEBUG_PROBS 0          // 单窗内部耗时汇报
#define START_MODE 0           // 0=串口触发, 1=按钮(D2)
#define POWER_MARKERS 1        // 1=print machine-readable phase markers

// 部署参数（仅作为信息输出，不改变当前触发流程）
#define ACQUISITION_SECONDS 2
#define ADANN_THRESHOLD 0.5f
#define LGBM_THRESHOLD 0.5f

enum FusionMode { WEIGHTED=0, CONF_GATE=1 };

// 输入翻转（训练时使用 1023-raw）
static const uint8_t CH_INV[FE_CH] = {1,1,1,1,1};

const int PIN_CH[FE_CH] = {A0, A1, A2, A3, A4};
static int16_t samples[FE_WIN][FE_CH];

static inline uint32_t tus() { return micros(); }
static inline float ms(uint32_t t0, uint32_t t1) { return (t1 - t0) / 1000.0f; }
static void phase_marker(const char* phase, const char* edge){
#if POWER_MARKERS
  Serial.print("PHASE,");
  Serial.print(phase);
  Serial.print(",");
  Serial.print(edge);
  Serial.print(",");
  Serial.println(millis());
#else
  (void)phase;
  (void)edge;
#endif
}

// 标签名（与训练一致）
const char* GESTURE_NAME[11]={"Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Static"};

// ---------- 前置声明 ----------
static bool   wait_for_trigger();
static void   countdown_321();
static float  acquire_window_ms();
static void   extract_190_features(float *feats190);
static void   features_per_channel_exact(const float *x, int n, float *o38);

// ---------- Instrumentation ----------
#define MAX_RUNS 256
struct Metrics {
  uint32_t windowing[MAX_RUNS];
  uint32_t standardize[MAX_RUNS];  // adann_std + lgbm_std
  uint32_t features[MAX_RUNS];
  uint32_t adann_infer[MAX_RUNS];
  uint32_t lgbm_infer[MAX_RUNS];
  uint32_t gate[MAX_RUNS];
  uint32_t uart_tx[MAX_RUNS];
  uint32_t inf_sum[MAX_RUNS];      // features + std + two inferences + gate + uart
  uint32_t e2e_sum[MAX_RUNS];      // windowing + inf_sum
  int N = 0;
  int disagree = 0;                // LGBM != ADANN 次数
} M;

struct TraceLast {
  uint32_t std_adann=0, inf_adann=0;
  uint32_t std_lgbm=0, inf_lgbm=0;
} T;

// 工具：拷贝并简单排序（N<=256，插入排序足够稳）
static void isort(uint32_t *a, int n){
  for(int i=1;i<n;++i){ uint32_t key=a[i]; int j=i-1; while(j>=0 && a[j]>key){ a[j+1]=a[j]; --j; } a[j+1]=key; }
}
static uint32_t percentile_u32(uint32_t *src, int n, float p){
  static uint32_t buf[MAX_RUNS];
  for(int i=0;i<n;++i) buf[i]=src[i];
  isort(buf,n);
  if(n==0) return 0;
  float pos = (p/100.f)*(n-1);
  int i0=(int)pos, i1=min(i0+1,n-1);
  float w=pos-i0;
  return (uint32_t)((1.f-w)*buf[i0] + w*buf[i1]);
}
static uint32_t vmax_u32(uint32_t *a, int n){ uint32_t m=0; for(int i=0;i<n;++i) if(a[i]>m) m=a[i]; return m; }

static void print_summary(){
  if(M.N==0){ Serial.println("No runs yet."); return; }
  Serial.println("\n==== Performance Summary ====");
  Serial.print("runs="); Serial.print(M.N);
  Serial.print("  acquisition="); Serial.print(ACQUISITION_SECONDS); Serial.println(" s");

  auto P3=[&](const char* name, uint32_t* arr){
    uint32_t md = percentile_u32(arr,M.N,50.f);
    uint32_t p95= percentile_u32(arr,M.N,95.f);
    uint32_t mx = vmax_u32(arr,M.N);
    Serial.print(name); Serial.print("  median="); Serial.print(md/1000.0f,3);
    Serial.print(" ms  p95="); Serial.print(p95/1000.0f,3);
    Serial.print(" ms  max="); Serial.print(mx/1000.0f,3); Serial.println(" ms");
  };

  // 总和
  P3("INF(sum)",   M.inf_sum);
  P3("E2E(sum)",   M.e2e_sum);

  // 分段
  P3("windowing",  M.windowing);
  P3("standardize",M.standardize);
  P3("features",   M.features);
  P3("adann_infer",M.adann_infer);
  P3("lgbm_infer", M.lgbm_infer);
  P3("gate",       M.gate);
  P3("uart_tx",    M.uart_tx);

  // 分歧率
  float disagree_ratio = (M.N>0)? (100.0f * (float)M.disagree / (float)M.N) : 0.f;
  Serial.print("disagree_ratio="); Serial.print(disagree_ratio,2); Serial.println("%");
  Serial.println("=============================\n");
}

// ---------- Trigger & Countdown ----------
static bool wait_for_trigger(){
#if START_MODE==0
  Serial.println("Press 's' + Enter to start. 'R' to report.");
  while (true){
    if (Serial.available()){
      int ch = Serial.read();
      if (ch=='R' || ch=='r'){ print_summary(); }
      else if (ch=='s' || ch=='S' || ch=='\n' || ch=='\r') return true;
    }
    delay(5);
  }
#else
  pinMode(2, INPUT_PULLUP);
  Serial.println("Press the button to start.");
  while (digitalRead(2)==HIGH) delay(2);
  return true;
#endif
}

static void countdown_321(){
  Serial.print("Starting in: 3"); delay(1000);
  Serial.print("\rStarting in: 2"); delay(1000);
  Serial.print("\rStarting in: 1"); delay(1000);
  Serial.print("\rGO!           \n");
}

// ---------- Acquire exactly 2s (100 frames @ 50 Hz) ----------
static float acquire_window_ms(){
  const uint32_t period_us = (uint32_t)(1000000.0f / FE_FS); // 20000 us
  uint32_t t0 = tus();
  uint32_t next = t0;
  for (int t=0; t<FE_WIN; ++t){
    for (int c=0; c<FE_CH; ++c){
      int v = analogRead(PIN_CH[c]);
      if (CH_INV[c]) v = 1023 - v;   // 训练同口径
      samples[t][c] = (int16_t)v;
    }
    next += period_us;
    while ((int32_t)(next - tus()) > 0) { /* spin */ }
  }
  uint32_t t1 = tus();
  return ms(t0, t1); // ~2000ms
}

// ---------- Utils ----------
static inline float safe_div(float a, float b){ return (fabsf(b) < 1e-10f) ? 0.f : (a / b); }
static inline int sgnf(float v){ return (v>0) - (v<0); }

// percentile helper
static float percentile(float *buf, int n, float p){
  static float tmp[FE_WIN];
  for(int i=0;i<n;++i) tmp[i]=buf[i];
  for(int i=1;i<n;++i){ float key=tmp[i]; int j=i-1; while(j>=0 && tmp[j]>key){ tmp[j+1]=tmp[j]; --j;} tmp[j+1]=key; }
  float pos = (p/100.0f) * (n-1);
  int i0 = (int)pos, i1 = i0 + 1;
  if (i1 >= n) return tmp[n-1];
  float w = pos - i0; return (1.f-w)*tmp[i0] + w*tmp[i1];
}

// ---------- Time-domain 18 ----------
static void time_domain_18(const float *x, int n, float *o){
  static float xs[FE_WIN];
  float mu=0.f; for(int i=0;i<n;++i) mu += x[i]; mu /= n;     // 原始均值（输出mean用它）
  for(int i=0;i<n;++i) xs[i]=x[i];

  float sum2=0, mn=1e30f, mx=-1e30f;
  for(int i=0;i<n;++i){ float v=xs[i]; sum2+=v*v; if(v<mn) mn=v; if(v>mx) mx=v; }
  float var=0.f; for(int i=0;i<n;++i){ float d=xs[i]-mu; var+=d*d; }
  var/=n; float std=sqrtf(var);

  static float tmp[FE_WIN]; for(int i=0;i<n;++i) tmp[i]=xs[i];
  for(int i=1;i<n;++i){ float k=tmp[i]; int j=i-1; while(j>=0 && tmp[j]>k){ tmp[j+1]=tmp[j]; --j;} tmp[j+1]=k; }
  float median=(n&1)? tmp[n/2] : 0.5f*(tmp[n/2-1]+tmp[n/2]);

  float m3=0,m4=0; if(std>1e-12f){ for(int i=0;i<n;++i){ float d=xs[i]-mu; float d2=d*d; m3+=d2*d; m4+=d2*d2; } m3/=n; m4/=n; }
  float skew=(std>1e-12f)? (m3/(std*std*std)) : 0.f;
  float kurt=(std>1e-12f)? (m4/(var*var)-3.f) : 0.f;
  if(!isfinite(skew))skew=0; if(!isfinite(kurt))kurt=0;

  float rms=sqrtf(sum2/n);
  float mav=0; for(int i=0;i<n;++i) mav+=fabsf(xs[i]); mav/=n;
  float wl=0; int zc=0; for(int i=1;i<n;++i){ wl+=fabsf(xs[i]-xs[i-1]); if(sgnf(xs[i])!=sgnf(xs[i-1])) ++zc; }
  int ssc=0; for(int i=1;i<n-1;++i){ if((xs[i]>xs[i-1]&&xs[i]>xs[i+1])||(xs[i]<xs[i-1]&&xs[i]<xs[i+1])) ++ssc; }

  float q1=percentile(tmp,n,25.f); for(int i=0;i<n;++i) tmp[i]=xs[i];
  float q3=percentile(tmp,n,75.f);
  float aav=0; for(int i=1;i<n;++i) aav+=fabsf(xs[i]-xs[i-1]); aav/=(n>1?(n-1):1);
  float iqr=q3-q1; float lb=q1-1.5f*iqr, ub=q3+1.5f*iqr; int outl=0;
  for(int i=0;i<n;++i) if(xs[i]<lb||xs[i]>ub) ++outl;

  int k=0; o[k++]=mu; o[k++]=std; o[k++]=var; o[k++]=mn; o[k++]=mx; o[k++]=(mx-mn); o[k++]=median; o[k++]=skew; o[k++]=kurt;
  o[k++]=rms; o[k++]=mav; o[k++]=wl; o[k++]=zc; o[k++]=ssc; o[k++]=q1; o[k++]=q3; o[k++]=aav; o[k++]=(float)outl;
}

// ---------- Frequency-domain 12 (density, detrend, one-sided) ----------
static void freq_domain_12(const float *x_in, int n, float fs, float *o){
  // detrend='constant'
  static float x[FE_WIN];
  float mean = 0.f; for(int i=0;i<n;++i) mean += x_in[i]; mean /= n;
  for(int i=0;i<n;++i) x[i] = x_in[i] - mean;

  const int K = n/2 + 1;
  static float psd[FE_WIN/2+1], fre[FE_WIN/2+1];

  for(int k=0;k<K;++k){
    float re=0, im=0;
    for(int t=0;t<n;++t){
      float ang = 2.0f*3.1415926535f*(k*(float)t/n);
      re += x[t]*cosf(ang);
      im -= x[t]*sinf(ang);
    }
    float p = (re*re + im*im) / (fs * n);    // density
    if (k>0 && k<K-1) p *= 2.0f;             // one-sided
    psd[k] = (p<0)?0:p;
    fre[k] = fs*k/n;
  }

  float P=0; for(int k=0;k<K;++k) P += psd[k];
  float sc=0, ss=0, se=0, m2=0, m3=0;
  if (P>1e-12f){
    static float pn[FE_WIN/2+1];
    for(int k=0;k<K;++k) pn[k]=psd[k]/P;
    for(int k=0;k<K;++k) sc += fre[k]*pn[k];
    for(int k=0;k<K;++k){
      float d = fre[k]-sc;
      ss += d*d*pn[k];
      if (pn[k]>1e-12f) se += -pn[k]*log2f(pn[k]);
      m2 += d*d*pn[k];
      m3 += d*d*d*pn[k];
    }
    ss = sqrtf(fmaxf(0.f, ss));
  }

  int imx=0; float b=psd[0]; for(int k=1;k<K;++k){ if(psd[k]>b){b=psd[k]; imx=k;} }
  float dom = fre[imx];

  float low=0, mid=0, high=0;
  for(int k=0;k<K;++k){ float f=fre[k]; if (f<=50.0f) low+=psd[k]; else if (f<=100.0f) mid+=psd[k]; else high+=psd[k]; }

  float sum2=0; for(int i=0;i<n;++i) sum2 += x_in[i]*x_in[i];
  float rms = sqrtf(sum2/n);
  float peak=0; for(int i=0;i<n;++i){ float a=fabsf(x_in[i]); if(a>peak) peak=a; }
  float pf = (rms>1e-10f)? (peak/rms) : 0.f;

  float var2=0; for(int i=0;i<n;++i){ float d=x_in[i]-mean; var2 += d*d; } var2/=n;
  float std = sqrtf(fmaxf(0.f,var2));
  float cv  = (fabsf(mean)>1e-10f)? (std/mean) : 0.f;

  int k=0; o[k++]=sc; o[k++]=dom; o[k++]=P; o[k++]=ss; o[k++]=se;
  o[k++]=low; o[k++]=mid; o[k++]=high; o[k++]=m2; o[k++]=m3; o[k++]=pf; o[k++]=cv;
}

// ---------- Wavelet ----------
static float ricker_normed_sample(int t, int M, float a){
  float u = (float)t - 0.5f*(M-1);
  float s = u / a;
  const float c = 2.0f / (sqrtf(3.0f*a) * powf(3.1415926535f, 0.25f));
  return c * (1.0f - s*s) * expf(-0.5f*s*s);
}
static void cwt_ricker_energy8(const float *x, int n, float *o){
  for(int ia=1; ia<=8; ++ia){
    float a = (float)ia;
    int M = min((int)(10.0f * a), n);
    static float w[FE_WIN];
    for(int t=0; t<M; ++t) w[t] = ricker_normed_sample(t, M, a);
    float e = 0.f;
    for(int t=0; t<n; ++t){
      float acc = 0.f;
      for(int k=0; k<M; ++k){
        int xi = t + k - M/2;
        if (xi>=0 && xi<n) acc += x[xi] * w[k];
      }
      e += acc * acc;
    }
    o[ia-1] = isfinite(e)? e : 0.f;
  }
}

// ---------- 每通道 38 维，拼 5 路=190 ----------
static void features_per_channel_exact(const float *x, int n, float *o38){
  float td[18]; time_domain_18(x,n,td);
  float fd[12]; freq_domain_12(x,n,FE_FS,fd);
  float wt[8];
#if USE_WAVELET
  cwt_ricker_energy8(x,n,wt);
#else
  for(int i=0;i<8;++i) wt[i]=0.f;
#endif
  int k=0; for(int i=0;i<18;++i)o38[k++]=td[i]; for(int i=0;i<12;++i)o38[k++]=fd[i]; for(int i=0;i<8;++i)o38[k++]=wt[i];
}

static void extract_190_features(float *feats190){
  for(int ch=0; ch<FE_CH; ++ch){
    static float buf[FE_WIN];
    for(int t=0;t<FE_WIN;++t) buf[t] = (float)samples[t][ch];
    float f38[38]; features_per_channel_exact(buf, FE_WIN, f38);
    for(int i=0;i<38;++i) feats190[ch*38 + i] = f38[i];
  }
}

// ---------- Models (计时版本) ----------
static inline void softmax_f(const float* x, int n, float* out){
  float m=x[0]; for(int i=1;i<n;++i) if(x[i]>m) m=x[i];
  float s=0; for(int i=0;i<n;++i){ out[i]=expf(x[i]-m); s+=out[i]; }
  float inv=1.f/s; for(int i=0;i<n;++i) out[i]*=inv;
}
static inline void softmax_d(const double* x, int n, double* out){
  double m=x[0]; for(int i=1;i<n;++i) if(x[i]>m) m=x[i];
  double s=0; for(int i=0;i<n;++i){ out[i]=exp(x[i]-m); s+=out[i]; }
  double inv=1.0/s; for(int i=0;i<n;++i) out[i]*=inv;
}

static inline int adann_predict_probs(const float* x190, float* probs11){
  uint32_t t0=tus();
  float x0[ADANN_INPUT_SIZE];
  for(int i=0;i<ADANN_INPUT_SIZE;++i) x0[i] = (x190[i] - adann_scaler_mean[i]) / adann_scaler_scale[i];
  uint32_t t1=tus();

  float h1[256]; matvec(fe1_w, fe1_b, x0, 256, ADANN_INPUT_SIZE, h1); for(int i=0;i<256;++i) h1[i]=relu(h1[i]);
  float h2[128]; matvec(fe2_w, fe2_b, h1, 128, 256, h2);              for(int i=0;i<128;++i) h2[i]=relu(h2[i]);
  float hf[ADANN_FEATURE_SIZE]; matvec(fe3_w, fe3_b, h2, ADANN_FEATURE_SIZE, 128, hf); for(int i=0;i<ADANN_FEATURE_SIZE;++i) hf[i]=relu(hf[i]);
  float g1[32]; matvec(gc1_w, gc1_b, hf, 32, ADANN_FEATURE_SIZE, g1); for(int i=0;i<32;++i) g1[i]=relu(g1[i]);
  float logits[ADANN_NUM_CLASSES]; matvec(gc2_w, gc2_b, g1, ADANN_NUM_CLASSES, 32, logits);
  uint32_t t2=tus();

  softmax_f(logits, ADANN_NUM_CLASSES, probs11);
  T.std_adann = t1 - t0;
  T.inf_adann = t2 - t1;

  int argm=0; float best=probs11[0];
  for(int i=1;i<ADANN_NUM_CLASSES;++i){ if(probs11[i]>best){ best=probs11[i]; argm=i; } }
  return argm;
}

static inline int lgbm_predict_probs(const float* x190, float* probs11){
  uint32_t t0=tus();
  double xz[BSL_MODEL_FEATURES];
  for(int i=0;i<BSL_MODEL_FEATURES;++i) xz[i] = (double)((x190[i] - scaler_mean[i]) / scaler_scale[i]);
  uint32_t t1=tus();

  double raw[11]; score(xz, raw);
  uint32_t t2=tus();

  double pd[11]; softmax_d(raw, 11, pd);
  for(int i=0;i<11;++i) probs11[i]=(float)pd[i];

  T.std_lgbm = t1 - t0;
  T.inf_lgbm = t2 - t1;

  int argm=0; float best=probs11[0];
  for(int i=1;i<11;++i){ if(probs11[i]>best){ best=probs11[i]; argm=i; } }
  return argm;
}

static inline int fuse_predict(const float* x190, float wA, float wL, float gate, FusionMode mode,
                               float* outp, int adann_top, int lgbm_top, uint32_t& gate_us, bool& disagree){
  uint32_t tg0=tus();
  float pa[11], pl[11];
  int pa_top = adann_predict_probs(x190, pa);
  int pl_top = lgbm_predict_probs(x190,  pl);

  disagree = (pa_top != pl_top);

  float pf[11];
  if (mode==WEIGHTED){ for(int i=0;i<11;++i) pf[i]=wA*pa[i]+wL*pl[i]; }
  else{
    GateDecision decision = manuscript_gate(pa, pl, ADANN_THRESHOLD, LGBM_THRESHOLD);
    for(int i=0;i<11;++i) pf[i] = decision.branch==0 ? pa[i] : (decision.branch==1 ? pl[i] : (i==10 ? 1.0f : 0.0f));
  }
  float s=0; for(int i=0;i<11;++i) s+=pf[i]; if(s>0){ float inv=1.f/s; for(int i=0;i<11;++i) pf[i]*=inv; }
  int pred=0; float best=pf[0]; for(int i=1;i<11;++i){ if(pf[i]>best){best=pf[i]; pred=i;} }
  if(outp) for(int i=0;i<11;++i) outp[i]=pf[i];
  uint32_t tg1=tus();
  gate_us = tg1 - tg0;

#if DEBUG_PROBS
  Serial.print("[Detail] stdA="); Serial.print(T.std_adann/1000.0f,3); Serial.print("ms, infA=");
  Serial.print(T.inf_adann/1000.0f,3); Serial.print("ms, stdL=");
  Serial.print(T.std_lgbm/1000.0f,3); Serial.print("ms, infL=");
  Serial.print(T.inf_lgbm/1000.0f,3); Serial.println("ms");
#endif
  return pred;
}

// ---------- Arduino main ----------
void setup(){
  Serial.begin(115200); while(!Serial);
  for(int c=0;c<FE_CH;++c) pinMode(PIN_CH[c], INPUT);
  Serial.println("BSL Fusion Ready.");
  Serial.print("acquisition="); Serial.print(ACQUISITION_SECONDS); Serial.println(" s");
}

void loop(){
  Serial.println("\n--- Ready ---");
  if (!wait_for_trigger()) return;
  phase_marker("cycle", "start");
  phase_marker("countdown", "start");
  countdown_321();
  phase_marker("countdown", "end");

  // 1) windowing
  phase_marker("acquisition", "start");
  uint32_t t_w0=tus();
  float t_acq=acquire_window_ms();
  uint32_t t_w1=tus();
  uint32_t windowing_us = t_w1 - t_w0;
  phase_marker("acquisition", "end");

  // 2) features
  phase_marker("features", "start");
  uint32_t t_fe0=tus(); float feats[190]; extract_190_features(feats); uint32_t t_fe1=tus();
  uint32_t features_us = t_fe1 - t_fe0;
  phase_marker("features", "end");

  // 3) fuse (包含各自标准化与推理时间，内部会写 T.std_*, T.inf_*)
  phase_marker("fusion", "start");
  uint32_t gate_us=0; bool disagree=false;
  float probs[11];
  int pred = fuse_predict(feats, 0.5f, 0.5f, 0.5f, CONF_GATE, probs, 0, 0, gate_us, disagree);
  phase_marker("fusion", "end");

  // 4) uart_tx（输出）
  phase_marker("uart", "start");
  uint32_t t_u0=tus();
  // 输出主结果
  Serial.print("Predicted = "); Serial.print(pred);
  Serial.print(" ("); Serial.print(GESTURE_NAME[pred]); Serial.println(")");
  // Top3
  int idx[11]; for(int i=0;i<11;++i) idx[i]=i;
  for(int i=0;i<11;++i) for(int j=i+1;j<11;++j) if(probs[idx[j]]>probs[idx[i]]){int t=idx[i]; idx[i]=idx[j]; idx[j]=t;}
  Serial.print("Top3: ");
  for(int k=0;k<3;++k){ int i=idx[k]; Serial.print(GESTURE_NAME[i]); Serial.print("="); Serial.print(probs[i],3); Serial.print(k==2?'\n':' '); }
  // Latency line（单窗）
  uint32_t standardize_us = T.std_adann + T.std_lgbm;
  Serial.print("[Latency] windowing="); Serial.print(windowing_us/1000.0f,3);
  Serial.print(" ms, features=");        Serial.print(features_us/1000.0f,3);
  Serial.print(" ms, standardize=");     Serial.print(standardize_us/1000.0f,3);
  Serial.print(" ms, ADANN=");           Serial.print(T.inf_adann/1000.0f,3);
  Serial.print(" ms, LGBM=");            Serial.print(T.inf_lgbm/1000.0f,3);
  Serial.print(" ms, gate=");            Serial.print(gate_us/1000.0f,3);
  Serial.println(" ms");
  uint32_t t_u1=tus();
  uint32_t uart_us = t_u1 - t_u0;
  phase_marker("uart", "end");

  // 5) 记录一次 run
  if (M.N >= MAX_RUNS) { M.N=0; M.disagree=0; } // 满了就覆盖（简单做法）
  M.windowing[M.N]   = windowing_us;
  M.features[M.N]    = features_us;
  M.standardize[M.N] = standardize_us;
  M.adann_infer[M.N] = T.inf_adann;
  M.lgbm_infer[M.N]  = T.inf_lgbm;
  M.gate[M.N]        = gate_us;
  M.uart_tx[M.N]     = uart_us;
  uint32_t inf_sum = features_us + standardize_us + T.inf_adann + T.inf_lgbm + gate_us + uart_us;
  uint32_t e2e_sum = windowing_us + inf_sum;
  M.inf_sum[M.N] = inf_sum;
  M.e2e_sum[M.N] = e2e_sum;
  if(disagree) M.disagree++;
  M.N++;

#if DEBUG_Z
  float mean_abs_z=0,max_abs_z=0;
  for(int i=0;i<190;++i){ float z=(feats[i]-scaler_mean[i])/scaler_scale[i]; float az=fabsf(z); mean_abs_z+=az; if(az>max_abs_z) max_abs_z=az; }
  Serial.print("Z(LGBM) mean|z|="); Serial.print(mean_abs_z/190.0f,2);
  Serial.print(" max|z|="); Serial.println(max_abs_z,2);
#endif
  phase_marker("cycle", "end");
}
