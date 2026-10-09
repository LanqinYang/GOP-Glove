#ifndef GOP_CONFIDENCE_GATE_H
#define GOP_CONFIDENCE_GATE_H
// branch: 0 ADANN, 1 LightGBM, -1 Static fallback.
struct GateDecision { int label; int branch; int gate_case; };
static inline GateDecision manuscript_gate(const float* a, const float* b,
                                           float ta=0.5f, float tb=0.5f) {
  int ya=0, yb=0;
  for(int i=1;i<11;++i) { if(a[i]>a[ya]) ya=i; if(b[i]>b[yb]) yb=i; }
  bool ha=a[ya]>=ta, hb=b[yb]>=tb;
  if(ha && hb && ya==yb) return {yb,1,1};
  if(ha && !hb) return {ya,0,2};
  if(hb && !ha) return {yb,1,2};
  if(ha && hb) {
    float second_a=-1.0f, second_b=-1.0f;
    for(int i=0;i<11;++i) {
      if(i!=ya && a[i]>second_a) second_a=a[i];
      if(i!=yb && b[i]>second_b) second_b=b[i];
    }
    return a[ya]-second_a > b[yb]-second_b ? GateDecision{ya,0,3} : GateDecision{yb,1,3};
  }
  return {10,-1,4};
}
#endif
