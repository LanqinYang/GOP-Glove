1. Introduction and Literature Review
Abstract: An end‑to‑end embedded system for British Sign Language (BSL) digit recognition is presented, built with self‑made graphite‑on‑paper strain sensors, a reduced‑wire pull‑down readout on Arduino Nano 33 BLE Sense Rev2 (hereafter “Arduino”), and a rigorously validated modelling pipeline. The glove provides five analogue channels at 50 Hz; signals are acquired via a deterministic start/stop serial protocol, then resampled to fixed‑length windows and standardized. To address cross‑subject variability under device constraints, this work proposes a hybrid method (DA‑LGBM) that couples a domain‑adversarial encoder with a compact tree classifier suitable for microcontrollers. Evaluation follows Leave‑One‑Subject‑Out as the primary protocol for enhancing robustness. Results indicate improved cross‑subject Accuracy and Macro‑F1 compared with baselines, alongside real‑time sliding‑window inference and low memory usage on Arduino. The hardware design, acquisition protocol, and deployment path are intentionally simple to facilitate reproducibility and align with the stated Aim and Objectives (O1–O5).

1.1 Rationale: Instability of DIY Flexible Sensors and Constraints for Edge AI

In Human–Computer Interaction (HCI) and wearable sensing, low-cost piezoresistive sensors (e.g., graphite-on-paper on gloves) offer privacy and low power, but their sensitivity comes with instability. Material creep and hysteresis yield path-dependent responses and slow recovery; temperature and humidity add baseline drift. Tape-based attachment induces adhesive pre-strain and micro-slippage on curved substrates; breadboard jumpers and shared grounds lower Signal-to-Noise Ratio (SNR) and cause intermittent dropouts [1–4]. Across users and sessions these effects present as domain shift, degrading generalization.

Tiny Machine Learning (TinyML) enables on-device inference on Microcontroller Units (MCUs), yet SRAM/Flash budgets and operator availability in TensorFlow Lite for Microcontrollers (TFLM) constrain feasible architectures: dynamic memory usage and attention/batch‑matmul operators are not supported in standard TFLM builds on Arduino‑class devices. The challenge is thus twofold: unstable, shifting data and tight deployment envelopes. These are addressed by combining domain‑robust representation learning with a deployment‑friendly decision model and by evaluating offline accuracy alongside online latency and throughput under streaming windows.

Human and protocol variability also matter. Strap tension and placement shift baselines and gains; co‑articulation between adjacent fingers introduces cross‑channel coupling; non‑standard or prematurely terminated gestures alter dynamics. This study focuses on per‑channel SNR and baseline drift, and treats cross‑subject dispersion as a proxy for domain‑shift severity. These measurements inform windowing and normalization, and motivate domain‑robust encoders with deployment‑friendly decision models (LOSO primary).

1.2 Aim

Design, implement, and rigorously validate an end-to-end British Sign Language (BSL) digit-gesture system that is robust to DIY sensor instability and meets real-time constraints on Arduino-class MCUs. The core approach is DA-LGBM, pairing Adversarial Discriminative Domain Adaptation (ADANN) for domain-invariant features with a LightGBM classifier for efficient, MCU-friendly inference.

1.3 Objective

O1 – Sensor & readout. Build a GoP resistive array with reduced-wire readout (pull-down resistor configuration); identify and quantify instability sources (adhesive stress/slippage, breadboard contact variability, hysteresis/creep, temperature drift) using operational metrics [1–4].

O2 – Instability metrics. Characterize per-channel SNR, drift rate, hysteresis loop area and cross-session/subject dispersion; link to design choices (windowing, normalization, feature selection).

O3 – Task & protocol. BSL 11 classes (digits 0–9 plus Static); five ADC channels at 50 Hz; each sequence resampled to 100×5. Evaluation uses Leave-One-Subject-Out (LOSO) as primary, with a standard 64/16/20 train/validation/test split as a secondary; report Accuracy and Macro-F1 with dispersion.

O4 – Methods & training. Under a unified pipeline, compare 1D-CNN, Transformer-Encoder, XGBoost, LightGBM, ADANN and DA-LGBM; use hyperparameter optimization only during development, then fix hyperparameters for final training and reporting.

O5 – TinyML deployment. Apply Post-Training Quantization (PTQ) to INT8 and 30–60% pruning; export a C/C++ representation/header; report RAM/Flash footprints and streaming-window inference latency.

O6 – Real-time viability. Demonstrate sliding-window online inference (win = 100, step = 10); report online accuracy, latency mean/variance, jitter and false-positive rate.

1.4 Related Work
1.4.1 Engineering Challenges and Characterization

GoP devices can be fabricated within minutes, enabling agile placement; the granular percolation networks that confer sensitivity also produce hysteresis and creep. Environmental drift, adhesive mechanics and contact variability further reduce SNR and create non‑stationary baselines across minutes, sessions and users [1–4]. For wearable recognition, static gauge factors or qualitative loop plots are insufficient. This work quantifies operational metrics that inform modelling: (i) per‑channel SNR across users/sessions; (ii) baseline drift trends; and (iii) cross‑session/subject dispersion as a proxy for domain‑shift severity [1–4]. These map to concrete choices: large drift → standardization and short‑window statistics; wide cross‑subject spread → domain‑robust encoders.

1.4.2 Domain Adaptation and Cross-Subject Generalization

Domain shift between source (training) and target (deployment) distributions is the dominant failure mode in wearable Human Activity Recognition (HAR). Domain‑Adversarial Neural Networks (DANN) attach a domain head to a shared encoder and use a Gradient Reversal Layer (GRL) so the encoder learns class‑discriminative yet domain‑agnostic features [8]. In HAR/biosignals, systematic studies show adversarial/UDA methods can recover substantial cross‑user performance without target labels, while warning about evaluation pitfalls: user leakage in random splits, tuning that peeks at target data and reporting point estimates without dispersion [9,10]. Consequently, LOSO is primary to mirror “new user” deployment, and a standard 64/16/20 split serves as a secondary baseline to quantify the optimism gap. This work adopts consistent windowing, balanced batches, class‑aware augmentation and calibrated handling of the Static class. The pipeline separates a research path (adversarial encoder) from a deployment path (low‑variance statistical features with a classical classifier) to preserve robustness within MCU limits [8–10].

1.4.3 TinyML Deployment: Operator Limits and Practical Alternatives

On MCUs, SRAM must hold buffers, activations and state; Flash stores program and parameters. PTQ to INT8 yields ~4× parameter compression and efficient integer kernels; pruning reduces arithmetic and memory with accuracy trade-offs dependent on redundancy and calibration [11,12]. In standard TFLM builds, attention/batch‑matmul operators and large dynamic tensors are unavailable, which excludes typical Transformer encoders on Arduino‑class devices [5,6].

Tree-boosting models (LightGBM, XGBoost) provide an alternative: inference reduces to comparisons and additions, mapping to branch‑heavy C/C++ with header‑only embedding and code‑generation [13–16]. The trade‑off is representational power: trees rely on features rather than learning invariances from raw sequences. This motivates hybrids: learn domain‑invariant embeddings where compute allows, then deploy a compact decision model; or train with two feature sources—adversarial embeddings for analysis and window‑level statistics (per‑channel mean/std/min/max) that remain stable under quantization and sensor noise. From a systems view, success needs end‑to‑end accounting: model size, peak SRAM, and streaming‑window latency/throughput across MCU/CPU/GPU [5–7,11–16].

1.4.4 This Work in Context

Prior work tends to optimize either recognition accuracy (deep encoders) or deployability (classical models on hand-crafted features). Few present a principled bridge tested against DIY instability and MCU limits. My approach, DA-LGBM, is that bridge:

Problem framing: treat instability metrics (SNR, drift, hysteresis loop area, cross-session/subject dispersion) as design signals for windowing, normalization and feature choice.

Modeling choice: couple an adversarial encoder per DANN/ADANN to learn domain‑invariant representations with a LightGBM classifier efficient to embed on constrained devices [8,13].

Terminology: we refer to the method as DA‑LGBM. Its implementation in the codebase appears under the module name `ADANN_LightGBM`.

Evaluation stance: LOSO is primary; the standard 64/16/20 split is secondary. Central tendency is reported with dispersion and ablations on augmentation, trial budget and epochs [9,10].

Deployment realism: INT8 PTQ and 30–60% pruning (TF models only), C/C++ header export, and latency/throughput measured on MCU/CPU/GPU [5–7,11–16]. LightGBM/DA‑LGBM tree branches do not use INT8 quantization.

This argument runs from instability evidence → domain-robust representation → deployment-stable statistics → on-device decision-making, evaluated under user-realistic splits. In short, DA-LGBM reconciles cross-subject robustness with tiny-device feasibility: adversarial features explain why generalization holds [8–10], tree inference ensures how it runs on MCUs [13–16], and the measurement suite tests whether it remains usable given DIY sensor realities [1–4].

2. Methodology

This chapter describes the sensor–glove build, acquisition and control, the data processing and modelling pipeline, the DA‑LGBM method, and the deployment and evaluation protocol on a resource‑constrained microcontroller. The main experiments use only the five analogue channels from the graphite‑on‑paper sensors.

2.1 Sensor and Data Acquisition System



Graphite was rubbed on both sides of an A4 sheet with an 8B pencil to form conductive layers. At each sensing site, stripped copper wire was stitched through the paper three times and taped; DuPont connectors on the other side connect the signal. The strip sensors were taped to the back of the glove, wired to a breadboard, and connected to the Arduino.

Pull-down readout (reduced wiring)

- Single-ended divider: the sensor Rs is tied to Vref, a pull-down Rpd to GND, and the divider node goes to the ADC (10‑bit, Vref = Vcc).
- c/1023 = Rpd/(Rpd + Rs) ⇒ Rs = Rpd·(1023 − c)/c.
- Firmware reports c′ = 1023 − c so that larger counts monotonically indicate larger resistance (more bending).

Sequential sampling and motivation for resampling

- Channels are sampled sequentially and serial I/O adds overhead; over a nominal 2 s at 50 Hz, effective frame count is slightly lower and timestamps are not perfectly uniform. All sequences are linearly resampled to a fixed length N = 100.

PC↔Arduino serial protocol (start/stop)

- Deterministic framing via explicit tokens: the host sends "S" (start) and "X" (stop); the Arduino streams CSV records with timestamps. The host performs per‑subject/gesture segmentation and persists raw streams for reproducibility.

2.2 Data Processing and Modelling

Resampling (linear interpolation)

alpha_t = (t/(N−1))·(T−1)
k = floor(alpha_t)
delta = alpha_t − k
x′(t) = (1 − delta)·x(k) + delta·x(k+1)

Standardization (fit on training fold)

x_tilde = (x − mu_train) / sigma_train

Lightweight augmentation (p_aug = 0.3)

add-noise:        x′ = x + ε,  ε ~ N(0, 0.005^2)
amplitude-scale:  x′ = α·x,   α ~ U[0.98, 1.02]
time-warp:        t′ = g(t) (piecewise‑linear monotone), segment speeds s_i ∈ [1/2, 2], x′ = x ∘ g^{-1}

Models and training

Six models are considered: 1D‑CNN, Transformer‑Encoder, XGBoost, LightGBM, ADANN, and the hybrid DA‑LGBM. All models are trained on fixed‑length windows with early stopping based on validation Macro‑F1. Hyperparameter optimization is used during development only; final training uses fixed hyperparameters.

Validation strategy (robustness emphasis)

LOSO is primary (cross‑subject generalization); a standard 64%/16%/20% split is secondary (baseline comparison).

Accuracy = (1/n) · Σ 1(pred_i = y_i)
Macro‑F1 = (1/C) · Σ_c 2·Prec_c·Rec_c/(Prec_c + Rec_c)

2.3 Core Method: DA‑LGBM (ADANN encoder + LightGBM classifier)

ADANN (domain adversarial encoder).
min_{F,C} max_{D}  L = L_ce(C(F(x)), y) − λ·L_dom(D(F(x)), d),
realized via a Gradient Reversal Layer (GRL) so that F learns gesture‑discriminative yet domain‑agnostic features. After training we obtain embeddings z = F(x) for analysis.

LightGBM (multiclass classifier).
L_lgbm = − Σ_i log( softmax_k( f_k(ξ_i) )[y_i] ) + Ω(Trees),
where ξ is either z (analysis path) or compact window‑level statistics φ(x) (deployment path).

End‑to‑end DA‑LGBM.
During development we validate domain invariance by training LightGBM on z; for deployment on resource‑constrained MCUs, we compute φ(x) per window and train LightGBM on φ(x) to obtain a small, C‑embeddable predictor. A flowchart of the training/deployment pipeline will be provided in the figure; full step‑by‑step pseudocode is given in the Appendix.
For desktop/mobile evaluation, an optional probability‑weighted ensemble of ADANN and LightGBM can be used, with the weight determined on a validation set; implementation details and formulas are provided in the Analysis section.

Algorithm 1 — DA‑LGBM (training and deployment)

Inputs: labelled windows {(x_i, y_i, d_i)}, target length N, augmentation parameters (p_aug, σ, α_min, α_max, s_min, s_max).
Outputs: header‑only predictor ŷ = f(φ(x)).

1  Preprocess:  resample to N; x_tilde = (x − μ_train)/σ_train
2  Augment (with prob p_aug):
      x ← x + N(0, 0.005^2)
      x ← α·x,  α ~ U[0.98, 1.02]
      x ← x ∘ g^{-1},  segment speeds s_i ∈ [1/2, 2]
3  Train ADANN:
      minimise  L_ce(C(F(x)), y) − λ·L_dom(D(F(x)), d)   (implemented via GRL)
4  Embed:      z = F(x)
5  LightGBM (analysis):  train on z, minimise L_lgbm(z)
6  Statistics:  φ(x) = [μ, σ, min, max]×5
7  LightGBM (deploy):    train on φ(x), minimise L_lgbm(φ)
8  Prune (TF‑only):  s(t) = s0 + (s1 − s0)·((t − t0)/(T − t0))^p,  s0=0.30 → s1=0.60
                     (strip pruning applied after training)
9  Quantise (TF‑only, INT8 affine):
      q = round(x/s) + z
      x_hat = s·(q − z)
10 Export:     generate C/C++ headers (DA‑LGBM/LightGBM as C trees; TF models as embedded TFLite binaries);
               MCU path computes φ(x) then performs tree lookup to output ŷ
11 Evaluate:   modes {LOSO, Standard} × {Full, Arduino};
               platforms Arduino / CPU / A100; warm‑up then repeated runs;
               report mean/variance/p99/FPS and model/header sizes.


Complexity: 统计 O(CN)；树推断 O(trees×depth)；MCU 峰值内存≈滑窗缓冲 + 20D 累加器 + 树表。

2.4 Deployment and Evaluation Protocol

Deployment scope
- On Arduino, the Transformer is not executed due to TFLM operator and dynamic‑tensor limits. DA‑LGBM (tree branch) and a compact 1D‑CNN (pruned + INT8) serve as MCU baselines. Tree models (including DA‑LGBM on MCU) use per‑window 20‑D statistics at inference; neural models are exported as embedded TFLite binaries for MCU‑capable ops.

Evaluation overview
- Primary protocol: Leave‑One‑Subject‑Out (LOSO). Secondary: stratified hold‑out (64/16/20) where space permits.
- Metrics: Accuracy and Macro‑F1 with dispersion across folds; confusion matrices for class‑wise analysis; optional confidence analysis.

Performance benchmarking (desktop/edge)
- Arduino: edge inference latency (sliding window).
- Colab/desktop CPU: single‑threaded latency aligned to MCU settings.
- NVIDIA A100: GPU throughput for completeness.

Real‑time testing
- Live gesture recognition with streaming windows on Arduino.
- Latency measurements (ms) and throughput (windows/s).

Cross‑platform comparison
- Model/header size (KB); inference latency comparison (CPU vs Arduino).
- Accuracy‑efficiency trade‑offs and deployment feasibility.

Details of measurement protocols (e.g., warm‑up counts, repeats, threading, quantized inputs) are provided in the Analysis section to keep the methodology within word limits.

3. Results, Analysis, Findings and Conclusions

3.1 Sensor Instability Characterisation

3.1 传感器不稳定性表征

Scope and motivation. We quantify how DIY graphite-on-paper (GoP) sensors behave under real use and why this creates domain shift across users/sessions. The analysis targets four operational metrics: (i) per-channel SNR distributions; (ii) baseline drift (slope/variance) over minutes; (iii) hysteresis loop shape/area under ramp–hold; and (iv) cross-session/subject dispersion. The “Static” pose is treated as a stress test for baseline consistency.
范围与动机。 本节量化 DIY 石墨纸（GoP）传感器在真实使用中的表现，并解释其为何导致跨用户/会话的分布偏移（domain shift）。我们围绕四类可操作指标展开：（i）通道级 SNR 分布；（ii）分钟尺度的基线漂移（斜率/方差）；（iii）在 ramp–hold 条件下的滞回曲线与面积；（iv）跨会话/被试离散度。将“静态”姿势当作基线一致性的压力测试。

Metric definitions. SNR is computed over fixed windows as 
10
log
⁡
10
(
𝑃
signal
/
𝑃
noise
)
10log
10
	​

(P
signal
	​

/P
noise
	​

), where “signal” is within-gesture variance and “noise” is residual after a short-term smoother; drift uses robust linear fits per channel/session to obtain slope and its dispersion; hysteresis loop area is measured under controlled flex–hold–release; dispersion is summarised by Coefficient of Variation (CV) across users/sessions. Micro-dropouts are flagged by derivative/level heuristics.
指标定义。 在固定窗口上计算 SNR：
10
log
⁡
10
(
𝑃
signal
/
𝑃
noise
)
10log
10
	​

(P
signal
	​

/P
noise
	​

)，其中“信号”为手势内变异，“噪声”为经短期平滑后的残差；漂移通过会话/通道的稳健线性回归得到斜率及其离散；滞回面积在受控的弯曲–保持–释放下测量；跨会话/被试离散用变异系数（CV）汇总。用导数与幅值启发式检测瞬时掉点（micro-dropouts）。

Cross-user variability (Fig. 1). On six users and eleven gestures, the mean CV is 0.1369, indicating decent overall consistency. Gesture 2 is most consistent (CV=0.131), while Static is most variable (CV=0.153). The latter is counter-intuitive but expected with GoP: individuals’ natural resting postures differ, so a universal “zero” baseline is suboptimal—user-specific zero calibration is advisable. Data balance per gesture/user is adequate; the CV histogram is unimodal with a mild right tail.
跨用户差异（图1）。 在 6 名被试、11 个手势上，平均 CV=0.1369，整体一致性较好。手势 2 最稳定（CV=0.131），而静态最不稳定（CV=0.153）。这虽反直觉，却符合 GoP 现实：个体静息姿态差异显著，统一“零点”不理想——建议进行用户级零点标定。数据在手势/被试间较均衡；CV 直方图为单峰、右尾略长。

Drift behaviour (Fig. 2). All five channels exhibit systemic negative drift without input. Ch2 is the least stable with the widest slope/variance; Ch4 is the most stable. Drift magnitude correlates with slope magnitude, confirming sustained baseline creep rather than random wander. This hardware limitation requires software drift compensation or frequent recalibration to maintain long-term integrity.
漂移特性（图2）。 五个通道在无输入时均存在系统性负漂移。Ch2 最不稳定（斜率/方差最宽），Ch4 最稳定。漂移幅值与斜率大小正相关，说明是持续基线蠕变而非纯随机游走。该硬件局限需要软件漂移补偿或更频繁的重标定来保证长期有效性。

Signal quality/SNR (Fig. 3). Ch3 shows a near-0 dB SNR across users—consistent with a hardware fault—yielding a bimodal overall SNR distribution. Functional channels typically exceed 20 dB. SNR vs gesture shows 0/1/2 consistently degrade quality across channels, hinting at mechanical design issues (strain, compression or kinking during those motions). Ch3 needs inspection/replacement; glove mechanics and wiring strain relief should be revised.
信号质量/SNR（图3）。 Ch3 的 SNR 接近 0 dB，各被试一致，表明存在器件级故障，导致总体 SNR 呈双峰。其余正常通道通常 >20 dB。按手势的 SNR 显示 0/1/2 在各通道上显著劣化，提示手套存在机械设计问题（相关动作引发过应变、压缩或折皱）。Ch3 需检修/更换；同时应改进手套结构与走线应力释放。

Static pose as instability probe (Fig. 4). Aggregating static windows highlights baseline dispersion and slow recovery after flex events. Between-user offsets dominate within-user noise, reinforcing the need for per-user standardisation and short-window statistics at inference.
以静态为不稳定性探针（图4）。 静态窗口汇总暴露出基线分散与屈曲后的缓慢恢复。跨用户偏置大于用户内噪声，进一步支持在推断时采用用户级标准化与短窗口统计量。

Engineering analysis (qualitative). Field notes and hardware inspection align with the metrics: (1) breadboard contacts and shared grounds inject intermittent contact resistance and common-impedance noise; (2) tape adhesion on curved fabric causes pre-strain and micro-slip, altering gains over time; (3) temperature/humidity shift material resistivity; (4) cable strain and bends create transient dropouts; (5) reduced-wire pull-down readout is sensitive to Rpd choice and grounding topology. Evidence includes wiring photos and logs; no simulation is used—this is an engineering diagnosis.
工程分析（定性）。 现场记录与硬件检查与上述指标一致：（1）面包板接触与共地引入瞬时接触电阻与公共阻抗噪声；（2）织物曲面上的胶带粘附导致预应力与微滑移，使增益随时间变化；（3）温湿度改变材料电阻率；（4）走线受力与折弯造成瞬断；（5）降连线的下拉读出对 Rpd 选值与接地拓扑较敏感。证据为连线照片与日志；不做仿真，定位为工程诊断。

Implications for modelling. Large drift → per-fold standardisation and short-window statistics; cross-subject spread → domain-robust encoders; frequent dropouts → fault-tolerant features over phase-sensitive ones. We therefore favour DA-LGBM: adversarial encoders for domain invariance, plus tree inference on compact window-level statistics for MCU deployment.
对建模的启示。 漂移大 → 按折标准化与短窗口统计；跨被试分散大 → 域鲁棒编码器；掉点频繁 → 选用容错特征优于相位敏感特征。由此我们采用 DA-LGBM：以对抗式编码获得域不变表示，在 MCU 端用窗口统计 + 树推断实现部署。

Limitations and scope. We did not isolate environmental confounds (temperature/humidity) or run Allan variance in the main text (optional in supplement). The characterisation focuses on operational choices, not material science.
局限与范围。 本节未单独隔离环境因素（温湿度），主文未给出 Allan 方差（可选放补充）。我们的侧重点是工程可操作选择，而非材料机理。

3.2 Recognition Performance and Ablations

3.2 识别性能与消融

Evaluation setup. We report LOSO-Full results on six users and eleven classes (ten digits + Static). Models are trained per fold with fixed hyper-parameters and no peeking into held-out users; early stopping is applied on the training fold. Macro-F1 is primary; Accuracy is secondary. Per-fold points + mean ±95% CI are shown (df=5, CI from per-fold SD). Paired Wilcoxon tests compare each model to DA-LGBM, with Holm–Bonferroni correction. Results are CPU (non-Arduino) and must not be mixed with on-device numbers.
评估设置。 报告 LOSO-Full（6 被试、11 类：10 个数字 + 静态）。各折在固定超参下训练，严格不偷看留出被试；用训练折早停。Macro-F1 为主指标，Accuracy 为辅。图中给出每折散点 + 折均值±95%CI（df=5，由折内 SD 换算）。与 DA-LGBM 的Wilcoxon 配对检验使用 Holm–Bonferroni 校正。结果为 CPU（非 Arduino），不得与板端结果混用。

Main results (LOSO-Full). DA-LGBM achieves the highest Macro-F1 and Accuracy on average, with tighter CIs than CNN/boosting baselines. ADANN is runner-up, consistent with the hypothesis that domain-invariant features improve cross-user robustness. Transformer Encoder/XGBoost/LightGBM form a middle tier; 1D-CNN trails due to larger fold-to-fold variance (hard subjects). This supports our design: domain-robust representation + lightweight decision generalises best under cross-subject shift.
主要结果（LOSO-Full）。 DA-LGBM 在 Macro-F1 与 Accuracy 上平均最优，且 CI 更紧；ADANN 位列第二，符合域不变表示增强跨用户鲁棒性的假设。Transformer/XGBoost/LightGBM 居中；1D-CNN 因跨折方差较大而略逊（困难被试）。这印证了我们的设计：域鲁棒表示 + 轻量决策在跨被试偏移下泛化更好。

Uncertainty and significance. With n=6 folds, CIs are of moderate width reflecting genuine cross-subject dispersion. Wilcoxon contrasts favour DA-LGBM directionally; after Holm correction, several pairings do not reach p<0.05. We therefore emphasise effect direction and CI overlap over binary significance, a conservative stance under small-n paired designs.
不确定性与显著性。 在 n=6 的设定下，CI 宽度中等，主要反映跨被试离散。Wilcoxon 与 DA-LGBM 的对比在方向上占优；经 Holm 校正后，部分对比未达 p<0.05。故我们更强调效应方向与 CI 重叠，这是小样本配对设计下更稳健的解读。

Hard classes and confusions. Aggregated confusion shows two recurring errors: (i) Static vs low-amplitude digits (0/1) under weak activation; (ii) fine-grained digits with similar dynamics (e.g., 3 vs 4). Both trace back to §3.1: low SNR on certain motions and co-articulation across fingers reduce separability. Models with domain-robust encoders (DA-LGBM/ADANN) show fewer Static-related false positives, implying mitigation of user-specific baseline drift and amplitude scaling.
难分类与混淆。 汇总混淆显示两类常见错误：(i) 弱激活下的静态与低幅度数字（0/1）互混；(ii) 细粒度数字（如 3 与 4）因动态相近而混淆。两者均可追溯至 §3.1：某些动作 SNR 偏低，以及手指协同运动降低了可分性。具备域鲁棒编码的模型（DA-LGBM/ADANN）在静态相关误报更少，说明其缓解了用户特异的基线漂移与幅值缩放。

Why DA-LGBM helps. DA-LGBM trains an adversarial encoder to suppress domain cues, then performs inference with a compact tree on stable window-level statistics (μ/σ/min/max×5). Under LOSO, this decouples user-specific offsets from gesture dynamics while preserving simple decision boundaries that generalise. The tighter CIs indicate lower sensitivity to fold outliers, unlike CNNs that entangle raw-sequence artefacts from DIY sensors.
DA-LGBM 的优势机理。 先用对抗式编码器抑制域线索，再以紧凑树模型基于稳定的窗口统计量（μ/σ/最小/最大×5）做推断。LOSO 下，这种搭配将用户偏置与手势动态解耦，同时保持易于泛化的简单边界。更紧的 CI 表明其对异常折更不敏感；相比之下，CNN 更容易受 DIY 传感器的原始序列伪影影响。

Sanity check (Standard split). A stratified 64/16/20 split (not plotted) yields uniformly higher scores than LOSO-Full across models—evidence of optimism from user leakage. Hence LOSO remains primary for deployment-realistic evaluation.
稳健性检查（Standard 划分）。 分层 64/16/20（未绘制）在各模型上均高于 LOSO-Full，体现了用户泄漏带来的乐观偏差。因此部署现实性的结论仍以 LOSO 为主。

Ablations. Three factors are ablated on DA-LGBM: augmentation, search budget (n-trials), and epochs. Turning augmentation off consistently hurts Macro-F1, especially on high-drift folds—light noise/warp is beneficial. Increasing n-trials (e.g., 50→200) shows diminishing returns once early stopping stabilises; extending epochs (50→100) gives marginal gains. Kalman filtering is excluded: pilot runs showed negligible/negative cross-user impact, likely from suppressing informative transients and interacting poorly with per-fold standardisation.
消融。 在 DA-LGBM 上考察 数据增强、搜索预算（n-trials）与训练轮数（epochs）。关闭增强会稳定地降低 Macro-F1，且在漂移较大的折上更明显——轻量噪声/时间扰动是有益的。n-trials（50→200）在早停稳定后呈现收益递减；**epochs（50→100）**增益也很小。Kalman 滤波未纳入最终方案：早期试验显示其对跨用户泛化收效甚微甚至为负，可能因为抑制了有判别力的瞬态并与按折标准化相冲突。

Takeaways. (i) Under LOSO-Full, DA-LGBM provides the best centre-of-mass performance with tighter uncertainty; (ii) domain-invariance matters more than deeper sequence encoders when sensor instability drives domain shift; (iii) light augmentations help while Kalman is unnecessary; (iv) CPU-side results should not be mixed with Arduino-mode deployment trade-offs (reported separately).
要点。 (i) 在 LOSO-Full 下，DA-LGBM 兼具最优中心趋势与更紧的不确定度；(ii) 当传感器不稳定主导分布偏移时，域不变表示比更深的序列编码更关键；(iii) 轻量增强有效，而 Kalman 无必要；(iv) CPU 侧结果与 Arduino 模式的部署权衡须分开解读（单列报告）。

3.3 Deployment and Latency

Deployment focuses on Arduino‑class MCUs. DA‑LGBM (tree branch) runs on 20‑D window statistics with a header‑only C implementation; a compact 1D‑CNN (pruned + INT8) serves as a neural MCU baseline, whereas the Transformer is excluded due to TFLM operator/dynamic‑tensor limits. Cross‑platform benchmarking includes Arduino, single‑threaded desktop CPU (settings aligned to MCU execution), and GPU (A100) for throughput bounds. Reported indicators are: model/header size (KB), mean latency (ms), p99 latency (ms), and throughput (windows/s). Measurement procedures (warm‑up, repetition counts, threading and quantization alignment) follow the protocol outlined in Methodology and are detailed in the Analysis section.

3.4 Findings and Conclusions

Key findings are summarized with respect to the Aim and Objectives. Typical observations include: (i) cross‑subject robustness of DA‑LGBM relative to baselines under LOSO; (ii) feasibility of real‑time sliding‑window inference on Arduino with bounded memory; (iii) the contribution of lightweight augmentation to generalization; and (iv) diminishing returns beyond moderate optimization budgets. These findings motivate the hybrid design as a principled bridge between domain‑robust representation learning and deployable decision models on resource‑constrained hardware.