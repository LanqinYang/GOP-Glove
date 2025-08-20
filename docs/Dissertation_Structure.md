## Dissertation 结构与写作要点（含 BSL 手势与 11 类说明）
Title: From Unstable Graphite-on-Paper Flexible Sensors to Real-Time Robust TinyML: DA-LGBM and an End-to-End Pipeline for BSL Gesture Recognition
本文件给出四章结构（含建议字数）、每章要写的关键点、必备图表/表格清单、当前缺口与两周行动计划。已将 RAO 与 BSL 手势识别内容融合到 Introduction。

### 1. Introduction and Literature Review（≈1200 字）
- Rationale（100–150）
  - DIY 柔性/应变传感器在真实佩戴中的不稳定性：材料滞回/蠕变、温漂、绑带差异、ADC/供电噪声、跨用户/会话偏移。
  - TinyML 端侧资源受限，现有高精深度模型难部署、可部署模型难跨主体泛化。
- Aim（50–80）
  - 构建并验证一套面向不稳定 DIY 传感器的端到端 BSL 手势识别系统，兼顾跨被试鲁棒性与 Arduino 实时推理，核心方法为 DA-LGBM（Hybrid ADANN + LightGBM）。
 - Objectives（150–200，条目化）
  - O1 传感器设计与读出原理：提出并实现 graphite-on-paper 柔性电阻式传感器与“减线”读出架构；量化不稳定来源（胶带粘附应力、面包板接触不稳、滞回/蠕变、温漂）。
  - O2 传感器不稳定性量化：SNR、漂移率、滞回、跨会话/跨用户一致性，并与方法设计关联。
  - O3 任务与数据协议：定义 BSL 11 类（0–9+静止）、5 通道/50 Hz、重采样 100×5 的协议与命名规范，支持 LOSO。
  - O4 方法与训练：在统一协议下比较 1D_CNN / Transformer_Encoder / XGBoost / LightGBM / ADANN / DA‑LGBM，采用 LOSO 评估跨主体泛化（HPO 仅为开发期工具，最终固定超参复训与评估）。
  - O5 TinyML 部署：Arduino 模式的剪枝（30→60%）与 INT8 量化、头文件导出与资源约束评估。
  - O6 实时可用性：在线滑窗演示（窗100/步10），报告在线准确率、延迟均值/方差、抖动与误报率。
- BSL 手势识别任务定义（200–300）
  - 场景与动机：BSL 数字手势在端侧的可用性价值与可穿戴应用。
  - 标签空间：11 类（数字 0–9 与静止态 Static）。
  - 输入模态与采样：5 通道模拟量、50 Hz；每次手势序列重采样到 100×5；文件命名携带 `user` 与 `gesture`（支持 LOSO）。
  - 评估指标：Accuracy、Macro-F1（LOSO 为主）。
- Related Work（700–850）
  - 传感器不稳定性与表征（SNR、漂移、滞回、跨主体差异）。
  - 域适应（DANN/ADANN）与跨主体泛化；CNN/Transformer 与传统树模型对比。
  - TinyML 部署：量化（INT8）、剪枝、多平台限制（TFLM 算子约束，Transformer 不导出）。
  - HPO 在开发期的角色：公平搜索（同预算/协议），最终报告固定超参用于复现。

### 2. Methodology（≈800 字）
- 2.1 传感器与采集系统（≈400）
  - 传感器设计（Graphite-on-Paper）：
    - 材料与结构：石墨涂层纸基，导电轨迹布局（拇/食/中/无/小指），电极接口与接触面处理；防潮/防汗考虑（如透明胶带/热缩管包覆）。
    - 机械集成：传感器通过胶带固定在手套指背/指腹位置；强调“胶带粘附引入的应力与滑移”。
  - 电气与连线：
    - 面包板跳线连接至 Nano 33 BLE A0–A4；标注引脚映射与地线汇聚；讨论“面包板接触不稳与应力导致的开路/接触电阻变化”。
    - 供电与 ADC：3.3V，50 Hz 采样；串口 115200；S/X 控制。
  - 会话与基线：当前实现未包含会话基线重置或 EMA 平滑；如需加入，可在采集端增加“静置校准”并在推理端加入轻量平滑（非本文默认设置）。
  - 不稳定来源与工程对策清单：材料滞回/蠕变、温漂、胶带粘附应力、面包板接触不稳、绑带松紧与滑移；本文采用“特征标准化、域鲁棒表征（ADANN）与端上统计特征（20 维）”缓解，未启用会话基线或 EMA。
- 2.2 数据处理与建模（≈300）
  - 预处理：重采样 100×5；轻量增强（AddNoise/TimeWarp/Amplitude-Scaling）；StandardScaler。
  - 关键预处理简式：重采样（线性插值）\(\alpha_t=\tfrac{t}{T-1}(N-1),\;k=\lfloor\alpha_t\rfloor,\;x'(t)=(1-\delta)\,x(k)+\delta\,x(k+1),\;\delta=\alpha_t-k\)；增强：\(x'=x+\epsilon,\;\epsilon\sim\mathcal{N}(0,\sigma^2)\) 与 \(x'=\alpha x,\;\alpha\sim\mathcal{U}[a,b]\)。
  - 六模型概述：1D_CNN、Transformer_Encoder、XGBoost、LightGBM、ADANN、DA-LGBM（核心）。
  - 训练协议：Standard vs LOSO；Optuna（TPE+MedianPruner，开发期）同预算/协议；最终固定超参训练，报告种子/版本。
  - 静止类与事件驱动推理（设计动机与协议设想）：训练时将 Static 作为第 11 类；推理设想为“空闲守门人”——当连续 \(M\) 个窗口满足 \(P(\text{Static})\ge \tau_s\) 保持空闲，仅当连续 \(K\) 个窗口满足 \(\max_{c\ne\text{Static}} P(c)\ge \tau_a\) 或变化度阈值（如 \(\mathrm{rolling\text{-}std}(x)\ge \sigma\) 或能量增量 \(\Delta E\ge \gamma\)）时触发识别。本文离线评测仍按统一窗口执行，事件驱动仅作为系统设计 rationale（不实现）。
- 2.3 核心方法 DA-LGBM（完整版数学表达，正文）
  - ADANN（域对抗）：
    - 目标：\( \min_{F,C} \max_{D} \, \mathcal{L} = \mathcal{L}_{ce}\big(C(F(x)), y\big) + \lambda \, \mathcal{L}_{dom}\big(D(F(x)), d\big) \)
    - 交叉熵：\( \mathcal{L}_{ce} = -\tfrac{1}{N} \sum_{n=1}^{N} \sum_{c=1}^{C} y_{nc}\log p_{nc}\), \( p = \mathrm{softmax}(C(F(x))) \)
    - GRL：对域损失的梯度在 \(F\) 端取反，实现 \( \min_F \max_D \)
  - LightGBM（多类 logloss，简式）：\( \mathcal{L}_{lgbm} = -\sum_i \log \mathrm{softmax}_k\big(f_k(\xi_i)\big)[y_i] + \Omega(\text{Trees}) \)
  - 融合（DA‑LGBM，两条特征路径）：
    - 研究路径：\( z = F(x) \)，\( \hat{y} = \arg\max_k f_{lgbm}(z) \)
    - 部署路径：\( \phi(x) \in \mathbb{R}^{20} \)（每通道均值/标准差/最小/最大），\( \hat{y} = \arg\max_k f_{lgbm}(\phi(x)) \)
    - 说明：部署选择 \( \phi(x) \) 以保证端侧稳定与可计算，并与训练协议一致
- 2.4 部署与评测协议（≈100）
  - 剪枝（30→60%）+ INT8 PTQ；导出头文件；Arduino/CPU/A100 延迟与吞吐测试（warmup+N runs）。量化/剪枝简式：\( q = \mathrm{round}(x/s)+z, \, \hat{x} = s(q- z); \, s(t)=s_0+(s_1-s_0)\big(\tfrac{t-t_0}{T-t_0}\big)^p \)

### 3. Results, Analysis, Findings and Conclusions（≈3500 字）
- 3.1 传感器不稳定性表征（≈800–1000）
  - SNR 箱线图（通道/用户/会话）；漂移趋势（可选 Allan 方差）；滞回回线与面积；跨会话/跨用户箱线图。
  - 结论：不稳定性→跨被试分布偏移→需要域鲁棒与端侧可控特征。
  - 工程讨论（定性）：面包板接触不稳与应力导致的瞬断/接触电阻变化，胶带粘附引发的受力与滑移；以电路照片/连线示意与短视频/日志片段（可选）作为证据；不做仿真，作为工程分析。
- 3.2 识别性能与消融（≈1200–1400）
  - 六模型 LOSO 主表（Acc/Macro-F1/CI）；统计检验（如 Wilcoxon，含效应量）。
  - 混淆矩阵与难分类手势分析。
  - 消融：增强开关、n_trials（50/200）、epochs（50/100）。说明：Kalman 滤波未纳入当前实现（早期试验显示收益不足）。
  - 模式说明（四模式取舍）：
    - 主表以 LOSO-full 为主（“robust”指跨被试泛化）；
    - Arduino-mode 的准确率仅作为“部署权衡”参考（不与 full 直接对比结论混用）；
    - standard-full 可放补充材料作为 sanity check；LOSO-arduino 非必须。
- 3.3 部署与延迟（≈800–1000）
  - 模型/头文件大小；三平台延迟与吞吐（Arduino vs Colab CPU vs Colab A100）；精度-延迟-尺寸权衡；Transformer 不导出的影响。
- 3.4 Findings & Conclusions（≈200–300）
  - 回答 H1–H4，凝练 3–5 条关键结论（DA-LGBM 跨被试占优、部署可行、增强贡献、HPO 效率）。

### 4. Discussion（≈500 字）
- 意义：从“不稳定传感器”到“可用系统”的工程闭环；混合范式在 TinyML 的价值。
- 局限：Transformer 不导出；能耗/功耗未测；数据规模与外部有效性。
- 未来：系统化标定与在线校准；能耗/功耗评测；更强域适应与更大数据集；更丰富的实时场景。

---

## 必备图表/表格清单（含 BSL）
- 传感器：SNR 箱线图、漂移趋势图、滞回回线与面积统计、跨会话/跨用户箱线图。
- 传感器设计与集成：
  - 手套与传感器实物照片（正/侧/特写）、电极/连线细节；
  - 石墨轨迹/电极布局示意图；
  - 面包板连线与引脚映射图；
  - 胶带固定位置与受力方向示意（展示“应力/滑移”）。
- BSL 任务：11 类标签示意/表（0–9 + Static），输入形态示意（100×5）。
- 识别：六模型 LOSO 主表、混淆矩阵、Optuna 收敛曲线、消融对比表。
- 部署：模型/头文件大小清单、三平台延迟图、精度-延迟-尺寸关系图。

## 当前缺口（需补齐）
- 传感器表征数据与脚本产物：`outputs/sensor_stats/` 下 CSV+图（SNR/漂移/滞回/CoV）。
- 消融实验：增强开关、n_trials/epochs 的表与图（不含 Kalman）。
- 部署对比：模型/头文件大小清单、三平台延迟图与关系图。
 
- 最终固定超参表：每模型一行（说明 HPO 仅用于开发期，不要求复跑）。

## 6 天行动计划（写作-实验双轨，压缩版）
- Day 1（硬件与引言）
  - 拍摄手套/传感器与连线照片；绘制布局/连线/受力示意图。
  - 写 Introduction：Rationale/Aim/Objectives + BSL 任务与 11 类；整理 Related Work 提纲。
- Day 2（表征与方法）
  - 采集短时稳定性数据（静止30s×3，会话×2，加载-卸载1循环/手指）。
  - 生成 SNR/漂移/滞回/CoV 图与表（`outputs/sensor_stats/`）。
  - 写 Methodology 2.1（传感器与采集）、2.4（评测协议）。
- Day 3（主结果）
  - 汇总六模型 LOSO 主表与混淆矩阵（已有结果优先，必要时补跑）。
  - 绘制 Optuna 收敛示意（可选）；开始写 Results 3.2。
- Day 4（消融与部署）
  - 最小消融：增强开关 + n_trials(50/200) 或 epochs(50/100) 二选一。
  - 导出模型/头文件大小清单；测 Arduino/CPU/A100 延迟并出图。
- Day 5（写作收敛）
  - 完成 Results 3.3 与 Findings，总体润色。
- Day 6（总成与检查）
  - 完成 Discussion；统一图表风格与编号；自检清单走查；参考文献与附录整理。

## 技术文档与实现指引（供写作引用）
- 已有：
  - `docs/Dissertation_RAO.md`（RAO 顶/中/底）
  - `docs/Dissertation_Techniques.md`（核心数学与伪代码，重点 DA-LGBM；附录含重采样/标准化）
  - `docs/RAO_Framework_Alignment.md`（RAO → 代码/数据/产物映射矩阵）
- 建议新增：
  - `docs/Sensor_Characterization_Protocol.md`（SNR/漂移/滞回采样与输出格式）
  - `docs/Realtime_Demo_Protocol.md`（在线滑窗协议与指标）
  - `docs/Final_Hyperparams.md`（最终固定超参表）


