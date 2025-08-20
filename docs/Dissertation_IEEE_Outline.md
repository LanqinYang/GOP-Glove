# From Unstable Graphite-on-Paper Flexible Sensors to Real-Time Robust TinyML: DA-LGBM and an End-to-End Pipeline for BSL Gesture Recognition

本文件给出一份遵循 IEEE 论文风格的完整写作大纲（章节结构、关键内容要点、建议字数映射、图表清单），用于指导最终报告写作，使其“格式在很大程度上遵循惯例”。章节编号与标题采用 IEEE 常见体例（I–VIII）。

## Abstract（150–200 词）
- 背景：DIY 石墨纸柔性传感器 + Data Glove，真实环境不稳定（滞回/蠕变、温漂、胶带应力、面包板接触不稳、跨用户/会话偏移）。
- 目标：在 TinyML 资源约束下实现跨被试鲁棒的 BSL 手势识别与实时推理。
- 方法：提出混合方法 DA-LGBM（ADANN + LightGBM），端到端管线（采集→训练→导出→Arduino 演示）。
- 结果：在 Arduino-LOSO 场景下，DA-LGBM 优于基线（示例：72%/78%/82%）；部署延迟与体积满足约束；在线演示稳定。
- 结论：从“不稳定传感器”到“可用系统”的可复现范式，具备工程与研究价值。

## Index Terms（5–8 个）
TinyML, domain adaptation, LightGBM, flexible sensors, BSL gesture recognition, real-time inference, Arduino, data glove.

## I. Introduction（合并 Related Work 总字数≈1200）
- 研究动机（Rationale）：真实佩戴下的不稳定性 + TinyML 部署约束，现有方法难兼顾泛化与可部署性。
- 任务定义与数据模态：BSL 手势识别，11 类（0–9 + Static），5 通道/50 Hz，序列重采样至 100×5。
- 研究空白：公开数据集与稳定硬件假设不适配 DIY；端侧协议与公平评测缺失。
- 贡献（Bullets）：
  1) 提出 DA-LGBM（域鲁棒特征 + 树分类），在不稳定传感器 + TinyML 条件下取得更好泛化与部署可行性；
  2) 端到端工程管线，含 Arduino 模式的剪枝+INT8 导出与公平的延迟协议；
  3) 传感器不稳定性定量表征（SNR、漂移、滞回、跨会话/跨用户一致性）与其对方法选择的影响；
  4) 实时在线演示作为“可用性证据”。

## II. Related Work（放在 I 后半段或独立章节，二选一）
- 柔性/应变传感器不稳定性：噪声、漂移、滞回、装配/佩戴差异；表征方法与缓解策略。
- 手势识别模型：CNN/Transformer 与树模型；跨主体泛化难点。
- 域适应：DANN/ADANN 的优势与局限；GRL 等实现。
- TinyML 部署：INT8 量化、剪枝、TFLM 算子限制（Transformer 不导出）。
- HPO：作为开发期工具；最终用固定超参复训与评测（无需重跑 HPO）。

（注：若学校要求合并，可将 II 融入 I 的“Related Work”小节，保证总字数≈1200。）

## III. System and Sensor Design（建议 300–400，用于 Methodology 映射）
- 机械与材料：Graphite-on-paper 传感轨迹布局（五指），电极与接触面处理，防潮/防汗包覆（胶带/热缩管）。
- 电气与连线：面包板 + 跳线至 Nano33 BLE A0–A4；供电 3.3V；串口 115200；S/X 控制；强调“面包板接触不稳与应力导致的瞬断/接触电阻变化”。
- 佩戴与基线：胶带固定引入的应力与滑移；佩戴后静置 3–5 秒建立基线，通道对齐与会话复位。
- 不稳定来源→工程对策映射表：滞回/蠕变、温漂、胶带应力、连线接触不稳、绑带松紧差异 → 基线重置、轻量平滑、标准化、域鲁棒特征 + 端上统计特征。

## IV. Methodology（≈800，对应 Methodology）
- 数据处理：重采样 100×5；轻量增强（AddNoise/TimeWarp/Amplitude-Scaling）；StandardScaler。
- 模型族概述：1D_CNN、Transformer_Encoder、XGBoost、LightGBM、ADANN、DA-LGBM（核心）。
- 核心方法 DA-LGBM（简式数学）：
  - 目标：L = L_ce(C(F(x)), y) + λ L_dom(D(F(x)), d)；GRL 实现 min_{F,C} max_{D}；
  - 双路径：研究用 z=F(x) 训练树；部署用 φ(x)∈R^{20}（每通道均值/标准差/最小/最大）。
- 训练协议：Standard vs LOSO；开发期使用 Optuna（TPE+MedianPruner，同预算/验证协议）选择最终超参；报告固定超参、随机种子与版本，复现实验不要求重跑 HPO。
- 部署与评测协议：剪枝（30→60% 多项式）+ INT8 PTQ；头文件导出与复制；Arduino/CPU/A100 延迟协议；实时滑窗（窗=100、步=10、2 分钟 × 3 轮）。

## V. Experimental Setup（建议 300–400）
- 数据集与采集：用户数/会话数/采样策略与命名规范；数据清洗准则。
- 评估指标与统计：Accuracy、Macro-F1（LOSO）；显著性检验（如 Wilcoxon）与效应量；CI 估计方法。
- 训练与超参：给出“固定超参表”（每模型一行）；trial/epoch 预算；硬件/软件版本。

## VI. Results（对应 3 章的主体，≈3500）
- 传感器不稳定性表征：SNR 箱线图、漂移趋势、滞回回线与面积、跨会话/跨用户箱线；讨论对方法选择/部署的影响。
- 识别性能（LOSO）：六模型主表（Acc、Macro-F1、CI）；统计显著性与效应量；混淆矩阵与错误分析。
- 消融：增强开关、n_trials（50/200）、epochs（50/100）、Kalman 移除（-1.52% 退化的实证）。
- 部署与延迟：模型与头文件大小；Arduino/CPU/A100 延迟与吞吐；精度-延迟-尺寸权衡。
- 实时演示：DA-LGBM vs 基线（1D_CNN/ADANN）在线准确率、延迟均值/方差、抖动、误报率、姿态保持稳定性。

## VII. Discussion（≈500）
- 意义：在不稳定传感器 + TinyML 场景下，DA-LGBM 的方法学与工程价值。
- 局限：Transformer 不导出；能耗/功耗未测；数据规模与外部有效性；HPO 的随机性（已通过固定超参与重复训练控制）。
- 未来：更系统标定与在线校准；能耗/功耗；更强域适应策略与更大数据集；更丰富的实时场景。

## VIII. Conclusion（100–150）
- 总结 DA-LGBM + 端到端管线的主要发现与可用性证据；指出可复现要素（固定超参/版本/工件）。

## Acknowledgment（可选）

## References（IEEE 风格）
- 采用 IEEE 引文格式；按出现顺序编号。

## Appendix（可选）
- 公式与伪代码细节（可引用 `docs/Dissertation_Techniques.md` 的内容进行改写粘贴）。

---

### 图表与表格清单（最少集）
- 传感器与系统：
  - Fig.1 系统总览图（硬件→训练→导出→测试）；
  - Fig.2–3 手套/传感器实物与连线图（含引脚映射/受力方向示意）。
- 传感器表征：
  - Fig.4 SNR 箱线图；Fig.5 漂移趋势；Fig.6 滞回回线与面积统计；Fig.7 跨会话/跨用户箱线图。
- 识别与消融：
  - Tab.1 LOSO 主表（六模型）；Fig.8 混淆矩阵；Fig.9 Optuna 收敛示意（可选，标注“开发期”）；Tab.2 消融对比表。
- 部署与延迟：
  - Tab.3 模型/头文件大小；Fig.10 三平台延迟对比；Fig.11 精度-延迟-尺寸关系图。
- 实时演示：
  - Fig.12 在线时序/箱线图；Tab.4 在线指标对比（DA-LGBM vs 基线）。

### 6 天执行计划（匹配 IEEE 结构）
- Day 1：拍摄/绘制 Fig.2–3；写 I（含 BSL 11 类）与 II 提纲。
- Day 2：采集并生成 Fig.4–7；写 III 与 IV（方法摘要与协议）。
- Day 3：汇总 Tab.1、Fig.8、Fig.9（可选）；写 VI 的识别部分。
- Day 4：完成消融 Tab.2；整理 Tab.3、Fig.10–11；写 VI 的部署部分。
- Day 5：做实时演示，产出 Fig.12、Tab.4；完成 VI 收尾、VII、VIII。
- Day 6：统一图表风格与编号；参考文献格式化（IEEE）；全文润色与查漏补缺。


