# RAO → Project Framework Alignment

本文件将《docs/Dissertation_RAO.md》中的 Rationale / Aim / Objectives 映射到项目内现有代码、数据路径、产物与验证方式，便于执行与验收。README 不改动。

## 1) Objectives 可追溯矩阵（Traceability Matrix）

| Obj | 描述 | 代码/入口 | 数据/输入 | 产物/输出 | 验证/指标 |
| --- | --- | --- | --- | --- | --- |
| O1 | 5通道50Hz采集，文件命名规范，量化不稳定性（SNR/漂移/滞回/跨会话/跨用户） | `arduino/data_collection/sensor_data_collector/sensor_data_collector.ino`；`python -m src.data.data_collector {test|auto}` | `datasets/gesture_csv/`（user_…_gesture_…） | 采集CSV；传感器统计汇总（建议：`outputs/sensor_stats/*.csv`） | SNR、drift_rate、Hysteresis Ratio、CoV、（可选）Allan方差 |
| O2 | 预处理与规范化（重采样100×5、轻量增强、StandardScaler） | `src/training/pipeline.py`（`load_data`、`augment_data`） | O1 CSV | 预处理后特征（中间产物可选保存）；`scaler_*.pkl` | 训练/验证集一致性；无异常nan/shape |
| O3 | 六模型统一训练（Optuna + LOSO） | `run.py --model_type … [--loso]` | O2 数据 | `outputs/<Model>/<mode>/<opt>/` 报告与图；`models/trained/...` 模型与头文件 | Acc/Macro-F1（LOSO）；Optuna收敛；统计检验 |
| O4 | ADANN_LightGBM 优势验证与消融 | 同 O3 | 同 O3 | 消融结果（增强开关、n_trials/epochs、Kalman移除） | 性能差异与显著性（p值、效应量） |
| O5 | Arduino TinyML 部署与延迟 | `--arduino` 导出；Arduino `Latency_*.ino` | 训练生成的 `.h` / `.tflite` | `arduino/tinyml_inference/*/Latency_*/` 头文件；延迟日志 | Arduino/CPU/A100 延迟、吞吐、模型体积 |
| O6 | 可复现性与工程资产 | `requirements.txt`；`run.py`；固定随机种子 | - | 完整运行记录与版本信息 | 重复运行方差；环境可重建 |

注：O1 传感器统计建议新增脚本（可选）：`src/evaluation/sensor_characterization.py` 读取 `datasets/gesture_csv/` 输出 `outputs/sensor_stats/*.csv` 与图。

## 2) Milestones（里程碑）

1. 采集与规范（O1）
   - 完成多用户、多会话、多手势的 CSV 采集
   - 生成 SNR/漂移/滞回/CoV 汇总表
2. 预处理与基线（O2+O3）
   - 跑通 1D_CNN / LightGBM / XGBoost / Transformer_Encoder / ADANN / ADANN_LightGBM（standard & LOSO 至少一种）
3. 核心方法与消融（O4）
   - 补齐增强开关、n_trials/epochs、Kalman 对比
4. 部署与延迟（O5）
   - 产出 Arduino/CPU/A100 延迟与体积对比表
5. 复现与归档（O6）
   - 固定种子、版本锁定、导出清单与路径一致性检查

## 3) Evidence Registry（论文证据登记）

- 传感器稳定性：`outputs/sensor_stats/` 下 CSV 与图（SNR、漂移、滞回、跨用户箱线等）
- 识别性能：`outputs/<Model>/<mode>/<opt>/` 下 JSON/图表（LOSO 主表、混淆矩阵、收敛曲线）
- 模型与导出：`models/trained/<Model>/<mode>/<opt>/` 下 `.keras/.pkl/.tflite/.h` 与 `scaler_*.pkl`
- 延迟：Arduino 串口 JSON 日志（建议保存至 `outputs/latency/<board>/<Model>/...`）；CPU/A100 Notebook 输出图/表

## 4) Writing Mapping（写作落点）

- Introduction & Literature（1200字）：RAO + 相关工作（DIY传感器稳定性、域适应、TinyML部署）
- Methodology（800字）：O1/O2/O3/O5 的方法摘要（正文少公式，细节见 `docs/Dissertation_Techniques.md`）
- Results（3500字）：
  - 先写 O1 的不稳定性量化（图表齐全）
  - 再写 O3/O4 的主结果与消融、统计显著性
  - 最后写 O5 的部署与延迟（含权衡）
- Discussion（500字）：意义、局限、未来工作

## 5) 执行与验收清单（Checklist）

- [ ] O1：采集完成，统计表生成并审阅
- [ ] O2：预处理配置固定并记录（增强参数、SEED）
- [ ] O3：六模型结果落盘、可复现实验日志
- [ ] O4：消融对比与统计检验完成
- [ ] O5：三平台延迟与体积对比表产出
- [ ] O6：环境/脚本/参数版本锁定与复核


