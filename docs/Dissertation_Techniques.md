# Techniques for Dissertation (BSL Gesture Recognition System)

This document summarizes the technical methods used in the project, structured for direct citation in a dissertation.

## A. Hybrid ADANN_LightGBM (Core Method)

We combine a domain-adaptive neural feature extractor (ADANN) with a LightGBM classifier to achieve both cross-subject robustness and deployability. The neural branch learns domain-invariant representations; the tree branch provides compact, efficient decision boundaries and an Arduino-friendly export path.

### A.1 Objective (domain-adversarial)
Let F be the feature extractor, C the gesture classifier, and D the domain discriminator with Gradient Reversal Layer (GRL). Given labeled gesture pairs (x, y) and domain labels d (subject IDs), we optimize

\[ \min_{F,C}\max_{D}\\ \mathcal{L}(F,C,D) = \mathcal{L}_{ce}\big(C(F(x)), y\big) + \lambda\,\mathcal{L}_{dom}\big(D(F(x)), d\big). \]

The CE term is standard multi-class cross-entropy
\[ \mathcal{L}_{ce} = -\frac{1}{N} \sum_{n=1}^{N}\sum_{c=1}^{C} y_{nc}\log p_{nc}, \quad p = \mathrm{softmax}(C(F(x))). \]

The domain loss encourages F to produce features that confuse D about subject identity; GRL implements the inner maximization by reversing gradients during backprop.

### A.2 Hybrid fusion and export
- Training: optimize ADANN via \( \mathcal{L} \) to obtain feature embeddings; train LightGBM on either (i) these embeddings or (ii) a compact 20-D statistical feature set for Arduino mode.
- Export: ADANN weights are emitted as a pure-C FP32 header; LightGBM is exported via m2cgen to C. On-device inference uses both heads where applicable, subject to flash limits.

### A.3 Optuna-driven HPO (summary)
We search hyperparameters \(\theta\) by maximizing validation accuracy:
\[ \theta^* = \arg\max_{\theta} \ \mathrm{ValAcc}(\theta). \]
We employ TPESampler (seeded) and MedianPruner. For Keras models, we use TFKerasPruningCallback; for XGBoost we use XGBoostPruningCallback; LightGBM uses internal validation.

### A.4 Arduino optimizations
- Pruning: polynomial sparsity schedule \( s(t)=s_0+(s_1-s_0)\big(\tfrac{t-t_0}{T-t_0}\big)^{p} \), followed by strip pruning.
- Quantization: post-training INT8 with scale–zero-point mapping \( q = \mathrm{round}(x/s)+z \), \( \hat{x} = s(q-z) \). Representative dataset is drawn from training features to calibrate activations.

### A.5 LOSO evaluation protocol
For subject set \( S \): for each \( s\in S \), train on \( D\setminus D_s \) and test on \( D_s \), report mean and std across folds.

### A.6 Pseudocode (training → export)
```text
# HPO and final export for Hybrid ADANN_LightGBM (sketch)
best = None
for trial in range(n_trials):
  params = sample_params(trial)
  X_tr, y_tr, X_val, y_val = prepare_data(params)
  adann = build_adann(params)
  fit_with_pruning_and_early_stop(adann, X_tr, y_tr, X_val, y_val)
  score = evaluate(adann, X_val, y_val)
  best = max(best, (score, params))

# Final training on full data
adann = build_adann(best.params)
adann.fit(X_all, y_all)

# Train LightGBM on embeddings or 20-D features (Arduino mode)
Z_all = extract_features(adann, X_all)  # or handcrafted_features(X_all)
lgbm = train_lightgbm(Z_all, y_all)

# Export
export_c_header_adann(adann)
export_c_header_lightgbm(lgbm)
copy_headers_to_arduino_latency_dirs()
```

### A.7 Pseudocode (Arduino latency)
```text
record_input_once()
warmup(W)
times = []
for r in range(R):
  t0 = now()
  y = infer_once(x_fixed)
  times.append(now() - t0)
report(mean(times), throughput=1/mean(times))
```

## 1. Data Processing
- Sampling: All sequences are linearly resampled to 100×5 (100 timesteps, 5 channels) to unify model inputs.
- Augmentation: Lightweight strategy combining Gaussian noise (AddNoise), time-warp (TimeWarp), and probabilistic amplitude scaling; probabilities/strengths tuned by Optuna/config.
- Normalization: StandardScaler is fitted on training data and applied to all splits to normalize per-feature statistics.

## 2. Reproducibility
- Global seeding across Python/NumPy/TensorFlow; TensorFlow deterministic ops enabled when possible.
- Optuna uses TPESampler with fixed seed and MedianPruner for stable, comparable HPO.

## 3. Hyperparameter Optimization (Optuna)
- Keras/TensorFlow models: batch size, learning rate, convolution kernel counts/sizes, dense units, dropout rate, augmentation probabilities, etc., with TFKerasPruningCallback.
- XGBoost/LightGBM: depth, learning rate, estimators/leaf nodes/subsample/colsample, with pruning (XGBoost) or built-in validation.
- LOSO: per-fold optimization with final re-training on full data using the best overall trial.

## 4. Model Families
- 1D_CNN: TFLite Micro friendly; quantization-aware deployment path.
- Transformer_Encoder: Desktop/mobile runtime; Arduino not exported due to current TFLM op limits.
- XGBoost/LightGBM: Tree-based models trained on statistical features; exportable to C via micromlgen/m2cgen.
- ADANN: Domain-adaptive representation learning for cross-subject generalization.
- ADANN_LightGBM: Hybrid with ADANN feature extractor + LightGBM classifier for accuracy and deployability.

## 5. Quantization and Pruning (Arduino Mode)
- Magnitude pruning with polynomial decay from 30% to 60%, followed by strip-pruning before export.
- Full integer post-training quantization (INT8) with representative dataset; TFLite conversion configured for TFLM compatibility where possible.
- Auto-export: C headers are generated and copied into `arduino/tinyml_inference/*/Latency_*` folders.

## 6. Feature Engineering for Tree Models
- On-device features: for each of the 5 channels, compute mean, std, min, max → 20-D vector.
- Alignment with training: same feature subset is used during Arduino-mode training to match inference protocol.

## 7. Latency Measurement Protocols
- Arduino (inference-only recommended): precompute input once (`record`), then run `latency R W` and report mean latency and IPS.
- CPU/A100: Notebooks provide end-to-end and inference-only options with controlled warmups and repetitions; results are logged and plotted.

## Appendix (optional formulas)

### Linear resampling to fixed length
For channel i and target index \( t\in\{0,\dots,99\} \):
\[ \alpha_t = \tfrac{t}{99}(N-1),\quad k=\lfloor \alpha_t \rfloor,\quad \delta=\alpha_t-k,\quad x'_i(t)=(1-\delta)x_i(k)+\delta x_i(k+1). \]

### Standardization (z-score)
\[ \tilde{x}_i = (x_i - \mu_i)/\sigma_i, \] where \(\mu_i,\sigma_i\) are estimated on training data.

## 8. Artifact Management
- Artifacts include `.keras`/`.tflite` (where applicable), `.h` headers for Arduino, and scalers (`.pkl`), organized under `models/trained/<Model>/<mode>/<opt>/` and `outputs/` for evaluations.

## 9. Limitations
- Transformer_Encoder not exported to Arduino due to missing `BATCH_MATMUL` in current TFLM builds.
- Hybrid headers must satisfy board flash limits; LightGBM branch complexity is constrained in Arduino mode.


