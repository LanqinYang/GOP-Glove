"""
Unified training pipeline for all models.
"""

import os
import numpy as np
import glob
import re
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import yaml
from tsaug import TimeWarp, AddNoise
import shutil

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
# Optional: TensorFlow Model Optimization Toolkit for pruning
try:
    import tensorflow_model_optimization as tfmot  # type: ignore
except Exception:
    tfmot = None
import optuna
from optuna.integration import TFKerasPruningCallback, XGBoostPruningCallback
# Kalman filter imports removed - tests showed it degraded performance

# Project-specific imports
from src.evaluation.evaluator import comprehensive_evaluation, generate_loso_summary_plots

# --- Constants ---
SEED = 42
SEQUENCE_LENGTH = 100
N_FEATURES = 5
N_CLASSES = 11
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# --- 设置全局随机种子确保结果可重现 ---
def set_global_seed(seed=SEED):
    """设置所有随机数生成器的种子，确保结果可重现"""
    import random
    import numpy as np
    import tensorflow as tf
    
    # Python内置random
    random.seed(seed)
    
    # NumPy随机种子
    np.random.seed(seed)
    
    # TensorFlow随机种子
    tf.random.set_seed(seed)
    
    # 如果有PyTorch，也设置种子
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # 设置环境变量以确保TensorFlow确定性
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# 已删除无用的滑动窗口标准化函数（验证后发现会降低性能）

# --- Data Loading ---
def load_data(csv_dir):
    """Load and preprocess data, extracting subject IDs for LOSO."""
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    all_data, all_labels, all_subjects = [], [], []
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()
            
            data = []
            for line in lines:
                if 'timestamp' in line or line.startswith('#'):
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    try:
                        values = [float(parts[i]) for i in range(1, 6)]
                        data.append(values)
                    except ValueError:
                        continue
            
            if data:
                data = np.array(data)
                # Resample to fixed length
                if len(data) != SEQUENCE_LENGTH:
                    indices = np.linspace(0, len(data) - 1, SEQUENCE_LENGTH)
                    resampled = np.zeros((SEQUENCE_LENGTH, N_FEATURES))
                    for i in range(N_FEATURES):
                        resampled[:, i] = np.interp(indices, range(len(data)), data[:, i])
                    data = resampled
                
                # Extract gesture ID
                gesture_id_match = re.search(r'gesture_(\d+)', csv_file)
                if not gesture_id_match:
                    continue
                gesture_id = int(gesture_id_match.group(1))

                # Extract subject ID
                subject_id_match = re.search(r'user_(\d+)', csv_file)
                if subject_id_match:
                    subject_id = int(subject_id_match.group(1))
                else:
                    subject_id = -1
                
                # 转换为int16以节省内存和提升性能（EMG信号通常在合理整数范围内）
                data = np.round(data).astype(np.int16)
                all_data.append(data)
                all_labels.append(gesture_id)
                all_subjects.append(subject_id)
        except Exception as e:
            print(f"Warning: Could not process file {csv_file}. Error: {e}")
            continue
    
    return np.array(all_data), np.array(all_labels), np.array(all_subjects)

# apply_kalman_filter function removed - tests showed it degraded performance by 1.52%

def augment_data(X_train, y_train, augment_params=None):
    """
    Applies data augmentation to the training set.
    If augment_params is provided, use those; otherwise use config.yaml.
    """
    if augment_params is None:
        try:
            # Prefer temporary config written by ablation scripts if present
            config_path = 'configs/temp_config.yaml' if os.path.exists('configs/temp_config.yaml') else 'configs/config.yaml'
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                config = full_config['data']['augmentation']
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: Augmentation config not found ({e}). Skipping augmentation.")
            return X_train, y_train
        
        if not config.get('enabled', False):
            return X_train, y_train
            
        augment_params = {
            'augment_factor': config['augment_factor'],
            'jitter_noise_level': config['jitter_noise_level'],
            'time_warp_max_speed': config['time_warp_max_speed'],
            'scale_range': config['scale_range'],
            'augment_prob': 0.5  # Default probability
        }
    
    print(f"🚀 Augmenting data... (Factor: {augment_params['augment_factor']}x)")

    # Create the copies of the data that will be augmented
    X_to_augment = np.repeat(X_train, augment_params['augment_factor'], axis=0)
    y_to_augment = np.repeat(y_train, augment_params['augment_factor'], axis=0)

    # 1. Define augmenters from the library (tsaug)
    augment_prob = augment_params.get('augment_prob', 0.5)
    augmenter = (
        AddNoise(scale=augment_params['jitter_noise_level']) @ augment_prob
        + TimeWarp(n_speed_change=5, max_speed_ratio=augment_params['time_warp_max_speed']) @ augment_prob
    )
    
    # Apply tsaug augmentations
    X_augmented = augmenter.augment(X_to_augment)

    # 2. Apply our custom scaling augmentation manually
    # 使用确定性随机数生成器确保可重现性
    rng = np.random.RandomState(SEED + len(X_train))  # 基于SEED但避免重复
    scale_range = augment_params['scale_range']
    for i in range(X_augmented.shape[0]):
        # Apply scaling with specified probability
        if rng.rand() < augment_prob:
            scale_factor = rng.uniform(scale_range[0], scale_range[1])
            X_augmented[i] = X_augmented[i] * scale_factor

    # 3. Combine original and augmented data
    X_final = np.vstack([X_train, X_augmented])
    y_final = np.concatenate([y_train, y_to_augment])

    # 数据增强后转回int16以保持性能
    X_final = np.round(X_final).astype(np.int16)
    
    return X_final, y_final

def load_and_clean_data(csv_dir, cleaning_mode='none', baseline_samples=5):
    """
    Load data. Kalman filtering removed due to performance degradation.
    Only basic data loading is now supported.
    """
    X, y, subjects = load_data(csv_dir)
    
    # Kalman filtering and other cleaning modes removed
    # Tests showed Kalman filter reduced accuracy by 1.52%
    return X, y, subjects

# --- TFLite & Arduino Header Generation ---

# --- Pruning helper (Arduino mode) ---
def apply_pruning_if_needed(model, arduino_mode, X_train_len=None, batch_size=32, epochs=10):
    """Optionally wrap model with magnitude pruning for Arduino mode.

    Returns (maybe_wrapped_model, strip_fn, extra_callbacks)
    - strip_fn: callable to strip pruning from a trained model, or None
    - extra_callbacks: list of callbacks required for pruning step updates
    """
    if not arduino_mode or tfmot is None:
        return model, None, []
    try:
        steps_per_epoch = int(np.ceil(max(1, (X_train_len or 1)) / max(1, batch_size)))
        end_step = steps_per_epoch * max(1, epochs)
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.30,
                final_sparsity=0.60,
                begin_step=0,
                end_step=end_step
            )
        }
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        # Recompile preserves optimizer/loss/metrics
        pruned_model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
        strip_fn = lambda m: tfmot.sparsity.keras.strip_pruning(m)
        extra_callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        print("Enabled magnitude pruning for Arduino mode (30%->60% polynomial decay).")
        return pruned_model, strip_fn, extra_callbacks
    except Exception as e:
        print(f"⚠️ Pruning not applied: {e}")
        return model, None, []

def generate_arduino_header_tflite(tflite_model, scaler, model_type, timestamp, output_dir, input_len=None, output_len=None):
    """Generates a C header file for a TFLite model."""
    model_hex = ', '.join(f'0x{b:02x}' for b in tflite_model)
    
    # Handle different scaler types
    if hasattr(scaler, 'mean_'):  # StandardScaler
        mean_values = ', '.join(f'{val:.8f}f' for val in scaler.mean_)
        scale_values = ', '.join(f'{val:.8f}f' for val in scaler.scale_)
        n_features = len(scaler.mean_)
    elif hasattr(scaler, 'center_'):  # RobustScaler
        mean_values = ', '.join(f'{val:.8f}f' for val in scaler.center_)
        scale_values = ', '.join(f'{val:.8f}f' for val in scaler.scale_)
        n_features = len(scaler.center_)
    else:
        raise ValueError(f"Unsupported scaler type: {type(scaler)}")
    
    header_content = f"""/*
 * BSL Gesture Recognition Model
 * Model Type: {model_type}
 * Timestamp:  {timestamp}
 */
#ifndef BSL_MODEL_H_{timestamp}
#define BSL_MODEL_H_{timestamp}

const int BSL_MODEL_FEATURES = {n_features};
const int BSL_MODEL_INPUT_LEN = {input_len if input_len is not None else '/*unknown*/0'};
const int BSL_MODEL_OUTPUT_LEN = {output_len if output_len is not None else '/*unknown*/0'};
const float scaler_mean[BSL_MODEL_FEATURES] = {{ {mean_values} }};
const float scaler_scale[BSL_MODEL_FEATURES] = {{ {scale_values} }};

alignas(16) const unsigned char model_data[] = {{ {model_hex} }};
const unsigned int model_data_len = {len(tflite_model)};

#endif // BSL_MODEL_H_{timestamp}
"""
    header_path = os.path.join(output_dir, f"bsl_model_{model_type}_{timestamp}.h")
    with open(header_path, 'w') as f:
        f.write(header_content)
    return header_path

def generate_lightgbm_arduino_header(model, scaler, model_type, timestamp, output_dir):
    """Generates a C header file for a LightGBM model.

    Supports both wrapper objects with attribute `lgb_model` and raw
    LightGBM estimators (e.g., LGBMClassifier/Booster-like with predict()).
    """
    try:
        # Try to use m2cgen for LightGBM to C conversion
        import m2cgen as m2c
        # Detect underlying estimator
        underlying_estimator = getattr(model, 'lgb_model', model)
        c_code = m2c.export_to_c(underlying_estimator)
        # Make C code Arduino/C++ friendly: eliminate compound literal in softmax call
        # Transform: softmax((double[]){ a, b, ... }, 11, varX); ->
        #            double __logits[] = { a, b, ... }; softmax(__logits, 11, varX);
        if 'softmax((double[]){' in c_code:
            c_code = c_code.replace('softmax((double[]){', 'double __logits[] = {')
            c_code = c_code.replace('}, 11, ', '}; softmax(__logits, 11, ')
        
        # Handle different scaler types
        if hasattr(scaler, 'mean_'):  # StandardScaler
            mean_values = ', '.join(f'{val:.8f}f' for val in scaler.mean_)
            scale_values = ', '.join(f'{val:.8f}f' for val in scaler.scale_)
            n_features = len(scaler.mean_)
        elif hasattr(scaler, 'center_'):  # RobustScaler
            mean_values = ', '.join(f'{val:.8f}f' for val in scaler.center_)
            scale_values = ', '.join(f'{val:.8f}f' for val in scaler.scale_)
            n_features = len(scaler.center_)
        else:
            raise ValueError(f"Unsupported scaler type: {type(scaler)}")
        
        header_content = f"""/*
 * BSL Gesture Recognition Model - LightGBM
 * Model Type: {model_type}
 * Timestamp:  {timestamp}
 * 
 * Usage:
 * 1. Normalize features using scaler_mean and scaler_scale
 * 2. Call score(features) for prediction
 */
#ifndef BSL_MODEL_LIGHTGBM_H_{timestamp}
#define BSL_MODEL_LIGHTGBM_H_{timestamp}

#include <math.h>

// Feature preprocessing constants
const int BSL_MODEL_FEATURES = {n_features};
const float scaler_mean[BSL_MODEL_FEATURES] = {{ {mean_values} }};
const float scaler_scale[BSL_MODEL_FEATURES] = {{ {scale_values} }};

// LightGBM model prediction function (from m2cgen)
{c_code}

// Main prediction function (C++ friendly wrapper)
int predict_gesture(float* raw_features) {{
    double normalized_features[BSL_MODEL_FEATURES];
    for (int i = 0; i < BSL_MODEL_FEATURES; i++) {{
        normalized_features[i] = (double)((raw_features[i] - scaler_mean[i]) / scaler_scale[i]);
    }}
    double scores[11];
    score(normalized_features, scores);
    int predicted_class = 0;
    double max_score = scores[0];
    for (int i = 1; i < 11; i++) {{
        if (scores[i] > max_score) {{
            max_score = scores[i];
            predicted_class = i;
        }}
    }}
    return predicted_class;
}}

#endif // BSL_MODEL_LIGHTGBM_H_{timestamp}
"""
        
    except ImportError:
        print("⚠️ m2cgen not available, generating basic LightGBM header template...")
        # Fallback: Generate a basic template
        header_content = f"""/*
 * BSL Gesture Recognition Model - LightGBM (Template)
 * Model Type: {model_type}
 * Timestamp:  {timestamp}
 * 
 * NOTE: This is a template. For full Arduino deployment,
 * install m2cgen: pip install m2cgen
 */
#ifndef BSL_MODEL_LIGHTGBM_H_{timestamp}
#define BSL_MODEL_LIGHTGBM_H_{timestamp}

// TODO: Install m2cgen and re-run training to generate full C code
// pip install m2cgen

const int BSL_MODEL_FEATURES = {len(scaler.mean_) if hasattr(scaler, 'mean_') else 'N/A'};

// Template prediction function
int predict_gesture(float* features) {{
    // TODO: Add full LightGBM implementation
    return 0;  // Placeholder
}}

#endif // BSL_MODEL_LIGHTGBM_H_{timestamp}
"""
    
    header_path = os.path.join(output_dir, f"bsl_model_{model_type}_{timestamp}.h")
    with open(header_path, 'w') as f:
        f.write(header_content)
    return header_path

def generate_adann_c_header_inline(adann_model, scaler, timestamp, output_dir):
    """Generate a pure-C Arduino header for ADANN with FP32 constants.

    - Store all weights/bias/scaler as float32 constants (fast on M4F FPU)
    - Provide simple matvec and relu routines
    """
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)

    sd = adann_model.state_dict()
    # Infer dimensions from weights
    fe1_w = sd['feature_extractor.0.weight'].cpu().numpy().astype(np.float32)  # [256, in]
    fe1_b = sd['feature_extractor.0.bias'].cpu().numpy().astype(np.float32)
    fe2_w = sd['feature_extractor.3.weight'].cpu().numpy().astype(np.float32)  # [128, 256]
    fe2_b = sd['feature_extractor.3.bias'].cpu().numpy().astype(np.float32)
    fe3_w = sd['feature_extractor.6.weight'].cpu().numpy().astype(np.float32)  # [F, 128]
    fe3_b = sd['feature_extractor.6.bias'].cpu().numpy().astype(np.float32)

    gc1_w = sd['gesture_classifier.0.weight'].cpu().numpy().astype(np.float32) # [32, F]
    gc1_b = sd['gesture_classifier.0.bias'].cpu().numpy().astype(np.float32)
    gc2_w = sd['gesture_classifier.3.weight'].cpu().numpy().astype(np.float32) # [C, 32]
    gc2_b = sd['gesture_classifier.3.bias'].cpu().numpy().astype(np.float32)

    input_size = fe1_w.shape[1]
    feature_size = fe3_b.shape[0]
    n_classes = gc2_b.shape[0]

    # scaler for input features
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_') and len(scaler.mean_) == input_size:
        mean = np.asarray(scaler.mean_, dtype=np.float32)
        scale = np.asarray(scaler.scale_, dtype=np.float32)
    else:
        # fallback: identity
        mean = np.zeros((input_size,), dtype=np.float32)
        scale = np.ones((input_size,), dtype=np.float32)

    def arr_to_c(name, arr):
        flat = arr.flatten().tolist()
        if len(flat) == 0:
            return f'const float {name}[0] = {{ }};\n'
        values = ', '.join(f'{float(v):.8f}f' for v in flat)
        return f'const float {name}[{len(flat)}] = {{ {values} }};\n'

    def mat_comment(w):
        return f"// [{w.shape[0]}, {w.shape[1]}]"

    header_path = os.path.join(output_dir, f"bsl_model_ADANN_{timestamp}.h")
    with open(header_path, 'w') as f:
        f.write(f"""/*\n * ADANN Gesture Head (FP32 constants)\n * Timestamp: {timestamp}\n */\n""")
        f.write(f"#ifndef BSL_MODEL_ADANN_H_{timestamp}\n#define BSL_MODEL_ADANN_H_{timestamp}\n\n")

        f.write(f"const int ADANN_INPUT_SIZE = {input_size};\n")
        f.write(f"const int ADANN_FEATURE_SIZE = {feature_size};\n")
        f.write(f"const int ADANN_NUM_CLASSES = {n_classes};\n\n")

        # scaler in Arduino-friendly const arrays
        f.write(arr_to_c('adann_scaler_mean', mean))
        f.write(arr_to_c('adann_scaler_scale', scale))
        f.write('\n')

        # weights
        f.write(f"{mat_comment(fe1_w)}\n")
        f.write(arr_to_c('fe1_w', fe1_w))
        f.write(arr_to_c('fe1_b', fe1_b))
        f.write(f"{mat_comment(fe2_w)}\n")
        f.write(arr_to_c('fe2_w', fe2_w))
        f.write(arr_to_c('fe2_b', fe2_b))
        f.write(f"{mat_comment(fe3_w)}\n")
        f.write(arr_to_c('fe3_w', fe3_w))
        f.write(arr_to_c('fe3_b', fe3_b))
        f.write(f"{mat_comment(gc1_w)}\n")
        f.write(arr_to_c('gc1_w', gc1_w))
        f.write(arr_to_c('gc1_b', gc1_b))
        f.write(f"{mat_comment(gc2_w)}\n")
        f.write(arr_to_c('gc2_w', gc2_w))
        f.write(arr_to_c('gc2_b', gc2_b))

        f.write("""
static inline void adann_normalize(float* x) {
    for (int i = 0; i < ADANN_INPUT_SIZE; ++i) {
        x[i] = (x[i] - adann_scaler_mean[i]) / adann_scaler_scale[i];
    }
}

static inline float relu(float v) { return v > 0.f ? v : 0.f; }

static inline void matvec(const float* W, const float* b, const float* x,
                          int out_dim, int in_dim, float* y) {
    for (int o = 0; o < out_dim; ++o) {
        float acc = b[o];
        int base = o * in_dim;
        for (int i = 0; i < in_dim; ++i) {
            acc += W[base + i] * x[i];
        }
        y[o] = acc;
    }
}

static inline int adann_predict(float* x_in) {
    float x0[ADANN_INPUT_SIZE];
    for (int i = 0; i < ADANN_INPUT_SIZE; ++i) x0[i] = x_in[i];
    adann_normalize(x0);

    float h1[256];
    matvec(fe1_w, fe1_b, x0, 256, ADANN_INPUT_SIZE, h1);
    for (int i = 0; i < 256; ++i) h1[i] = relu(h1[i]);

    float h2[128];
    matvec(fe2_w, fe2_b, h1, 128, 256, h2);
    for (int i = 0; i < 128; ++i) h2[i] = relu(h2[i]);

    float hf[ADANN_FEATURE_SIZE];
    matvec(fe3_w, fe3_b, h2, ADANN_FEATURE_SIZE, 128, hf);
    for (int i = 0; i < ADANN_FEATURE_SIZE; ++i) hf[i] = relu(hf[i]);

    float g1[32];
    matvec(gc1_w, gc1_b, hf, 32, ADANN_FEATURE_SIZE, g1);
    for (int i = 0; i < 32; ++i) g1[i] = relu(g1[i]);

    float logits[ADANN_NUM_CLASSES];
    matvec(gc2_w, gc2_b, g1, ADANN_NUM_CLASSES, 32, logits);

    int argm = 0; float best = logits[0];
    for (int c = 1; c < ADANN_NUM_CLASSES; ++c) {
        if (logits[c] > best) { best = logits[c]; argm = c; }
    }
    return argm;
}

#endif // BSL_MODEL_ADANN_H
""")
    return header_path

def convert_to_tflite(model, arduino_mode, X_train_scaled=None):
    """Converts a Keras model to a TFLite model, with specific optimizations."""
    @tf.function
    def model_func(x):
        return model(x)
    
    input_shape = [1] + list(model.input_shape[1:])
    concrete_func = model_func.get_concrete_function(
        tf.TensorSpec(shape=input_shape, dtype=tf.float32)
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    # Always enable SELECT_TF_OPS for compatibility with complex layers (non-Arduino path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    
    if arduino_mode:
        print("Applying full integer quantization for Arduino (INT8, no SELECT_TF_OPS)...")
        def representative_dataset():
            for i in range(min(100, len(X_train_scaled))):
                yield [X_train_scaled[i:i+1].astype(np.float32)]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        # For Arduino (TFLite Micro), restrict to INT8 builtins only
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    return converter.convert()

# --- Hyperparameter Optimization ---

def copy_header_to_arduino_dir(model_type: str, header_path: str, mode_suffix: str):
    """Copy the generated header to the Arduino example directory with canonical name.

    Supported mappings:
      - 1D_CNN -> arduino/tinyml_inference/1D_CNN_inference/bsl_model_1D_CNN.h
      - Transformer_Encoder -> arduino/tinyml_inference/Transformer_Encoder_inference/bsl_model_Transformer_Encoder.h
      - XGBoost -> arduino/tinyml_inference/XGBoost_Arduino_inference/bsl_model_XGBoost_Arduino.h
      - LightGBM -> arduino/tinyml_inference/LightGBM_Arduino_inference/bsl_model_LightGBM.h
      - ADANN -> arduino/tinyml_inference/ADANN_inference/bsl_model_ADANN.h
      - ADANN_LightGBM -> arduino/tinyml_inference/ADANN_LightGBM_inference/bsl_model_ADANN_LightGBM.h
    """
    mapping = {
        '1D_CNN': (
            os.path.join('arduino', 'tinyml_inference', '1D_CNN_inference'),
            'bsl_model_1D_CNN.h'
        ),
        'Transformer_Encoder': (
            os.path.join('arduino', 'tinyml_inference', 'Transformer_Encoder_inference'),
            'bsl_model_Transformer_Encoder.h'
        ),
        'XGBoost': (
            os.path.join('arduino', 'tinyml_inference', 'XGBoost_Arduino_inference'),
            'bsl_model_XGBoost_Arduino.h'
        ),
        'LightGBM': (
            os.path.join('arduino', 'tinyml_inference', 'LightGBM_Arduino_inference'),
            'bsl_model_LightGBM.h'
        ),
        'ADANN': (
            os.path.join('arduino', 'tinyml_inference', 'ADANN_inference'),
            'bsl_model_ADANN.h'
        ),
        'ADANN_LightGBM': (
            os.path.join('arduino', 'tinyml_inference', 'ADANN_LightGBM_inference'),
            'bsl_model_ADANN_LightGBM.h'
        )
    }
    if model_type not in mapping:
        return None
    target_dir, target_name = mapping[model_type]
    try:
        os.makedirs(target_dir, exist_ok=True)
        # 1) Copy to canonical root location
        target_path = os.path.join(target_dir, target_name)
        shutil.copyfile(header_path, target_path)
        print(f"Arduino header copied to: {target_path}")

        # Determine mode for latency folder
        mode_dir = 'loso' if mode_suffix.startswith('loso') else 'standard'

        # 2) Copy ONLY into Latency_* directory as requested
        latency_dir = os.path.join(target_dir, f"Latency_{mode_dir}")
        os.makedirs(latency_dir, exist_ok=True)
        latency_target_path = os.path.join(latency_dir, target_name)
        shutil.copyfile(header_path, latency_target_path)
        print(f"Arduino header copied to (latency): {latency_target_path}")

        # 3) Clean up legacy non-latency folders (loso/standard) headers if present
        for legacy in ['loso', 'standard']:
            legacy_dir = os.path.join(target_dir, legacy)
            legacy_header = os.path.join(legacy_dir, target_name)
            try:
                if os.path.isfile(legacy_header):
                    os.remove(legacy_header)
                    print(f"Removed legacy header: {legacy_header}")
                # Remove directory if empty after deletion (do not remove if contains other files like .ino)
                if os.path.isdir(legacy_dir) and len([p for p in os.listdir(legacy_dir) if not p.startswith('.')]) == 0:
                    os.rmdir(legacy_dir)
                    print(f"Removed empty legacy directory: {legacy_dir}")
            except Exception as e:
                print(f"⚠️ Failed to clean legacy path {legacy_dir}: {e}")

        return target_path
    except Exception as e:
        print(f"⚠️ Failed to copy header to Arduino dir: {e}")
        return None

def objective_tf(trial, args, model_creator, X_train_orig, y_train_orig, X_val_orig, y_val):
    """Generic Optuna objective function for TensorFlow models with full optimization."""
    params = model_creator.define_hyperparams(trial, args.arduino)
    try:
        # Kalman filtering removed - tests showed it degraded performance by 1.52%
        X_train_filtered = X_train_orig
        X_val_filtered = X_val_orig
        
        # Extract augmentation parameters if they exist
        if not args.arduino and 'augment_factor' in params:
            augment_params = {
                'augment_factor': params['augment_factor'],
                'jitter_noise_level': params['jitter_noise_level'],
                'time_warp_max_speed': params['time_warp_max_speed'],
                'scale_range': params['scale_range'],
                'augment_prob': params['augment_prob']
            }
            # Apply optimized augmentation
            X_train_aug, y_train_aug = augment_data(X_train_filtered, y_train_orig, augment_params)
            # Re-extract features with augmented data
            X_train, scaler = model_creator.extract_and_scale_features(X_train_aug, fit=True, arduino_mode=args.arduino)
            X_val_scaled, _ = model_creator.extract_and_scale_features(X_val_filtered, scaler=scaler, arduino_mode=args.arduino)
        else:
            X_train, scaler = model_creator.extract_and_scale_features(X_train_filtered, fit=True, arduino_mode=args.arduino)
            X_val_scaled, _ = model_creator.extract_and_scale_features(X_val_filtered, scaler=scaler, arduino_mode=args.arduino)
            y_train_aug = y_train_orig
        
        model = model_creator.create_model(params, args.arduino)
        # Apply pruning in Arduino mode if available
        model, strip_fn, pruning_callbacks = apply_pruning_if_needed(
            model,
            args.arduino,
            X_train_len=X_train.shape[0],
            batch_size=params['batch_size'],
            epochs=args.epochs
        )
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            TFKerasPruningCallback(trial, "val_accuracy")
        ] + pruning_callbacks
        epochs = args.epochs
        model.fit(
            X_train, y_train_aug,
            batch_size=params['batch_size'],
            epochs=epochs,
            validation_data=(X_val_scaled, y_val),
            callbacks=callbacks,
            verbose=0
        )
        # If pruning was applied, strip it to get plain weights for evaluation/export
        if strip_fn is not None:
            try:
                model = strip_fn(model)
            except Exception:
                pass
        return max(model.history.history['val_accuracy'])
    except Exception as e:
        if isinstance(e, optuna.exceptions.TrialPruned):
            raise
        print(f"Trial failed with error: {e}. Reporting as pruned.")
        return 0.0

def objective_xgb(trial, args, model_creator, X_train, y_train, X_val, y_val):
    """Generic Optuna objective function for XGBoost models."""
    from sklearn.metrics import accuracy_score
    params = model_creator.define_hyperparams(trial, args.arduino)
    try:
        pruning_callback = XGBoostPruningCallback(trial, "validation_0-mlogloss")
        model = model_creator.create_model(params, args.arduino, callbacks=[pruning_callback])
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        return accuracy_score(y_val, y_pred)
    except Exception as e:
        if isinstance(e, optuna.exceptions.TrialPruned):
            raise
        print(f"Trial failed with error: {e}. Reporting as pruned.")
        return 0.0

def objective_lightgbm(trial, args, model_creator, X_train, y_train, X_val, y_val):
    """Generic Optuna objective function for LightGBM models."""
    from sklearn.metrics import accuracy_score
    import traceback
    
    try:
        params = model_creator.define_hyperparams(trial, args.arduino)
        
        # 检查数据完整性 - 确保训练集包含所有验证集的类别
        train_classes = set(y_train)
        val_classes = set(y_val)
        if not val_classes.issubset(train_classes):
            missing_classes = val_classes - train_classes
            print(f"Error: Validation set has classes not in training set: {missing_classes}")
            return 0.0
        
        # 检查输入数据是否已经是特征数据（2D）还是原始数据（3D）
        if len(X_train.shape) == 3:
            # 原始数据，需要特征提取
            X_train_feat, scaler = model_creator.extract_and_scale_features(X_train, fit=True, arduino_mode=args.arduino)
            X_val_feat, _ = model_creator.extract_and_scale_features(X_val, scaler=scaler, arduino_mode=args.arduino)
        else:
            # 已经是特征数据，直接使用
            X_train_feat, X_val_feat = X_train, X_val
        
        # 检查特征形状
        if X_train_feat.shape[0] == 0 or X_val_feat.shape[0] == 0:
            print(f"Error: Empty feature arrays. Train: {X_train_feat.shape}, Val: {X_val_feat.shape}")
            return 0.0
        
        model = model_creator.create_model(params, args.arduino)
        # LightGBM wrapper handles validation internally
        model.fit(X_train_feat, y_train, validation_data=(X_val_feat, y_val), verbose=0)
        y_pred = model.predict(X_val_feat)
        
        # 检查预测结果
        if len(y_pred) == 0:
            print(f"Error: Empty predictions")
            return 0.0
            
        return accuracy_score(y_val, y_pred)
        
    except Exception as e:
        print(f"LightGBM trial failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        print("Full traceback:")
        traceback.print_exc()
        return 0.0

# --- Optuna Artifacts Saving ---

def _save_optuna_artifacts(study, args, output_dir, fold_id=None):
    """Save Optuna study results to JSON and CSV files for analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Helper to convert NumPy types to native Python types for JSON serialization
    def _to_basic_type(obj):
        import numpy as np  # local import to avoid top-level dependency issues
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj
    
    # Determine file suffix based on mode and fold
    if fold_id is not None:
        suffix = f"_fold_{fold_id}"
    else:
        suffix = ""
    
    training_mode = "loso" if args.loso else "standard" 
    optimization_mode = "arduino" if args.arduino else "full"
    
    # Save best parameters
    best_params_file = f"best_params_{args.model_type}_{training_mode}_{optimization_mode}{suffix}_{timestamp}.json"
    best_params_path = os.path.join(output_dir, best_params_file)
    
    # Safely convert best params (may contain NumPy types)
    safe_best_params = {k: _to_basic_type(v) for k, v in study.best_params.items()}
    best_params_data = {
        'model_type': args.model_type,
        'training_mode': training_mode,
        'optimization_mode': optimization_mode,
        'fold_id': _to_basic_type(fold_id) if fold_id is not None else None,
        'timestamp': timestamp,
        'best_trial_number': _to_basic_type(study.best_trial.number),
        'best_value': _to_basic_type(study.best_value),
        'best_params': safe_best_params,
        'n_trials': _to_basic_type(len(study.trials)),
        'study_direction': study.direction.name
    }
    
    with open(best_params_path, 'w') as f:
        json.dump(best_params_data, f, indent=2)
    print(f"Best parameters saved: {best_params_path}")
    
    # Save trials data
    trials_file = f"trials_{args.model_type}_{training_mode}_{optimization_mode}{suffix}_{timestamp}.csv"
    trials_path = os.path.join(output_dir, trials_file)
    
    # Convert trials to DataFrame-compatible format
    trials_data = []
    for trial in study.trials:
        trial_dict = {
            'trial_number': trial.number,
            'value': trial.value,
            'state': trial.state.name,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            'duration_seconds': (trial.datetime_complete - trial.datetime_start).total_seconds() 
                               if trial.datetime_complete and trial.datetime_start else None
        }
        # Add all parameters as separate columns
        for param_name, param_value in trial.params.items():
            trial_dict[f'param_{param_name}'] = param_value
        trials_data.append(trial_dict)
    
    # Save as CSV
    import pandas as pd
    trials_df = pd.DataFrame(trials_data)
    trials_df.to_csv(trials_path, index=False)
    print(f"Trials data saved: {trials_path}")
    
    return best_params_path, trials_path

# --- Main Pipeline ---

def run_training_pipeline(args, model_creator):
    """
    A unified training pipeline for any given model.
    """
    # 首先设置全局随机种子，确保结果可重现
    set_global_seed(SEED)
    print(f"🎲 Global random seed set to {SEED} for reproducible results")
    
    # 1. Setup paths
    training_mode = "loso" if args.loso else "standard"
    optimization_mode = "arduino" if args.arduino else "full"
    final_model_type = f"{args.model_type}_{training_mode}_{optimization_mode}"
    
    output_dir = os.path.join('outputs', args.model_type, training_mode, optimization_mode)
    trained_model_dir = os.path.join(args.output_dir, args.model_type, training_mode, optimization_mode)

    # Append optional output suffix for ablation runs to avoid collision and simplify aggregation
    try:
        suffix_raw = getattr(args, 'output_suffix', '') or ''
    except Exception:
        suffix_raw = ''
    if isinstance(suffix_raw, str) and suffix_raw.strip():
        # Sanitize to filesystem-friendly form: replace spaces and non-alnum/[_-] with '_'
        safe_suffix = re.sub(r'[^A-Za-z0-9_\-]+', '_', suffix_raw.strip())
        output_dir = os.path.join(output_dir, safe_suffix)
        trained_model_dir = os.path.join(trained_model_dir, safe_suffix)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(trained_model_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🚀 Starting Training Pipeline: {final_model_type}")
    print(f"   - Evaluation outputs: {output_dir}")
    print(f"   - Trained models: {trained_model_dir}")
    print(f"{'='*60}\n")
    
    # 2. Load and Process Data
    print("💾 Loading original data...")
    X, y, subjects = load_data(args.csv_dir)
    
    # Kalman filter removed - tests showed it degraded performance by 1.52%
    print("✅ Data loaded without Kalman filtering (performance optimization)")

    if args.loso:
        run_loso_pipeline(args, model_creator, X, y, subjects, output_dir, trained_model_dir)
    else:
        run_standard_pipeline(args, model_creator, X, y, output_dir, trained_model_dir)

def run_standard_pipeline(args, model_creator, X, y, output_dir, trained_model_dir):
    """Pipeline for standard train/val/test split."""
    # Data splitting
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VAL_SIZE, random_state=SEED, stratify=y_temp)
    
    # Apply Data Augmentation on the training set
    X_train, y_train = augment_data(X_train, y_train)

    # Feature extraction & Scaling
    X_train_feat, scaler = model_creator.extract_and_scale_features(X_train, fit=True, arduino_mode=args.arduino)
    X_val_feat, _ = model_creator.extract_and_scale_features(X_val, scaler=scaler, arduino_mode=args.arduino)
    X_test_feat, _ = model_creator.extract_and_scale_features(X_test, scaler=scaler, arduino_mode=args.arduino)

    # Hyperparameter Optimization - 选择正确的objective函数
    if args.model_type == 'XGBoost':
        objective_func = objective_xgb
        obj_args = (X_train_feat, y_train, X_val_feat, y_val)
    elif args.model_type == 'LightGBM':
        objective_func = objective_lightgbm
        obj_args = (X_train_feat, y_train, X_val_feat, y_val)
    else:
        # TensorFlow系列：将原始数据（未特征提取/缩放）传入 objective，让其内部完成增强与缩放
        objective_func = objective_tf
        obj_args = (X_train, y_train, X_val, y_val)
    study = optuna.create_study(
            direction='maximize', 
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )
    study.optimize(lambda trial: objective_func(trial, args, model_creator, *obj_args), n_trials=args.n_trials)
    
    # Save Optuna optimization artifacts
    _save_optuna_artifacts(study, args, output_dir)
    
    # Train final model
    best_params = study.best_params
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print("Best parameters:", best_params)
    final_model = model_creator.create_model(best_params, args.arduino)

    epochs = args.epochs
    if args.model_type != 'XGBoost':
         # Apply pruning if Arduino mode and TF model
         final_model, strip_fn, pruning_callbacks = apply_pruning_if_needed(
             final_model,
             args.arduino,
             X_train_len=X_train_feat.shape[0],
             batch_size=best_params.get('batch_size', 32),
             epochs=epochs
         )
         callbacks = [EarlyStopping(patience=15, restore_best_weights=True)] + pruning_callbacks
         history = final_model.fit(X_train_feat, y_train, validation_data=(X_val_feat, y_val), epochs=epochs, callbacks=callbacks, verbose=1)
         # Strip pruning wrappers before saving/exporting
         if strip_fn is not None:
             try:
                 final_model = strip_fn(final_model)
             except Exception:
                 pass
    else: # XGBoost
        final_model.fit(X_train_feat, y_train, eval_set=[(X_val_feat, y_val)], early_stopping_rounds=25, verbose=False)
        history = None

    # Save model, scaler, header, etc.
    save_artifacts(final_model, scaler, args, trained_model_dir, output_dir, history, X_train_feat)

    # Final Evaluation
    comprehensive_evaluation(final_model, X_test_feat, y_test, scaler, output_dir, f"{args.model_type}_standard", history, class_names=[f'Gesture_{i}' for i in range(N_CLASSES)])

def run_loso_pipeline(args, model_creator, X, y, subjects, output_dir, trained_model_dir):
    """Pipeline for Leave-One-Subject-Out cross-validation."""
    unique_subjects = np.unique(subjects[subjects != -1])
    if len(unique_subjects) < 2:
        print("❌ ERROR: LOSO training requires at least 2 subjects.")
        return

    all_fold_evaluations = []
    if args.model_type == 'XGBoost':
        objective_func = objective_xgb
    elif args.model_type == 'LightGBM':
        objective_func = objective_lightgbm
    else:
        objective_func = objective_tf
    best_trial_overall = None

    for subject_id in unique_subjects:
        print(f"\n--- LOSO Fold: Testing on Subject {subject_id} ---")
        test_indices = np.where(subjects == subject_id)[0]
        train_indices = np.where(subjects != subject_id)[0]
        
        X_train_full, y_train_full = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # For LOSO, we need to handle augmentation differently for optimization
        if args.model_type == 'XGBoost':
            # XGBoost: Apply augmentation first, then extract features
            X_train_full, y_train_full = augment_data(X_train_full, y_train_full)
            X_train_full_feat, scaler = model_creator.extract_and_scale_features(X_train_full, fit=True, arduino_mode=args.arduino)
            X_test_feat, _ = model_creator.extract_and_scale_features(X_test, scaler=scaler, arduino_mode=args.arduino)
            X_train_feat, X_val_feat, y_train, y_val = train_test_split(X_train_full_feat, y_train_full, test_size=VAL_SIZE, random_state=SEED, stratify=y_train_full)
            study = optuna.create_study(
            direction='maximize', 
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )
            study.optimize(lambda trial: objective_func(trial, args, model_creator, X_train_feat, y_train, X_val_feat, y_val), n_trials=args.n_trials)
            # Save Optuna artifacts for each fold
            _save_optuna_artifacts(study, args, output_dir, fold_id=subject_id)
        elif args.model_type == 'LightGBM':
            # LightGBM: Apply augmentation first, then extract features (similar to XGBoost)
            X_train_full, y_train_full = augment_data(X_train_full, y_train_full)
            X_train_full_feat, scaler = model_creator.extract_and_scale_features(X_train_full, fit=True, arduino_mode=args.arduino)
            X_test_feat, _ = model_creator.extract_and_scale_features(X_test, scaler=scaler, arduino_mode=args.arduino)
            X_train_feat, X_val_feat, y_train, y_val = train_test_split(X_train_full_feat, y_train_full, test_size=VAL_SIZE, random_state=SEED, stratify=y_train_full)
            study = optuna.create_study(
            direction='maximize', 
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )
            study.optimize(lambda trial: objective_func(trial, args, model_creator, X_train_feat, y_train, X_val_feat, y_val), n_trials=args.n_trials)
            # Save Optuna artifacts for each fold
            _save_optuna_artifacts(study, args, output_dir, fold_id=subject_id)
        else:
            # TensorFlow models: Pass raw data for full optimization
            X_train_orig, X_val_orig, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=VAL_SIZE, random_state=SEED, stratify=y_train_full)
            study = optuna.create_study(
            direction='maximize', 
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )
            study.optimize(lambda trial: objective_func(trial, args, model_creator, X_train_orig, y_train, X_val_orig, y_val), n_trials=args.n_trials)
            # Save Optuna artifacts for each fold
            _save_optuna_artifacts(study, args, output_dir, fold_id=subject_id)
            
            # After optimization, prepare test data with best parameters
            best_params = study.best_params
            # Kalman filtering removed - tests showed it degraded performance by 1.52%
            X_test_filtered = X_test
            
            # Prepare final training data for fold model
            if 'augment_factor' in best_params:
                augment_params = {
                    'augment_factor': best_params['augment_factor'],
                    'jitter_noise_level': best_params['jitter_noise_level'],
                    'time_warp_max_speed': best_params['time_warp_max_speed'],
                    'scale_range': [best_params['scale_min'], best_params['scale_max']],
                    'augment_prob': best_params['augment_prob']
                }
                X_train_final_aug, y_train_final = augment_data(X_train_full, y_train_full, augment_params)
            else:
                X_train_final_aug, y_train_final = X_train_full, y_train_full
            
            X_train_final_feat, scaler = model_creator.extract_and_scale_features(X_train_final_aug, fit=True, arduino_mode=args.arduino)
            X_test_feat, _ = model_creator.extract_and_scale_features(X_test_filtered, scaler=scaler, arduino_mode=args.arduino)
        
        # Track the best trial across all folds
        if best_trial_overall is None or study.best_trial.value > best_trial_overall.value:
            best_trial_overall = study.best_trial
            print(f"🏆 New best trial found with validation accuracy: {best_trial_overall.value:.4f}")

        print(f"Fold {subject_id} best params: {study.best_params}")
        fold_model = model_creator.create_model(study.best_params, args.arduino)

        epochs = args.epochs
        history_fold = None
        if args.model_type == 'XGBoost':
            fold_model.fit(X_train_full_feat, y_train_full, eval_set=[(X_val_feat, y_val)], early_stopping_rounds=15, verbose=False)
        elif args.model_type == 'LightGBM':
            # LightGBM training with validation data - 使用完整的训练数据
            history_fold = fold_model.fit(X_train_full_feat, y_train_full, validation_data=(X_val_feat, y_val), verbose=0)
        else:
            # TensorFlow models
            # Apply pruning for Arduino mode
            # Determine training features depending on branch
            if 'X_train_final_feat' in locals():
                train_X = X_train_final_feat
                train_y = y_train_final
                val_X, val_y = None, None
                x_len = train_X.shape[0]
            else:
                # Use scaled features from earlier branch for TF models
                # Prepare features here to avoid UnboundLocalError
                X_train_full_feat, scaler = model_creator.extract_and_scale_features(X_train_full, fit=True, arduino_mode=args.arduino)
                X_test_feat, _ = model_creator.extract_and_scale_features(X_test, scaler=scaler, arduino_mode=args.arduino)
                X_train_feat, X_val_feat, y_train, y_val = train_test_split(X_train_full_feat, y_train_full, test_size=VAL_SIZE, random_state=SEED, stratify=y_train_full)
                train_X, train_y = X_train_feat, y_train
                val_X, val_y = X_val_feat, y_val
                x_len = train_X.shape[0]

            fold_model, strip_fn, pruning_callbacks = apply_pruning_if_needed(
                fold_model,
                args.arduino,
                X_train_len=x_len,
                batch_size=study.best_params.get('batch_size', 32),
                epochs=args.epochs
            )
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)] + pruning_callbacks
            
            # For TensorFlow models, use the optimized training data
            if 'X_train_final_feat' in locals():
                history_fold = fold_model.fit(train_X, train_y, epochs=epochs, callbacks=callbacks, verbose=0)
            else:
                history_fold = fold_model.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=epochs, callbacks=callbacks, verbose=0)
            # Strip pruning before evaluation/export
            if strip_fn is not None:
                try:
                    fold_model = strip_fn(fold_model)
                except Exception:
                    pass
            
        fold_eval = comprehensive_evaluation(fold_model, X_test_feat, y_test, scaler, output_dir, f"{args.model_type}_loso_fold_{subject_id}", history=history_fold)
        all_fold_evaluations.append(fold_eval)

    # --- Aggregate LOSO Results ---
    if not all_fold_evaluations or best_trial_overall is None:
        print("❌ ERROR: No LOSO folds were successfully evaluated or no best trial found. Aborting final model training.")
        return

    avg_accuracy = np.mean([eval_report['test_accuracy'] for eval_report in all_fold_evaluations])
    avg_f1 = np.mean([eval_report['classification_report']['macro avg']['f1-score'] for eval_report in all_fold_evaluations])
    print(f"\n{'='*60}")
    print("📊 LOSO Cross-Validation Summary:")
    print(f"   - Average Accuracy: {avg_accuracy:.4f}")
    print(f"   - Average Macro F1-Score: {avg_f1:.4f}")
    print(f"{'='*60}")

    # --- Final Deployment Model Training (after LOSO) ---
    print("\n--- 🏋️ Training Final Deployment Model on All Data ---")
    
    best_params_overall = best_trial_overall.params
    print(f"Using best hyperparameters found across all folds: {best_params_overall}")
    print(f"(Achieved validation accuracy of {best_trial_overall.value:.4f} in its best fold)")

    # 1. Feature extraction and scaling on the ENTIRE dataset
    X_all_feat, scaler_final = model_creator.extract_and_scale_features(X, fit=True, arduino_mode=args.arduino)
    
    # 2. Create the final model with the best overall parameters
    final_model = model_creator.create_model(best_params_overall, args.arduino)
    
    # 3. Train the final model on the ENTIRE dataset (no validation split)
    epochs = args.epochs
    history = None
    if args.model_type == 'XGBoost':
        print("Training final XGBoost model...")
        final_model.fit(X_all_feat, y)
    elif args.model_type == 'LightGBM':
        print("Training final LightGBM model...")
        history = final_model.fit(X_all_feat, y, verbose=1)
    else:
        print(f"Training final Keras model for {epochs} epochs...")
        # Apply pruning in Arduino mode for final deployment model as well
        final_model, strip_fn, pruning_callbacks = apply_pruning_if_needed(
            final_model,
            args.arduino,
            X_train_len=X_all_feat.shape[0],
            batch_size=best_params_overall.get('batch_size', 32),
            epochs=epochs
        )
        callbacks = [EarlyStopping(patience=10, restore_best_weights=True)] + pruning_callbacks
        history = final_model.fit(X_all_feat, y, epochs=epochs, callbacks=callbacks, verbose=1)
        # Strip pruning wrappers before saving/exporting
        if strip_fn is not None:
            try:
                final_model = strip_fn(final_model)
            except Exception:
                pass
    
    # 4. Save the final, deployable artifacts
    save_artifacts(final_model, scaler_final, args, trained_model_dir, output_dir, history, X_all_feat, is_final=True)

    # 5. Generate and save the final summary plot
    generate_loso_summary_plots(history, all_fold_evaluations, output_dir, f"{args.model_type}_loso_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # 5.5. 统计显著性分析 (如果有多个模型的LOSO结果)
    try:
        from src.evaluation.statistical_tests import ModelComparisonStatistics
        
        # 提取LOSO各fold的准确率
        fold_accuracies = [eval_report['test_accuracy'] for eval_report in all_fold_evaluations]
        
        print(f"\n📊 当前模型 {args.model_type} LOSO结果统计:")
        print(f"   各Fold准确率: {[f'{acc:.4f}' for acc in fold_accuracies]}")
        print(f"   平均准确率: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        print(f"   📝 提示: 保存此结果用于后续模型间统计比较")
        
        # 保存统计分析数据
        stats_data = {
            'model_type': args.model_type,
            'fold_accuracies': fold_accuracies,
            'mean_accuracy': float(np.mean(fold_accuracies)),
            'std_accuracy': float(np.std(fold_accuracies)),
            'n_folds': len(fold_accuracies),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        stats_path = os.path.join(output_dir, f"loso_stats_{args.model_type}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"   统计数据已保存至: {stats_path}")
        
    except ImportError:
        print("⚠️ 统计检验模块未找到，跳过统计分析")

    # 6. Save a summary of the LOSO evaluation process
    loso_summary = {
        "average_accuracy": avg_accuracy,
        "average_f1_macro": avg_f1,
        "best_hyperparameters_for_deployment": best_params_overall,
        "best_trial_value (val_accuracy)": best_trial_overall.value,
        "fold_details": all_fold_evaluations
    }
    summary_path = os.path.join(output_dir, f"loso_summary_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        # Use a custom encoder to handle numpy types if they appear
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        json.dump(loso_summary, f, indent=2, cls=NumpyEncoder)
    print(f"LOSO evaluation summary saved to: {summary_path}")


def save_artifacts(model, scaler, args, trained_model_dir, output_dir, history, X_train_data, is_final=False):
    """Saves all model artifacts to the specified directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if is_final:
        # For LOSO, the final model is the definitive one for that configuration
        mode_suffix = "loso_final"
    else:
        # For standard mode, it's just a regular run
        mode_suffix = "standard"

    full_model_name = f"{args.model_type}_{mode_suffix}_{timestamp}"
    
    # Save model artifacts by framework
    if args.model_type in ['ADANN', 'ADANN_LightGBM']:
        # PyTorch/Hybrid: 使用包装器自带的保存逻辑，落地为 .pth
        model_path = os.path.join(trained_model_dir, f"{full_model_name}.keras")
        model.save(model_path)  # 内部会写入同名 .pth
        # 额外：为 ADANN_LightGBM 生成 LightGBM 分支的 Arduino 头文件
        if args.model_type == 'ADANN_LightGBM':
            try:
                # 从包装器中取出 LightGBM 子模型与其 scaler
                lgb_estimator = None
                lgb_scaler = None
                if hasattr(model, 'hybrid_model') and isinstance(model.hybrid_model, dict):
                    lgb_estimator = model.hybrid_model.get('lightgbm')
                    lgb_scaler = model.hybrid_model.get('lgb_scaler')
                if lgb_estimator is not None and lgb_scaler is not None:
                    header_path = generate_lightgbm_arduino_header(
                        lgb_estimator, lgb_scaler, f"{args.model_type}_LGB_BRANCH", timestamp, trained_model_dir
                    )
                    print(f"Hybrid LightGBM Arduino header saved: {header_path}")
                    copy_header_to_arduino_dir('ADANN_LightGBM', header_path, mode_suffix)
                else:
                    print("⚠️ Hybrid LightGBM header skipped: missing lgb estimator or scaler in hybrid model")
            except Exception as e:
                print(f"⚠️ Failed to generate hybrid LightGBM Arduino header: {e}")
    elif args.model_type in ['XGBoost', 'LightGBM']:
        model_path = os.path.join(trained_model_dir, f"{full_model_name}.pkl")
        if args.model_type == 'LightGBM':
            # LightGBM has its own save method
            model.save(model_path)
        else:
            # XGBoost uses pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
    else:
        # Keras/TensorFlow 系列
        model_path = os.path.join(trained_model_dir, f"{full_model_name}.keras")
        model.save(model_path)
    print(f"Model saved: {model_path}")
            
    # Save Scaler
    scaler_path = os.path.join(trained_model_dir, f"scaler_{full_model_name}.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved: {scaler_path}")
    
    # Save Header File
    if args.model_type == 'XGBoost':
        from micromlgen import port
        c_code = port(model)
        header_content = f"// XGBoost Model\n{c_code}"
        header_path = os.path.join(trained_model_dir, f"{full_model_name}.h")
        with open(header_path, 'w') as f:
            f.write(header_content)
        print(f"Arduino header saved: {header_path}")
        if args.arduino:
            copy_header_to_arduino_dir(args.model_type, header_path, mode_suffix)
    elif args.model_type == 'LightGBM':
        # Generate LightGBM Arduino header
        header_path = generate_lightgbm_arduino_header(model, scaler, args.model_type, timestamp, trained_model_dir)
        print(f"LightGBM Arduino header saved: {header_path}")
        if args.arduino:
            copy_header_to_arduino_dir(args.model_type, header_path, mode_suffix)
    elif args.model_type in ['ADANN', 'ADANN_LightGBM']:
        print(f"PyTorch model ({args.model_type}): Skipping TFLite conversion")
        # Auto-generate ADANN C header for Arduino latency test
        if args.model_type == 'ADANN':
            try:
                import torch
                # 允许安全反序列化自定义类（我们信任自保存的包）
                try:
                    from src.training.train_adann import AdversarialFeatureExtractor
                    torch.serialization.add_safe_globals([AdversarialFeatureExtractor])
                except Exception:
                    pass
                # 兼容PyTorch 2.6+ 的 weights_only 默认行为
                try:
                    pkg = torch.load(model_path, map_location='cpu', weights_only=False)
                except TypeError:
                    pkg = torch.load(model_path, map_location='cpu')
                adann_model = pkg.get('adann_model') or pkg.get('model_object')
                # reuse training scaler
                adann_scaler = pkg.get('scaler') or pkg.get('adann_scaler') or scaler
                header_path = generate_adann_c_header_inline(adann_model, adann_scaler, timestamp, trained_model_dir)
                print(f"ADANN Arduino header saved: {header_path}")
                if args.arduino:
                    copy_header_to_arduino_dir('ADANN', header_path, mode_suffix)
            except Exception as e:
                print(f"⚠️ ADANN header generation skipped: {e}")
        if args.model_type == 'ADANN_LightGBM':
            try:
                # 从包装器中取出 LightGBM 子模型与其 scaler
                lgb_estimator = None
                lgb_scaler = None
                adann_estimator = None
                adann_scaler = None
                if hasattr(model, 'hybrid_model') and isinstance(model.hybrid_model, dict):
                    lgb_estimator = model.hybrid_model.get('lightgbm')
                    lgb_scaler = model.hybrid_model.get('lgb_scaler') or scaler
                    adann_estimator = model.hybrid_model.get('adann')
                    adann_scaler = model.hybrid_model.get('adann_scaler') or scaler
                if lgb_estimator is not None and lgb_scaler is not None:
                    header_path = generate_lightgbm_arduino_header(
                        lgb_estimator, lgb_scaler, f"{args.model_type}_LGB_BRANCH", timestamp, trained_model_dir
                    )
                    print(f"Hybrid LightGBM Arduino header saved: {header_path}")
                    if args.arduino:
                        copy_header_to_arduino_dir('ADANN_LightGBM', header_path, mode_suffix)
                else:
                    print("⚠️ Hybrid LightGBM header skipped: missing lgb estimator or scaler in hybrid model")
                if adann_estimator is not None and adann_scaler is not None:
                    adann_header = generate_adann_c_header_inline(adann_estimator, adann_scaler, timestamp, trained_model_dir)
                    print(f"Hybrid ADANN Arduino header saved: {adann_header}")
                    # Also copy ADANN header into the ADANN_LightGBM Arduino dir for combined inference
                    try:
                        mapping_dir = os.path.join('arduino', 'tinyml_inference', 'ADANN_LightGBM_inference')
                        os.makedirs(mapping_dir, exist_ok=True)
                        target_name = 'bsl_model_ADANN.h'
                        target_path = os.path.join(mapping_dir, target_name)
                        shutil.copyfile(adann_header, target_path)
                        # copy to latency dirs
                        for mode_dir in ['Latency_standard', 'Latency_loso']:
                            os.makedirs(os.path.join(mapping_dir, mode_dir), exist_ok=True)
                            shutil.copyfile(adann_header, os.path.join(mapping_dir, mode_dir, target_name))
                        print(f"ADANN header copied to ADANN_LightGBM Arduino dirs: {mapping_dir}")
                    except Exception as e:
                        print(f"⚠️ Failed to copy ADANN header into ADANN_LightGBM dir: {e}")
                else:
                    print("⚠️ Hybrid ADANN header skipped: missing adann estimator or scaler in hybrid model")
            except Exception as e:
                print(f"⚠️ Failed to generate hybrid LightGBM/ADANN Arduino headers: {e}")
    elif args.model_type == 'Transformer_Encoder' and args.arduino:
        # Still export header like CNN flow, but use non-Arduino TFLite conversion (SELECT_TF_OPS)
        # so that a .h is produced for testing, even though it won't compile on TFLite Micro.
        print("⚠️ Transformer_Encoder on Arduino: falling back to SELECT_TF_OPS TFLite conversion to generate header for testing.")
        print("   The generated header will not be runnable on TFLite Micro, but preserves the export/testing flow.")
        print("Reloading Keras model for TFLite conversion (attempt INT8 + SELECT_TF_OPS)...")
        model = tf.keras.models.load_model(model_path)

        # Build converter manually to allow mixed INT8 + SELECT_TF_OPS
        @tf.function
        def _model_func(x):
            return model(x)
        input_shape = [1] + list(model.input_shape[1:])
        concrete_func = _model_func.get_concrete_function(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32)
        )
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter._experimental_lower_tensor_list_ops = False
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_dataset():
            for i in range(min(100, len(X_train_data))):
                yield [X_train_data[i:i+1].astype(np.float32)]
        converter.representative_dataset = representative_dataset

        # Try full INT8 with SELECT_TF_OPS allowed (may still fail for some graphs)
        try:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            tflite_model = converter.convert()
            print("TFLite conversion succeeded with INT8 + SELECT_TF_OPS.")
        except Exception as e:
            print(f"⚠️ INT8 conversion failed ({e}); falling back to SELECT_TF_OPS with default dtypes.")
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter._experimental_lower_tensor_list_ops = False
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            # Do not force int8 I/O types; let converter keep flex/float for unsupported parts
            tflite_model = converter.convert()
            print("TFLite conversion succeeded with SELECT_TF_OPS (mixed precision).")
        header_path = generate_arduino_header_tflite(tflite_model, scaler, args.model_type, timestamp, trained_model_dir)
        tflite_path = os.path.join(trained_model_dir, f"{full_model_name}.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved (SELECT_TF_OPS): {tflite_path}")
        if args.arduino:
            copy_header_to_arduino_dir(args.model_type, header_path, mode_suffix)
    else:
        # "Save and Reload" trick to ensure model is clean for TFLite conversion
        print("Reloading Keras model for stable TFLite conversion...")
        model = tf.keras.models.load_model(model_path)

        tflite_model = convert_to_tflite(model, args.arduino, X_train_data)
        header_path = generate_arduino_header_tflite(tflite_model, scaler, args.model_type, timestamp, trained_model_dir)
        tflite_path = os.path.join(trained_model_dir, f"{full_model_name}.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved: {tflite_path}")
        if args.arduino:
            copy_header_to_arduino_dir(args.model_type, header_path, mode_suffix)
        
    # Header file generation completed above

    # Save Params (robust to both dict and Keras History)
    def _get_metric(history_obj, key: str):
        if history_obj is None:
            return 0.0
        # Keras History
        if hasattr(history_obj, 'history') and isinstance(history_obj.history, dict):
            seq = history_obj.history.get(key)
            if isinstance(seq, (list, tuple)) and len(seq) > 0:
                return float(seq[-1])
            return 0.0
        # Dict-like
        if isinstance(history_obj, dict):
            seq = history_obj.get(key)
            if isinstance(seq, (list, tuple)) and len(seq) > 0:
                return float(seq[-1])
            if isinstance(seq, (int, float)):
                return float(seq)
            return 0.0
        return 0.0

    if is_final:
        final_accuracy = _get_metric(history, 'accuracy')
    else:
        val_acc = _get_metric(history, 'val_accuracy')
        final_accuracy = val_acc if val_acc > 0 else _get_metric(history, 'accuracy')
        
    params_path = os.path.join(output_dir, f'params_{full_model_name}.json')
    with open(params_path, 'w') as f:
        json.dump({
            'model_type': args.model_type,
            'mode': 'loso_final' if is_final else 'standard',
            'timestamp': timestamp,
            'final_accuracy': final_accuracy,
            'hyperparameters': model.optimizer.get_config() if hasattr(model, 'optimizer') and hasattr(model.optimizer, 'get_config') else {}
        }, f, indent=2)
    print(f"Params saved: {params_path}")