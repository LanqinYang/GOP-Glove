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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
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

# --- 滑动窗口标准化 (基于最新研究的电极位移鲁棒性方法) ---
def apply_sliding_window_normalization(trial_data, window_size=200, overlap_ratio=0.5):
    """
    应用滑动窗口标准化 - 基于2025年研究的电极位移鲁棒性方法
    
    参数:
    - trial_data: 形状为(time_steps, n_channels)的数组
    - window_size: 滑动窗口大小
    - overlap_ratio: 窗口重叠比例
    
    返回:
    - 标准化后的数据
    """
    if len(trial_data) == 0:
        return trial_data
    
    trial_data = np.array(trial_data)
    normalized_data = np.zeros_like(trial_data)
    
    step_size = int(window_size * (1 - overlap_ratio))
    
    for channel in range(trial_data.shape[1]):
        channel_data = trial_data[:, channel]
        
        # 对每个滑动窗口进行z-score标准化
        for start in range(0, len(channel_data), step_size):
            end = min(start + window_size, len(channel_data))
            
            if end - start < window_size // 2:  # 窗口太小则跳过
                continue
                
            window_data = channel_data[start:end]
            window_mean = np.mean(window_data)
            window_std = np.std(window_data) + 1e-8
            
            # 标准化当前窗口
            normalized_window = (window_data - window_mean) / window_std
            
            # 处理重叠部分的平滑融合
            if start == 0:
                normalized_data[start:end, channel] = normalized_window
            else:
                # 重叠区域使用加权平均
                overlap_start = start
                overlap_end = min(start + int(window_size * overlap_ratio), end)
                
                if overlap_end > overlap_start:
                    # 线性权重
                    weights = np.linspace(0, 1, overlap_end - overlap_start)
                    normalized_data[overlap_start:overlap_end, channel] = (
                        (1 - weights) * normalized_data[overlap_start:overlap_end, channel] +
                        weights * normalized_window[:overlap_end - overlap_start]
                    )
                
                # 非重叠区域直接赋值
                if overlap_end < end:
                    normalized_data[overlap_end:end, channel] = normalized_window[overlap_end - start:]
    
    return normalized_data

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
            with open('configs/config.yaml', 'r') as f:
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
    scale_range = augment_params['scale_range']
    for i in range(X_augmented.shape[0]):
        # Apply scaling with specified probability
        if np.random.rand() < augment_prob:
            scale_factor = np.random.uniform(scale_range[0], scale_range[1])
            X_augmented[i] = X_augmented[i] * scale_factor

    # 3. Combine original and augmented data
    X_final = np.vstack([X_train, X_augmented])
    y_final = np.concatenate([y_train, y_to_augment])

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

def generate_arduino_header_tflite(tflite_model, scaler, model_type, timestamp, output_dir):
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

    # Always enable SELECT_TF_OPS for compatibility with complex layers
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    
    if arduino_mode:
        print("Applying full integer quantization for Arduino...")
        def representative_dataset():
            for i in range(min(100, len(X_train_scaled))):
                yield [X_train_scaled[i:i+1].astype(np.float32)]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8, 
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    return converter.convert()

# --- Hyperparameter Optimization ---

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
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            TFKerasPruningCallback(trial, "val_accuracy")
        ]
        epochs = args.epochs if not args.arduino else min(args.epochs, 20)
        model.fit(
            X_train, y_train_aug,
            batch_size=params['batch_size'],
            epochs=epochs,
            validation_data=(X_val_scaled, y_val),
            callbacks=callbacks,
            verbose=0
        )
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

# --- Main Pipeline ---

def run_training_pipeline(args, model_creator):
    """
    A unified training pipeline for any given model.
    """
    # 1. Setup paths
    training_mode = "loso" if args.loso else "standard"
    optimization_mode = "arduino" if args.arduino else "full"
    final_model_type = f"{args.model_type}_{training_mode}_{optimization_mode}"
    
    output_dir = os.path.join('outputs', args.model_type, training_mode, optimization_mode)
    trained_model_dir = os.path.join(args.output_dir, args.model_type, training_mode, optimization_mode)
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

    # Hyperparameter Optimization
    objective_func = objective_xgb if args.model_type == 'XGBoost' else objective_tf
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective_func(trial, args, model_creator, X_train_feat, y_train, X_val_feat, y_val), n_trials=args.n_trials)
    
    # Train final model
    best_params = study.best_params
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print("Best parameters:", best_params)
    final_model = model_creator.create_model(best_params, args.arduino)
    
    epochs = args.epochs
    if args.model_type != 'XGBoost':
         callbacks = [EarlyStopping(patience=15, restore_best_weights=True)]
         history = final_model.fit(X_train_feat, y_train, validation_data=(X_val_feat, y_val), epochs=epochs, callbacks=callbacks, verbose=1)
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
    objective_func = objective_xgb if args.model_type == 'XGBoost' else objective_tf
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
            study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
            study.optimize(lambda trial: objective_func(trial, args, model_creator, X_train_feat, y_train, X_val_feat, y_val), n_trials=args.n_trials)
        else:
            # TensorFlow models: Pass raw data for full optimization
            X_train_orig, X_val_orig, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=VAL_SIZE, random_state=SEED, stratify=y_train_full)
            study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
            study.optimize(lambda trial: objective_func(trial, args, model_creator, X_train_orig, y_train, X_val_orig, y_val), n_trials=args.n_trials)
            
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
        if args.model_type != 'XGBoost':
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
            
            # For TensorFlow models, use the optimized training data
            if 'X_train_final_feat' in locals():
                history_fold = fold_model.fit(X_train_final_feat, y_train_final, epochs=epochs, callbacks=callbacks, verbose=0)
            else:
                history_fold = fold_model.fit(X_train_feat, y_train, validation_data=(X_val_feat, y_val), epochs=epochs, callbacks=callbacks, verbose=0)
        else:
            fold_model.fit(X_train_feat, y_train, eval_set=[(X_val_feat, y_val)], early_stopping_rounds=15, verbose=False)
            
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
    if args.model_type != 'XGBoost':
        print(f"Training final Keras model for {epochs} epochs...")
        # No callbacks like EarlyStopping, as there's no validation set. Train for the full duration.
        history = final_model.fit(X_all_feat, y, epochs=epochs, verbose=1)
    else:
        print("Training final XGBoost model...")
        final_model.fit(X_all_feat, y)
    
    # 4. Save the final, deployable artifacts
    save_artifacts(final_model, scaler_final, args, trained_model_dir, output_dir, history, X_all_feat, is_final=True)

    # 5. Generate and save the final summary plot
    generate_loso_summary_plots(history, all_fold_evaluations, output_dir, f"{args.model_type}_loso_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

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
    
    # Save Keras model / pickle model
    if args.model_type != 'XGBoost':
        model_path = os.path.join(trained_model_dir, f"{full_model_name}.keras")
        model.save(model_path)
    else:
        model_path = os.path.join(trained_model_dir, f"{full_model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
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
    elif args.model_type in ['ADANN', 'ADANN_LightGBM']:
        print(f"PyTorch model ({args.model_type}): Skipping TFLite conversion")
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
        
    # Header file generation completed above

    # Save Params
    if is_final:
        # For the final LOSO model, there is no validation set, only training accuracy.
        final_accuracy = history.history['accuracy'][-1] if history and 'accuracy' in history.history else 0
    else:
        # For standard models, we report the validation accuracy.
        final_accuracy = history.history['val_accuracy'][-1] if history and 'val_accuracy' in history.history else 0
        
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