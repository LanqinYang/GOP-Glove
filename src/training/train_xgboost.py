"""
BSL Gesture Recognition Training - XGBoost
Author: Lambert Yang
"""

import os
import numpy as np
import glob
import re
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import xgboost as xgb
from scipy.stats import skew, kurtosis
from micromlgen import port

import optuna
from datetime import datetime
import json

# Add visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set random seeds
SEED = 42
np.random.seed(SEED)

# Constants
SEQUENCE_LENGTH = 100
N_FEATURES = 5
N_CLASSES = 11
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# Full model parameters (high accuracy)
FULL_MAX_ESTIMATORS = 1000
FULL_MAX_DEPTH = 10

# Arduino-optimized parameters (small file size)
ARDUINO_MAX_ESTIMATORS = 50
ARDUINO_MAX_DEPTH = 4


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


def extract_features_full(X):
    """Extract full feature set for maximum accuracy"""
    features = []
    
    for sample in X:
        sample_features = []
        
        # For each sensor channel
        for channel in range(N_FEATURES):
            data = sample[:, channel]
            
            # Full statistical features
            sample_features.extend([
                np.mean(data),
                np.std(data),
                np.min(data),
                np.max(data),
                np.median(data),
                skew(data),
                kurtosis(data),
                np.var(data)
            ])
            
            # Rolling window features
            window_size = 10
            if len(data) >= window_size:
                rolling_means = []
                for i in range(len(data) - window_size + 1):
                    rolling_means.append(np.mean(data[i:i+window_size]))
                sample_features.extend([
                    np.mean(rolling_means),
                    np.std(rolling_means)
                ])
            else:
                sample_features.extend([0, 0])
        
        features.append(sample_features)
    
    return np.array(features)


def extract_features_arduino(X):
    """Extract Arduino-optimized features for small file size"""
    features = []
    
    for sample in X:
        sample_features = []
        
        # For each sensor channel - only extract essential features
        for channel in range(N_FEATURES):
            data = sample[:, channel]
            
            # Only basic statistical features (4 instead of 10)
            sample_features.extend([
                np.mean(data),
                np.std(data),
                np.min(data),
                np.max(data)
            ])
        
        features.append(sample_features)
    
    return np.array(features)


def objective_full(trial, X_train, y_train, X_val, y_val):
    """Full model objective function for maximum accuracy"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, FULL_MAX_ESTIMATORS),
        'max_depth': trial.suggest_int('max_depth', 3, FULL_MAX_DEPTH),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
    }
    
    try:
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=N_CLASSES,
            eval_metric='mlogloss',
            seed=SEED,
            **params
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy
    except:
        return 0.0


def objective_arduino(trial, X_train, y_train, X_val, y_val):
    """Arduino-optimized objective function for small file size"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, ARDUINO_MAX_ESTIMATORS),
        'max_depth': trial.suggest_int('max_depth', 2, ARDUINO_MAX_DEPTH),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 2),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5)
    }
    
    try:
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=N_CLASSES,
            eval_metric='mlogloss',
            seed=SEED,
            **params
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy
    except:
        return 0.0


def generate_arduino_header(model, scaler, model_type, timestamp, output_dir):
    """使用micromlgen为XGBoost模型生成Arduino兼容的C代码头文件"""
    # 使用micromlgen生成C代码
    model_code = port(model)
    
    # 获取归一化参数
    mean_values = ', '.join(f'{val:.8f}f' for val in scaler.mean_)
    scale_values = ', '.join(f'{val:.8f}f' for val in scaler.scale_)
    feature_count = len(scaler.mean_)
    
    # 生成头文件内容
    header_content = f"""/*
 * BSL Gesture Recognition Model - Arduino Optimized
 *
 * Model Type: {model_type}
 * Timestamp:  {timestamp}
 * Generated with micromlgen for Arduino compatibility
 * File Size: Optimized for <1MB Arduino memory constraints
 */

#ifndef BSL_MODEL_H_{timestamp}
#define BSL_MODEL_H_{timestamp}

#include <math.h>

// Feature count for normalization
const int BSL_MODEL_FEATURES = {feature_count};

// Normalization parameters (StandardScaler)
const float scaler_mean[BSL_MODEL_FEATURES] = {{ {mean_values} }};
const float scaler_scale[BSL_MODEL_FEATURES] = {{ {scale_values} }};

// Normalization function
void normalize_features(float* features, int length) {{
    for (int i = 0; i < length; i++) {{
        features[i] = (features[i] - scaler_mean[i]) / scaler_scale[i];
    }}
}}

// Generated XGBoost model code
{model_code}

#endif // BSL_MODEL_H_{timestamp}
"""
    
    # 创建输出目录
    model_output_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 保存头文件
    header_path = os.path.join(model_output_dir, f"bsl_model_{model_type}_{timestamp}.h")
    with open(header_path, 'w') as f:
        f.write(header_content)
        
    return header_path


def comprehensive_evaluation(model, X_test, y_test, scaler, output_dir, timestamp, history=None, class_names=None, fold_summary=None):
    """
    Comprehensive model evaluation with detailed metrics and visualizations
    """
    if class_names is None:
        class_names = [f'Gesture_{i}' for i in range(N_CLASSES)]
    
    print("\n" + "="*50)
    print("COMPREHENSIVE EVALUATION")
    print("="*50)
    
    # 1. Basic evaluation - accuracy only for XGBoost
    print("\n1. Basic Evaluation:")
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"   Test Accuracy: {test_acc:.4f}")
    
    # 2. Predictions and class probabilities
    print("\n2. Generating Predictions:")
    try:
        y_pred_proba = model.predict_proba(X_test)
    except:
        # Fallback if predict_proba not available
        y_pred_proba = np.eye(N_CLASSES)[y_pred]
    
    # 3. Class distribution analysis
    print("\n3. Class Distribution Analysis:")
    y_test_counts = Counter(y_test)
    y_pred_counts = Counter(y_pred)
    
    print("   True distribution:")
    for i in range(N_CLASSES):
        count = y_test_counts.get(i, 0)
        print(f"   {class_names[i]}: {count} samples ({count/len(y_test)*100:.1f}%)")
    
    print("   Predicted distribution:")
    for i in range(N_CLASSES):
        count = y_pred_counts.get(i, 0)
        print(f"   {class_names[i]}: {count} samples ({count/len(y_pred)*100:.1f}%)")
    
    # 4. Confusion Matrix
    print("\n4. Confusion Matrix Analysis:")
    cm = confusion_matrix(y_test, y_pred)
    
    # Most confused pairs
    most_confused = []
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if i != j and cm[i, j] > 0:
                most_confused.append((class_names[i], class_names[j], cm[i, j]))
    
    most_confused.sort(key=lambda x: x[2], reverse=True)
    print("   Top confusions:")
    for true_class, pred_class, count in most_confused[:5]:
        print(f"   {true_class} → {pred_class}: {count} times")
    
    # 5. Classification Report
    print("\n5. Classification Report:")
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 6. Per-class accuracy
    print("\n6. Per-Class Accuracy:")
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(per_class_acc):
        print(f"   {class_names[i]}: {acc:.4f}")
    
    # 7. Confidence analysis
    print("\n7. Confidence Analysis:")
    confidence_scores = np.max(y_pred_proba, axis=1)
    print(f"   Average confidence: {np.mean(confidence_scores):.4f}")
    print(f"   Min confidence: {np.min(confidence_scores):.4f}")
    print(f"   Max confidence: {np.max(confidence_scores):.4f}")
    
    # Low confidence predictions
    low_conf_threshold = 0.5
    low_conf_indices = np.where(confidence_scores < low_conf_threshold)[0]
    print(f"   Low confidence predictions (<{low_conf_threshold}): {len(low_conf_indices)}")
    
    # 8. Error analysis
    print("\n8. Error Analysis:")
    error_indices = np.where(y_pred != y_test)[0]
    print(f"   Total errors: {len(error_indices)}")
    
    if len(error_indices) > 0:
        print("   Sample errors:")
        for i in error_indices[:5]:  # Show first 5 errors
            true_label = class_names[y_test[i]]
            pred_label = class_names[y_pred[i]]
            confidence = confidence_scores[i]
            print(f"   Sample {i}: True={true_label}, Pred={pred_label}, Confidence={confidence:.4f}")
    
    # 9. Save evaluation results
    print("\n9. Saving Evaluation Results:")
    eval_results = {
        'timestamp': timestamp,
        'test_accuracy': float(test_acc),
        'class_distribution': {
            'true': {str(k): int(v) for k, v in y_test_counts.items()},
            'predicted': {str(k): int(v) for k, v in y_pred_counts.items()}
        },
        'per_class_accuracy': {class_names[i]: float(acc) for i, acc in enumerate(per_class_acc)},
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'confidence_stats': {
            'mean': float(np.mean(confidence_scores)),
            'min': float(np.min(confidence_scores)),
            'max': float(np.max(confidence_scores)),
            'std': float(np.std(confidence_scores))
        },
        'error_analysis': {
            'total_errors': int(len(error_indices)),
            'error_rate': float(len(error_indices) / len(y_test)),
            'low_confidence_predictions': int(len(low_conf_indices))
        }
    }
    
    # Save evaluation results
    eval_path = os.path.join(output_dir, f'evaluation_XGBoost_{timestamp}.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"   Evaluation results saved: {eval_path}")
    
    # 10. Generate visualizations
    print("\n10. Generating Visualizations:")
    
    # Create figure with subplots - 3x3 layout to provide space for summary metrics
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    fig.suptitle(f'BSL Gesture Recognition Evaluation - XGBoost - {timestamp}', fontsize=20)
    
    # Plot 1: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14)
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # Plot 2: Per-class accuracy
    axes[0, 1].bar(range(N_CLASSES), per_class_acc, color='skyblue')
    axes[0, 1].set_title('Per-Class Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_xticks(range(N_CLASSES))
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: Class distribution
    x = np.arange(N_CLASSES)
    width = 0.35
    true_counts = [y_test_counts.get(i, 0) for i in range(N_CLASSES)]
    pred_counts = [y_pred_counts.get(i, 0) for i in range(N_CLASSES)]
    
    axes[0, 2].bar(x - width/2, true_counts, width, label='True', color='lightcoral')
    axes[0, 2].bar(x + width/2, pred_counts, width, label='Predicted', color='lightblue')
    axes[0, 2].set_title('Class Distribution', fontsize=14)
    axes[0, 2].set_xlabel('Class')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 2].legend()
    
    # Plot 4: Confidence score distribution
    axes[1, 0].hist(confidence_scores, bins=20, alpha=0.7, color='green')
    axes[1, 0].axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(confidence_scores):.3f}')
    axes[1, 0].set_title('Confidence Score Distribution', fontsize=14)
    axes[1, 0].set_xlabel('Confidence Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Plot 5: Precision, Recall, F1-Score by Class
    metrics = ['precision', 'recall', 'f1-score']
    metric_values = np.array([[report[class_names[i]][metric] for i in range(N_CLASSES)] 
                             for metric in metrics])
    
    x = np.arange(N_CLASSES)
    width = 0.25
    for i, metric in enumerate(metrics):
        axes[1, 1].bar(x + i*width, metric_values[i], width, label=metric, alpha=0.8)
    
    axes[1, 1].set_title('Precision, Recall, F1-Score by Class', fontsize=14)
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1)
    
    # Plot 6: Errors by Class
    correct = (y_pred == y_test)
    error_by_class = [np.sum((y_test == i) & ~correct) for i in range(N_CLASSES)]
    axes[1, 2].bar(range(N_CLASSES), error_by_class, color='red', alpha=0.7)
    axes[1, 2].set_title('Errors by Class', fontsize=14)
    axes[1, 2].set_xlabel('Class')
    axes[1, 2].set_ylabel('Number of Errors')
    axes[1, 2].set_xticks(range(N_CLASSES))
    axes[1, 2].set_xticklabels(class_names, rotation=45, ha='right')
    
    # Plots 7 & 8: Placeholders for training history (not available for XGBoost)
    for i in range(2):
        axes[2, i].text(0.5, 0.5, 'Training History\nNot Available\n(XGBoost)', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[2, i].transAxes, fontsize=14)
        axes[2, i].set_title(f'Training History {i+1}', fontsize=14)
        axes[2, i].axis('off')

    # Plot 9: Overall Metrics Summary
    axes[2, 2].axis('off')
    axes[2, 2].set_title('Overall Metrics', fontsize=14, pad=20)
    
    macro_f1 = report['macro avg']['f1-score']
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']

    metrics_text = (
        f"Overall Accuracy: {test_acc:.4f}\n\n"
        f"Macro Average:\n"
        f"  - F1-Score:   {macro_f1:.4f}\n"
        f"  - Precision:  {macro_precision:.4f}\n"
        f"  - Recall:     {macro_recall:.4f}"
    )

    if fold_summary:
        metrics_text += (
            f"\n\nLOSO Summary:\n"
            f"  - Avg Accuracy: {fold_summary['avg_accuracy']:.4f}\n"
            f"  - Std Accuracy: {fold_summary['std_accuracy']:.4f}\n"
            f"  - Avg F1:       {fold_summary['avg_f1']:.4f}\n"
            f"  - Std F1:       {fold_summary['std_f1']:.4f}"
        )

    axes[2, 2].text(0.5, 0.5, metrics_text, 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    fontsize=14,
                    fontfamily='monospace',
                    transform=axes[2, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", fc='aliceblue', ec='black', lw=1))

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save visualization
    viz_path = os.path.join(output_dir, f'evaluation_plots_XGBoost_{timestamp}.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"   Evaluation plots saved: {viz_path}")
    
    plt.close()
    
    # 11. Save detailed predictions
    predictions_data = {
        'predictions': [int(x) for x in y_pred.tolist()],
        'true_labels': [int(x) for x in y_test.tolist()],
        'probabilities': [[float(x) for x in row] for row in y_pred_proba.tolist()],
        'confidence_scores': [float(x) for x in confidence_scores.tolist()],
        'correct_predictions': [bool(x) for x in correct.tolist()]
    }
    
    pred_path = os.path.join(output_dir, f'predictions_XGBoost_{timestamp}.json')
    with open(pred_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    print(f"   Detailed predictions saved: {pred_path}")
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    
    return eval_results


def train_model(csv_dir, output_dir, n_trials=50, model_type="XGBoost", arduino_mode=False):
    """训练XGBoost模型 - 支持完整版本和Arduino优化版本"""
    
    if arduino_mode:
        print(f"\n🤖 开始训练Arduino优化的{model_type}模型...")
        print(f"目标：生成小于1MB的Arduino兼容.h文件")
        print(f"限制：最大{ARDUINO_MAX_ESTIMATORS}棵树，最大深度{ARDUINO_MAX_DEPTH}")
        model_suffix = "_Arduino"
    else:
        print(f"\n🚀 开始训练完整版{model_type}模型...")
        print(f"目标：最大化模型精度")
        print(f"参数：最大{FULL_MAX_ESTIMATORS}棵树，最大深度{FULL_MAX_DEPTH}")
        model_suffix = ""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    X, y = load_data(csv_dir)
    print(f"加载数据：{len(X)} 个样本")
    
    # 根据模式选择特征提取方法
    if arduino_mode:
        X_features = extract_features_arduino(X)
        print(f"Arduino优化特征提取：{X_features.shape[1]} 个特征")
    else:
        X_features = extract_features_full(X)
        print(f"完整特征提取：{X_features.shape[1]} 个特征")
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=SEED, stratify=y_train
    )
    
    # 数据归一化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 优化超参数
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n开始超参数优化 (试验次数: {n_trials})...")
    study = optuna.create_study(direction='maximize')
    
    # 智能早停策略
    def smart_early_stop_callback(study, trial):
        # 简单策略：准确率达到完美时停止
        if study.best_value >= 1.0:
            print(f"🎯 早停: 验证准确率达到 {study.best_value:.4f} (≥100%) 在第 {trial.number} 次试验")
            study.stop()
    
    # 根据模式选择目标函数
    if arduino_mode:
        study.optimize(
            lambda trial: objective_arduino(trial, X_train_scaled, y_train, X_val_scaled, y_val),
            n_trials=n_trials,
            callbacks=[smart_early_stop_callback],
            show_progress_bar=True
        )
    else:
        study.optimize(
            lambda trial: objective_full(trial, X_train_scaled, y_train, X_val_scaled, y_val),
            n_trials=n_trials,
            callbacks=[smart_early_stop_callback],
            show_progress_bar=True
        )
    
    # 使用最佳参数训练最终模型
    best_params = study.best_params
    print(f"\n最佳参数：{best_params}")
    
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=N_CLASSES,
        eval_metric='mlogloss',
        seed=SEED,
        **best_params
    )
    
    model.fit(X_train_scaled, y_train)
    
    # 评估模型
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    val_acc = accuracy_score(y_val, model.predict(X_val_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    
    print(f"\n模型性能：")
    print(f"训练准确率: {train_acc:.4f}")
    print(f"验证准确率: {val_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    
    # 更新模型类型名称
    final_model_type = model_type + model_suffix
    
    # 生成Arduino头文件
    header_path = generate_arduino_header(model, scaler, final_model_type, timestamp, output_dir)
    
    # 检查文件大小
    file_size = os.path.getsize(header_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"\n生成的头文件：{header_path}")
    print(f"文件大小：{file_size_mb:.2f} MB")
    
    if arduino_mode:
        if file_size_mb > 1.0:
            print("⚠️  警告：Arduino模式文件仍超过1MB，可能需要进一步调整参数")
        else:
            print("✅ 文件大小符合Arduino要求！")
    else:
        print(f"ℹ️  完整版模型文件大小：{file_size_mb:.2f} MB")
    
    # 保存模型参数
    params_path = os.path.join(output_dir, final_model_type, f"params_{final_model_type}_{timestamp}.json")
    with open(params_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'model_type': final_model_type,
            'timestamp': timestamp,
            'arduino_mode': arduino_mode,
            'performance': {
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc
            },
            'model_info': {
                'n_features': X_features.shape[1],
                'n_classes': N_CLASSES,
                'file_size_mb': file_size_mb,
                'max_estimators': ARDUINO_MAX_ESTIMATORS if arduino_mode else FULL_MAX_ESTIMATORS,
                'max_depth': ARDUINO_MAX_DEPTH if arduino_mode else FULL_MAX_DEPTH
            }
        }, f, indent=2)
    
    # 综合评估
    model_output_dir = os.path.join(output_dir, final_model_type)
    comprehensive_evaluation(model, X_test_scaled, y_test, scaler, model_output_dir, timestamp, 
                           class_names=[f'Gesture_{i}' for i in range(N_CLASSES)])
    
    return model, scaler, header_path


def train_loso_model(csv_dir, output_dir, model_type, n_trials=50, arduino_mode=False):
    """
    使用“留一被试法”（LOSO）交叉验证来训练和评估XGBoost模型。
    """
    model_suffix = "_Arduino" if arduino_mode else ""
    final_model_type = model_type + model_suffix
    
    print(f"\n🚀 开始对 {final_model_type} 模型进行LOSO训练...")
    X, y, subjects = load_data(csv_dir)
    
    unique_subjects = np.unique(subjects)
    unique_subjects = unique_subjects[unique_subjects != -1]

    if len(unique_subjects) < 2:
        print("❌错误：LOSO训练至少需要2个被试。请检查您的数据和文件名。")
        return

    print(f"从 {len(unique_subjects)} 个被试中加载了 {len(X)} 个样本。")

    all_fold_evaluations = []
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Static']
    model_output_dir = os.path.join(output_dir, final_model_type)
    os.makedirs(model_output_dir, exist_ok=True)
    
    extract_features = extract_features_arduino if arduino_mode else extract_features_full
    
    for subject_id in unique_subjects:
        print(f"\n{'='*20} LOSO轮次：在被试 {subject_id} 上测试 {'='*20}")
        
        test_indices = np.where(subjects == subject_id)[0]
        train_indices = np.where(subjects != subject_id)[0]
        
        X_train_full, y_train_full = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        X_train_full_feat = extract_features(X_train_full)
        X_test_feat = extract_features(X_test)
        
        X_train_feat, X_val_feat, y_train, y_val = train_test_split(
            X_train_full_feat, y_train_full, test_size=VAL_SIZE, random_state=SEED, stratify=y_train_full
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_feat)
        X_val_scaled = scaler.transform(X_val_feat)
        X_test_scaled = scaler.transform(X_test_feat)
        
        print(f"为被试 {subject_id} 优化超参数...")
        study = optuna.create_study(direction='maximize')
        objective_func = objective_arduino if arduino_mode else objective_full
        study.optimize(lambda trial: objective_func(trial, X_train_scaled, y_train, X_val_scaled, y_val), n_trials=n_trials)
        
        best_params = study.best_params
        model = xgb.XGBClassifier(objective='multi:softmax', num_class=N_CLASSES, eval_metric='mlogloss', seed=SEED, **best_params)
        model.fit(X_train_scaled, y_train)
        
        fold_timestamp = f"{final_model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_subject_{subject_id}"
        eval_results = comprehensive_evaluation(model, X_test_scaled, y_test, scaler, model_output_dir, fold_timestamp, class_names=class_names)
        all_fold_evaluations.append(eval_results)
    
    accuracies = [eval_res['test_accuracy'] for eval_res in all_fold_evaluations]
    f1_scores = [eval_res['classification_report']['macro avg']['f1-score'] for eval_res in all_fold_evaluations]
    
    summary = {
        'model_type': final_model_type,
        'mean_test_accuracy': np.mean(accuracies),
        'std_test_accuracy': np.std(accuracies),
        'mean_macro_f1_score': np.mean(f1_scores),
        'std_macro_f1_score': np.std(f1_scores),
        'fold_evaluations': all_fold_evaluations
    }
    
    summary_path = os.path.join(output_dir, f'loso_summary_{final_model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"LOSO摘要报告已保存至: {summary_path}")

    # --- 最终部署模型训练 ---
    print(f"\n{'='*20} 训练最终部署模型 {'='*20}")
    
    X_features_final = extract_features(X)
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        X_features_final, y, test_size=0.1, random_state=SEED, stratify=y
    )
    
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train_final, y_train_final, test_size=VAL_SIZE, random_state=SEED, stratify=y_train_final
    )
    
    final_scaler = StandardScaler()
    X_train_opt_scaled = final_scaler.fit_transform(X_train_opt)
    X_val_opt_scaled = final_scaler.transform(X_val_opt)

    print("为最终模型优化超参数...")
    final_study = optuna.create_study(direction='maximize')
    final_objective = objective_arduino if arduino_mode else objective_full
    final_study.optimize(lambda trial: final_objective(trial, X_train_opt_scaled, y_train_opt, X_val_opt_scaled, y_val_opt), n_trials=n_trials)
    
    final_best_params = final_study.best_params
    final_model = xgb.XGBClassifier(objective='multi:softmax', num_class=N_CLASSES, eval_metric='mlogloss', seed=SEED, **final_best_params)
    
    X_train_final_scaled = final_scaler.fit_transform(X_train_final)
    X_test_final_scaled = final_scaler.transform(X_test_final)
    final_model.fit(X_train_final_scaled, y_train_final)

    timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_final_deployment_model"
    header_path = generate_arduino_header(final_model, final_scaler, final_model_type, timestamp, output_dir)
    
    fold_summary_for_plot = {
        'avg_accuracy': np.mean(accuracies), 'std_accuracy': np.std(accuracies), 
        'avg_f1': np.mean(f1_scores), 'std_f1': np.std(f1_scores)
    }
    comprehensive_evaluation(
        final_model, X_test_final_scaled, y_test_final, final_scaler, 
        model_output_dir, f"{final_model_type}_{timestamp}", class_names=class_names, fold_summary=fold_summary_for_plot
    )
    print("✅ 最终部署模型训练完成！")
    return header_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练XGBoost BSL手势识别模型')
    parser.add_argument('--csv_dir', default='datasets/gesture_csv', help='CSV数据文件夹路径')
    parser.add_argument('--output_dir', default='models/trained', help='模型输出路径')
    parser.add_argument('--model_type', default='XGBoost', 
                       choices=['1D_CNN', 'XGBoost', 'CNN_LSTM', 'Transformer_Encoder'],
                       help='Type of model to train')
    parser.add_argument('--n_trials', type=int, default=50, help='超参数优化试验次数')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs parameter (ignored for XGBoost)')
    parser.add_argument('--arduino', action='store_true', 
                       help='使用Arduino优化模式（文件<1MB，但精度稍低）')
    args = parser.parse_args()
    
    train_model(args.csv_dir, args.output_dir, args.n_trials, args.model_type, args.arduino) 