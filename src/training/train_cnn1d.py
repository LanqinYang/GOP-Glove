"""
BSL Gesture Recognition Training - CNN-1D
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
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, Sequential # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

import optuna
from datetime import datetime
import json

# Add visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set random seeds
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Constants
SEQUENCE_LENGTH = 100
N_FEATURES = 5
N_CLASSES = 11
TEST_SIZE = 0.2
VAL_SIZE = 0.2


def load_data(csv_dir):
    """Load and preprocess data"""
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    all_data, all_labels = [], []
    
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
                gesture_id = int(re.search(r'gesture_(\d+)', csv_file).group(1))
                all_data.append(data)
                all_labels.append(gesture_id)
        except:
            continue
    
    return np.array(all_data), np.array(all_labels)


def create_model(params, arduino_mode=False):
    """Create CNN model with flexible architecture"""
    model = Sequential()
    model.add(layers.Input(shape=(SEQUENCE_LENGTH, N_FEATURES)))
    
    # Arduino模式使用简化架构
    if arduino_mode:
        # Arduino优化：固定简单架构
        model.add(layers.Conv1D(16, 3, activation='relu'))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Conv1D(32, 3, activation='relu'))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(32, activation='relu'))
        if params.get('use_dropout', False):
            model.add(layers.Dropout(0.3))
        model.add(layers.Dense(N_CLASSES, activation='softmax'))
    else:
        # 标准模式：完整架构
        for i in range(params['n_conv_layers']):
            filters = params[f'conv{i+1}_filters']
            kernel_size = params[f'conv{i+1}_kernel']
            activation = params['activation']
            
            model.add(layers.Conv1D(filters, kernel_size, activation=activation))
            
            # Optional BatchNormalization
            if params['use_batch_norm']:
                model.add(layers.BatchNormalization())
                
            model.add(layers.MaxPooling1D(2))
            
            # Optional Dropout
            if params['use_conv_dropout']:
                model.add(layers.Dropout(params['conv_dropout']))
        
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(params['dense_units'], activation=params['activation']))
        
        # Optional Dense Dropout
        if params['use_dense_dropout']:
            model.add(layers.Dropout(params['dense_dropout']))
            
        model.add(layers.Dense(N_CLASSES, activation='softmax'))
    
    # 编译模型
    lr = 1e-3 if arduino_mode else params['learning_rate']
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def generate_arduino_header(tflite_model, scaler, model_type, timestamp, output_dir):
    """生成一个简洁的Arduino头文件，包含模型和归一化参数。"""
    model_hex = ', '.join(f'0x{b:02x}' for b in tflite_model)
    mean_values = ', '.join(f'{val:.8f}f' for val in scaler.mean_)
    scale_values = ', '.join(f'{val:.8f}f' for val in scaler.scale_)
    
    header_content = f"""/*
 * BSL Gesture Recognition Model
 *
 * Model Type: {model_type}
 * Timestamp:  {timestamp}
 * Model Size: {len(tflite_model)} bytes
 */

#ifndef BSL_MODEL_H_{timestamp}
#define BSL_MODEL_H_{timestamp}

// Feature count for normalization
const int BSL_MODEL_FEATURES = {len(scaler.mean_)};

// Normalization parameters (StandardScaler)
const float scaler_mean[BSL_MODEL_FEATURES] = {{ {mean_values} }};
const float scaler_scale[BSL_MODEL_FEATURES] = {{ {scale_values} }};

// TFLite model data
alignas(16) const unsigned char model_data[] = {{
    {model_hex}
}};
const unsigned int model_data_len = {len(tflite_model)};

#endif // BSL_MODEL_H_{timestamp}
"""
    
    arduino_dir = os.path.join(output_dir, model_type)
    os.makedirs(arduino_dir, exist_ok=True)
    
    header_path = os.path.join(arduino_dir, f"bsl_model_{model_type}_{timestamp}.h")
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    return header_path


def objective(trial, X_train, y_train, X_val, y_val, arduino_mode=False):
    """Optuna objective function with enhanced search space"""
    
    if arduino_mode:
        # Arduino模式：简化参数空间
        params = {
            'use_dropout': trial.suggest_categorical('use_dropout', [True, False]),
            'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32])
        }
        epochs = 10  # Arduino模式使用更少epochs
    else:
        # 标准模式：完整参数空间
        params = {
            'n_conv_layers': trial.suggest_int('n_conv_layers', 2, 4),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
            'use_conv_dropout': trial.suggest_categorical('use_conv_dropout', [True, False]),
            'use_dense_dropout': trial.suggest_categorical('use_dense_dropout', [True, False]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'swish']),
            'dense_units': trial.suggest_int('dense_units', 32, 128),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
        }
        
        # Conv layer parameters
        for i in range(params['n_conv_layers']):
            params[f'conv{i+1}_filters'] = trial.suggest_int(f'conv{i+1}_filters', 16, 128)
            # Kernel size should cover ~10% of sequence (3-15, step=2)
            params[f'conv{i+1}_kernel'] = trial.suggest_int(f'conv{i+1}_kernel', 3, 15, step=2)
        
        # Dropout parameters (only if used)
        if params['use_conv_dropout']:
            params['conv_dropout'] = trial.suggest_float('conv_dropout', 0.1, 0.5)
        else:
            params['conv_dropout'] = 0.0
            
        if params['use_dense_dropout']:
            params['dense_dropout'] = trial.suggest_float('dense_dropout', 0.2, 0.6)
        else:
            params['dense_dropout'] = 0.0
        
        epochs = 20  # 标准模式epochs
    
    try:
        model = create_model(params, arduino_mode)
        history = model.fit(
            X_train, y_train,
            batch_size=params['batch_size'],
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )
        return max(history.history['val_accuracy'])
    except:
        return 0.0


def objective_arduino(trial, X_train, y_train, X_val, y_val):
    """Arduino优化的目标函数"""
    return objective(trial, X_train, y_train, X_val, y_val, arduino_mode=True)


def comprehensive_evaluation(model, X_test, y_test, scaler, output_dir, timestamp, history=None, class_names=None):
    """
    Comprehensive model evaluation with detailed metrics and visualizations
    """
    if class_names is None:
        class_names = [f'Gesture_{i}' for i in range(N_CLASSES)]
    
    print("\n" + "="*50)
    print("COMPREHENSIVE EVALUATION")
    print("="*50)
    
    # 1. Basic evaluation - loss and accuracy
    print("\n1. Basic Evaluation:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    
    # 2. Predictions and class probabilities
    print("\n2. Generating Predictions:")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
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
        'test_loss': float(test_loss),
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
    
    # Add training history if available
    if history is not None:
        eval_results['training_history'] = {
            'epochs_trained': len(history.history['accuracy']),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'best_val_accuracy_epoch': int(np.argmax(history.history['val_accuracy']) + 1),
            'train_accuracy_history': [float(x) for x in history.history['accuracy']],
            'val_accuracy_history': [float(x) for x in history.history['val_accuracy']],
            'train_loss_history': [float(x) for x in history.history['loss']],
            'val_loss_history': [float(x) for x in history.history['val_loss']]
        }
    
    # Save evaluation results
    eval_path = os.path.join(output_dir, f'evaluation_{timestamp}.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"   Evaluation results saved: {eval_path}")
    
    # 10. Generate visualizations
    print("\n10. Generating Visualizations:")
    
    # Create figure with subplots - 3x3 layout to provide space for summary metrics
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    fig.suptitle(f'BSL Gesture Recognition Evaluation - {timestamp}', fontsize=20)
    
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
    
    # Plots 7 & 8: Training history
    if history is not None:
        epochs_range = range(1, len(history.history['accuracy']) + 1)
        axes[2, 0].plot(epochs_range, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[2, 0].plot(epochs_range, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[2, 0].set_title('Training & Validation Accuracy', fontsize=14)
        axes[2, 0].set_xlabel('Epochs')
        axes[2, 0].set_ylabel('Accuracy')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_ylim(0, 1.05)
        
        axes[2, 1].plot(epochs_range, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[2, 1].plot(epochs_range, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[2, 1].set_title('Training & Validation Loss', fontsize=14)
        axes[2, 1].set_xlabel('Epochs')
        axes[2, 1].set_ylabel('Loss')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_yscale('log')
    else:
        for i in range(2):
            axes[2, i].text(0.5, 0.5, 'Training History\nNot Available', 
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
    viz_path = os.path.join(output_dir, f'evaluation_plots_{timestamp}.png')
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
    
    pred_path = os.path.join(output_dir, f'predictions_{timestamp}.json')
    with open(pred_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    print(f"   Detailed predictions saved: {pred_path}")
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    
    return eval_results


def train_model(csv_dir, output_dir, model_type, n_trials=100, epochs=50, arduino_mode=False):
    """主训练函数"""
    if arduino_mode:
        print(f"\n🤖 开始训练Arduino优化的{model_type}模型...")
        print(f"目标：生成小于1MB的Arduino兼容.h文件")
        model_suffix = "_Arduino"
    else:
        print(f"\n🚀 开始训练完整版{model_type}模型...")
        print(f"目标：最大化模型精度")
        model_suffix = ""

    print(f"Loading data from {csv_dir}...")
    X, y = load_data(csv_dir)
    print(f"Loaded {len(X)} samples")
    
    # Split data first
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VAL_SIZE, random_state=SEED, stratify=y_temp)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Fit scaler ONLY on training data
    print("Fitting scaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    
    # Transform validation and test sets using the fitted scaler
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Optimize hyperparameters
    print(f"Optimizing hyperparameters with {n_trials} trials...")
    study = optuna.create_study(direction='maximize')
    
    # 智能早停策略
    def smart_early_stop_callback(study, trial):
        # 简单策略：准确率达到完美时停止
        if study.best_value >= 1.0:
            print(f"🎯 早停: 验证准确率达到 {study.best_value:.4f} (≥100%) 在第 {trial.number} 次试验")
            study.stop()
    
    # 根据模式选择目标函数
    if arduino_mode:
        objective_func = lambda trial: objective_arduino(trial, X_train_scaled, y_train, X_val_scaled, y_val)
    else:
        objective_func = lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val)
    
    study.optimize(objective_func, n_trials=n_trials, callbacks=[smart_early_stop_callback])
    
    best_params = study.best_params
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model
    print("Training final model...")
    model = create_model(best_params, arduino_mode)
    
    # Arduino模式使用更少的epochs和更简单的回调
    if arduino_mode:
        final_epochs = min(epochs, 20)  # Arduino模式最多20个epochs
        batch_size = best_params.get('batch_size', 16)
        callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
    else:
        final_epochs = epochs
        batch_size = best_params.get('batch_size', 32)
        callbacks = []
        if epochs > 30:
            callbacks = [EarlyStopping(patience=15, restore_best_weights=True, verbose=1)]
    
    history = model.fit(
        X_train_scaled, y_train,
        batch_size=batch_size,
        epochs=final_epochs,
        validation_data=(X_val_scaled, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Create model-specific output directory
    final_model_type = model_type + model_suffix
    model_output_dir = os.path.join(output_dir, final_model_type)
    os.makedirs(model_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nUsing model type: {final_model_type}")
    print(f"Saving to: {model_output_dir}")
    
    # Save model (H5 format for compatibility)
    model_path = os.path.join(model_output_dir, f"bsl_model_{final_model_type}_{timestamp}.h5")
    model.save(model_path)
    
    # Save scaler with timestamp
    scaler_path = os.path.join(model_output_dir, f'scaler_{final_model_type}_{timestamp}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save parameters
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    params_path = os.path.join(model_output_dir, f'params_{final_model_type}_{timestamp}.json')
    with open(params_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'model_type': final_model_type,
            'timestamp': timestamp,
            'arduino_mode': arduino_mode,
            'performance': {
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss)
            }
        }, f, indent=2)

    # 应用量化 - 使用函数包装解决兼容性问题
    @tf.function
    def model_func(x):
        return model(x)
    
    # 获取输入规格
    input_shape = [1, SEQUENCE_LENGTH, N_FEATURES]
    concrete_func = model_func.get_concrete_function(
        tf.TensorSpec(shape=input_shape, dtype=tf.float32)
    )
    
    # 使用标准的from_concrete_functions方法
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    if arduino_mode:
        print("应用Arduino优化量化...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    else:
        print("应用标准量化...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 直接转换，不使用任何异常处理回退
    tflite_model = converter.convert()

    # 打印模型大小
    model_size = len(tflite_model)
    print(f"TFLite模型大小: {model_size} bytes ({model_size/1024:.1f} KB)")
    
    # 生成Arduino头文件
    header_path = generate_arduino_header(tflite_model, scaler, final_model_type, timestamp, output_dir)
    
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

    # Create TFLite file
    tflite_path = os.path.join(model_output_dir, f"bsl_model_{final_model_type}_{timestamp}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model: {tflite_path} ({len(tflite_model)/1024:.1f} KB)")

    # COMPREHENSIVE EVALUATION
    # Define class names for better readability in evaluation reports
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Static']
    comprehensive_evaluation(model, X_test_scaled, y_test, scaler, model_output_dir, f"{final_model_type}_{timestamp}", history, class_names)
    
    # 重新获取测试准确率和损失，因为它们可能在评估函数中计算
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    # 更新参数文件，包含文件大小
    with open(params_path, 'r+') as f:
        params_data = json.load(f)
        params_data['file_size_mb'] = file_size_mb
        f.seek(0)
        json.dump(params_data, f, indent=2)
        f.truncate()

    print(f"\nModel saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")
    print(f"Arduino header: {header_path}")
    print(f"参数文件: {params_path}")
    print(f"测试准确率: {test_accuracy:.4f}")
    print(f"Evaluation results: {model_output_dir}/evaluation_{final_model_type}_{timestamp}.json")
    print(f"Evaluation plots: {model_output_dir}/evaluation_plots_{final_model_type}_{timestamp}.png")
    
    return header_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', default='datasets/gesture_csv')
    parser.add_argument('--output_dir', default='models/trained')
    parser.add_argument('--model_type', required=True,
                       choices=['1D_CNN', 'XGBoost', 'CNN_LSTM', 'Transformer_Encoder'],
                       help='Type of model to train (required)')
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--arduino', action='store_true',
                       help='使用Arduino优化模式（文件<1MB，但精度稍低）')
    args = parser.parse_args()
    
    train_model(args.csv_dir, args.output_dir, args.model_type, args.n_trials, args.epochs, args.arduino)