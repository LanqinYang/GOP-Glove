"""
Comprehensive model evaluation module
"""

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

def comprehensive_evaluation(model, X_test, y_test, scaler, output_dir, timestamp, history=None, class_names=None, fold_summary=None):
    """
    Comprehensive model evaluation with detailed metrics and visualizations
    """
    if class_names is None:
        # Fallback if class_names are not provided
        n_classes = len(np.unique(y_test))
        class_names = [f'Gesture_{i}' for i in range(n_classes)]
    
    print("\n" + "="*50)
    print("COMPREHENSIVE EVALUATION")
    print("="*50)
    
    # 1. Basic evaluation - loss and accuracy
    print("\n1. Basic Evaluation:")
    try:
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
    except Exception: # Handle non-Keras models like XGBoost
        from sklearn.metrics import accuracy_score
        y_pred_for_acc = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_for_acc)
        test_loss = -1 # Not applicable
        print(f"   Test Accuracy: {test_acc:.4f} (Loss not applicable for this model type)")

    
    # 2. Predictions and class probabilities
    print("\n2. Generating Predictions:")
    
    # Correctly handle predictions for both Keras and Scikit-learn models
    try:
        # This works for scikit-learn models like XGBoost
        y_pred_proba = model.predict_proba(X_test)
    except AttributeError:
        # This is the fallback for Keras models
        y_pred_proba = model.predict(X_test)

    # Derive integer class predictions from the probabilities
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 3. Class distribution analysis
    print("\n3. Class Distribution Analysis:")
    y_test_counts = Counter(y_test)
    y_pred_counts = Counter(y_pred)
    
    print("   True distribution:")
    for i in range(len(class_names)):
        count = y_test_counts.get(i, 0)
        print(f"   {class_names[i]}: {count} samples ({count/len(y_test)*100:.1f}%)")
    
    print("   Predicted distribution:")
    for i in range(len(class_names)):
        count = y_pred_counts.get(i, 0)
        print(f"   {class_names[i]}: {count} samples ({count/len(y_pred)*100:.1f}%)")
    
    # 4. Confusion Matrix
    print("\n4. Confusion Matrix Analysis:")
    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    
    # Most confused pairs
    most_confused = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
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
        'test_loss': float(test_loss) if test_loss != -1 else "N/A",
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
            'final_train_loss': float(history.history['loss'][-1]),
            'train_accuracy_history': [float(x) for x in history.history['accuracy']],
            'train_loss_history': [float(x) for x in history.history['loss']]
        }
        
        # Add validation metrics only if they exist (not in LOSO without validation split)
        if 'val_accuracy' in history.history:
            eval_results['training_history'].update({
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'best_val_accuracy': float(max(history.history['val_accuracy'])),
                'best_val_accuracy_epoch': int(np.argmax(history.history['val_accuracy']) + 1),
                'val_accuracy_history': [float(x) for x in history.history['val_accuracy']],
                'val_loss_history': [float(x) for x in history.history['val_loss']]
            })
    
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
    
    n_classes = len(class_names)
    
    # Plot 1: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14)
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # Plot 2: Per-class accuracy
    axes[0, 1].bar(range(n_classes), per_class_acc, color='skyblue')
    axes[0, 1].set_title('Per-Class Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_xticks(range(n_classes))
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: Class distribution
    x = np.arange(n_classes)
    width = 0.35
    true_counts = [y_test_counts.get(i, 0) for i in range(n_classes)]
    pred_counts = [y_pred_counts.get(i, 0) for i in range(n_classes)]
    
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
    metric_values = np.array([[report[class_names[i]][metric] for i in range(n_classes)] 
                             for metric in metrics])
    
    x = np.arange(n_classes)
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
    error_by_class = [np.sum((y_test == i) & ~correct) for i in range(n_classes)]
    axes[1, 2].bar(range(n_classes), error_by_class, color='red', alpha=0.7)
    axes[1, 2].set_title('Errors by Class', fontsize=14)
    axes[1, 2].set_xlabel('Class')
    axes[1, 2].set_ylabel('Number of Errors')
    axes[1, 2].set_xticks(range(n_classes))
    axes[1, 2].set_xticklabels(class_names, rotation=45, ha='right')
    
    # Plots 7 & 8: Training history
    if history is not None:
        epochs_range = range(1, len(history.history['accuracy']) + 1)
        axes[2, 0].plot(epochs_range, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        
        # Plot validation accuracy only if it exists
        if 'val_accuracy' in history.history:
            axes[2, 0].plot(epochs_range, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
            axes[2, 0].set_title('Training & Validation Accuracy', fontsize=14)
        else:
            axes[2, 0].set_title('Training Accuracy', fontsize=14)
            
        axes[2, 0].set_xlabel('Epochs')
        axes[2, 0].set_ylabel('Accuracy')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_ylim(0, 1.05)
        
        axes[2, 1].plot(epochs_range, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        
        # Plot validation loss only if it exists
        if 'val_loss' in history.history:
            axes[2, 1].plot(epochs_range, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            axes[2, 1].set_title('Training & Validation Loss', fontsize=14)
        else:
            axes[2, 1].set_title('Training Loss', fontsize=14)
            
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


def generate_loso_summary_plots(history, all_fold_evaluations, output_dir, file_identifier):
    """
    Generates a concise summary plot for the LOSO run, showing final model
    training history and aggregated cross-validation metrics.
    """
    # Create a figure with 1 row and 3 columns for a compact layout
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle(f'BSL Gesture Recognition - LOSO Final Summary - {file_identifier}', fontsize=18)

    # --- Plot 1: Final Model Training Accuracy ---
    if history and hasattr(history, 'history') and 'accuracy' in history.history:
        epochs_range = range(1, len(history.history['accuracy']) + 1)
        axes[0].plot(epochs_range, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0].set_title('Final Model Training Accuracy', fontsize=14)
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'Training History\nNot Available', 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14, alpha=0.7, transform=axes[0].transAxes)
        axes[0].set_title('Final Model Training Accuracy', fontsize=14)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

    # --- Plot 2: Final Model Training Loss ---
    if history and hasattr(history, 'history') and 'loss' in history.history:
        epochs_range = range(1, len(history.history['loss']) + 1)
        axes[1].plot(epochs_range, history.history['loss'], 'r-', label='Training Loss', linewidth=2)
        axes[1].set_title('Final Model Training Loss', fontsize=14)
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Training History\nNot Available', 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14, alpha=0.7, transform=axes[1].transAxes)
        axes[1].set_title('Final Model Training Loss', fontsize=14)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    # --- Plot 3: Overall Metrics Summary Text Box ---
    axes[2].axis('off')
    axes[2].set_title('LOSO Evaluation Summary', fontsize=14, pad=20)
    
    # Calculate all necessary metrics
    accuracies = [eval_report['test_accuracy'] for eval_report in all_fold_evaluations]
    f1_scores = [eval_report['classification_report']['macro avg']['f1-score'] for eval_report in all_fold_evaluations]
    precisions = [eval_report['classification_report']['macro avg']['precision'] for eval_report in all_fold_evaluations]
    recalls = [eval_report['classification_report']['macro avg']['recall'] for eval_report in all_fold_evaluations]
    
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    avg_precision = np.mean(precisions)
    std_precision = np.std(precisions)
    avg_recall = np.mean(recalls)
    std_recall = np.std(recalls)

    metrics_text = (
        f"    LOSO Cross-Validation Summary\n"
        f"----------------------------------------\n"
        f" Avg. Accuracy:  {avg_accuracy:.4f} (± {std_accuracy:.4f})\n"
        f" Avg. F1-Score:    {avg_f1:.4f} (± {std_f1:.4f})\n"
        f" Avg. Precision: {avg_precision:.4f} (± {std_precision:.4f})\n"
        f" Avg. Recall:      {avg_recall:.4f} (± {std_recall:.4f})"
    )

    axes[2].text(0.5, 0.5, metrics_text, 
                    horizontalalignment='center', verticalalignment='center', fontsize=13,
                    fontfamily='monospace', transform=axes[2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", fc='aliceblue', ec='black', lw=1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(output_dir, f'loso_summary_plots_{file_identifier}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"LOSO summary plot saved to: {plot_path}") 