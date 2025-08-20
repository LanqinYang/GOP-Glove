"""
Aggregate training artifacts and generate comprehensive evaluation summaries.
This script collects all training outputs, Optuna results, and model artifacts
to create consolidated reports for dissertation analysis.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


def find_output_files(base_dir="outputs"):
    """
    Recursively find all evaluation and optimization artifacts.
    
    Returns:
        dict: Organized file paths by category and model type
    """
    artifacts = {
        'evaluation_results': [],
        'best_params': [],
        'trials': [],
        'loso_stats': [],
        'loso_summaries': [],
        'predictions': []
    }
    
    # Walk through all output directories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Categorize files based on naming patterns
            if file.startswith('evaluation_') and file.endswith('.json'):
                artifacts['evaluation_results'].append(file_path)
            elif file.startswith('best_params_') and file.endswith('.json'):
                artifacts['best_params'].append(file_path)
            elif file.startswith('trials_') and file.endswith('.csv'):
                artifacts['trials'].append(file_path)
            elif file.startswith('loso_stats_') and file.endswith('.json'):
                artifacts['loso_stats'].append(file_path)
            elif file.startswith('loso_summary_') and file.endswith('.json'):
                artifacts['loso_summaries'].append(file_path)
            elif file.startswith('predictions_') and file.endswith('.json'):
                artifacts['predictions'].append(file_path)
    
    return artifacts


def load_evaluation_results(file_paths):
    """Load and consolidate evaluation results from multiple files."""
    results = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract metadata from file path
            path_parts = Path(file_path).parts
            if len(path_parts) >= 5:  # outputs/model_type/training_mode/optimization_mode/file
                # Find the "outputs" index and extract from there
                try:
                    outputs_idx = path_parts.index('outputs')
                    if len(path_parts) > outputs_idx + 3:
                        model_type = path_parts[outputs_idx + 1]
                        training_mode = path_parts[outputs_idx + 2]  # loso or standard
                        optimization_mode = path_parts[outputs_idx + 3]  # full or arduino
                    else:
                        raise IndexError("Not enough path components")
                except (ValueError, IndexError):
                    # Fallback to old method
                    model_type = path_parts[-5]
                    training_mode = path_parts[-4]  # loso or standard
                    optimization_mode = path_parts[-3]  # full or arduino
            else:
                # Fallback: extract from filename
                filename = Path(file_path).stem
                parts = filename.split('_')
                model_type = parts[1] if len(parts) > 1 else 'unknown'
                # Try to extract training_mode from filename
                if 'loso' in filename:
                    training_mode = 'loso'
                elif 'standard' in filename:
                    training_mode = 'standard'
                else:
                    training_mode = 'unknown'
                optimization_mode = 'unknown'
            
            # Add metadata to results
            data['model_type'] = model_type
            data['training_mode'] = training_mode
            data['optimization_mode'] = optimization_mode
            data['file_path'] = file_path
            
            results.append(data)
            
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    return results


def load_best_params(file_paths):
    """Load and consolidate best parameters from Optuna optimization."""
    params_data = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            params_data.append(data)
            
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    return params_data


def load_trials_data(file_paths):
    """Load and consolidate Optuna trials data."""
    all_trials = []
    
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            
            # Extract metadata from filename
            filename = Path(file_path).stem
            parts = filename.split('_')
            
            # Add metadata columns
            df['model_type'] = parts[1] if len(parts) > 1 else 'unknown'
            df['training_mode'] = parts[2] if len(parts) > 2 else 'unknown'
            df['optimization_mode'] = parts[3] if len(parts) > 3 else 'unknown'
            
            # Handle fold information if present
            fold_info = [p for p in parts if p.startswith('fold')]
            df['fold_id'] = fold_info[0].replace('fold', '') if fold_info else None
            
            df['source_file'] = file_path
            all_trials.append(df)
            
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    return pd.concat(all_trials, ignore_index=True) if all_trials else pd.DataFrame()


def generate_main_results_table(evaluation_results):
    """Generate the main results table for dissertation."""
    
    # Keep all results but organize by training_mode and optimization_mode
    if not evaluation_results:
        print("Warning: No evaluation results found for main table generation")
        return pd.DataFrame()
    
    # Group by model type and mode combinations  
    table_data = []
    
    for result in evaluation_results:
        model_type = result.get('model_type', 'Unknown')
        optimization_mode = result.get('optimization_mode', 'unknown')
        
        training_mode = result.get('training_mode', 'unknown')
        
        # Create combined mode identifier
        combined_mode = f"{training_mode}/{optimization_mode}"
        
        row = {
            'Model': model_type,
            'Training_Mode': training_mode,
            'Optimization_Mode': optimization_mode, 
            'Combined_Mode': combined_mode,
            'Accuracy': result.get('test_accuracy', 0.0),
            'Macro-F1': result.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0.0),
            'Precision': result.get('classification_report', {}).get('macro avg', {}).get('precision', 0.0),
            'Recall': result.get('classification_report', {}).get('macro avg', {}).get('recall', 0.0),
            'Timestamp': result.get('timestamp', 'Unknown')
        }
        
        # Add confidence metrics if available
        conf_stats = result.get('confidence_stats', {})
        row['Avg_Confidence'] = conf_stats.get('mean', 0.0)
        row['Confidence_Std'] = conf_stats.get('std', 0.0)
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Sort by accuracy (descending)
    if not df.empty:
        df = df.sort_values('Accuracy', ascending=False)
    
    return df


def generate_hyperparameter_analysis(best_params_data, trials_df):
    """Analyze hyperparameter optimization results."""
    
    analysis = {
        'convergence_analysis': {},
        'parameter_importance': {},
        'best_configurations': {}
    }
    
    # Check if trials_df is empty
    if trials_df.empty:
        print("   No trials data available for hyperparameter analysis")
        # Try to extract some basic hyperparameter info from model configs
        analysis['note'] = "Hyperparameter optimization data not available. Using model configuration defaults."
        analysis['recommendation'] = "Re-run training with new pipeline to generate Optuna optimization data."
        return analysis
    
    # Group by model type
    for model_type in trials_df['model_type'].unique():
        model_trials = trials_df[trials_df['model_type'] == model_type]
        
        # Convergence analysis
        if not model_trials.empty:
            convergence_data = {
                'total_trials': len(model_trials),
                'completed_trials': len(model_trials[model_trials['state'] == 'COMPLETE']),
                'best_values_progression': model_trials['value'].cummax().tolist(),
                'trial_numbers': model_trials['trial_number'].tolist()
            }
            analysis['convergence_analysis'][model_type] = convergence_data
        
        # Parameter importance (simplified analysis)
        param_cols = [col for col in model_trials.columns if col.startswith('param_')]
        param_importance = {}
        
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            if model_trials[param_col].notna().any():
                # Calculate correlation with objective value
                numeric_vals = pd.to_numeric(model_trials[param_col], errors='coerce')
                if not numeric_vals.isna().all():
                    correlation = numeric_vals.corr(model_trials['value'])
                    param_importance[param_name] = abs(correlation) if not pd.isna(correlation) else 0.0
        
        analysis['parameter_importance'][model_type] = param_importance
    
    # Best configurations
    for params in best_params_data:
        model_type = params.get('model_type', 'Unknown')
        if model_type not in analysis['best_configurations']:
            analysis['best_configurations'][model_type] = []
        
        config = {
            'training_mode': params.get('training_mode'),
            'optimization_mode': params.get('optimization_mode'),
            'best_value': params.get('best_value'),
            'parameters': params.get('best_params', {})
        }
        analysis['best_configurations'][model_type].append(config)
    
    return analysis


def create_comprehensive_plots(evaluation_results, trials_df, output_dir):
    """Create comprehensive visualization plots."""
    
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, 'comprehensive_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Limit to key models for readability
    key_models = ['1D_CNN', 'XGBoost', 'LightGBM', 'ADANN', 'ADANN_LightGBM', 'Transformer_Encoder']
    filtered_results = [r for r in evaluation_results if r.get('model_type') in key_models]
    
    # Plot 1: Model Performance Comparison (simplified)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BSL Gesture Recognition - Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting (use filtered_results)
    loso_results = [r for r in filtered_results if r.get('training_mode') == 'loso']
    
    if loso_results:
        # Get best result per model type
        model_best_results = {}
        for r in loso_results:
            model_type = r.get('model_type', 'Unknown')
            accuracy = r.get('test_accuracy', 0.0)
            if model_type not in model_best_results or accuracy > model_best_results[model_type]['accuracy']:
                model_best_results[model_type] = {
                    'accuracy': accuracy,
                    'f1_score': r.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0.0),
                    'mode': r.get('optimization_mode', 'unknown')
                }
        
        models = list(model_best_results.keys())
        accuracies = [model_best_results[m]['accuracy'] for m in models]
        f1_scores = [model_best_results[m]['f1_score'] for m in models]
        modes = [model_best_results[m]['mode'] for m in models]
        
        # Accuracy comparison
        colors = ['skyblue' if mode == 'full' else 'lightcoral' for mode in modes]
        bars = axes[0, 0].bar(range(len(models)), accuracies, color=colors)
        axes[0, 0].set_title('Best LOSO Accuracy by Model')
        axes[0, 0].set_xlabel('Model Type')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].set_ylim(0, 1.05)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # F1-Score comparison
        bars = axes[0, 1].bar(range(len(models)), f1_scores, color=colors)
        axes[0, 1].set_title('Best LOSO Macro F1-Score by Model')
        axes[0, 1].set_xlabel('Model Type')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].set_ylim(0, 1.05)
        
        # Add value labels on bars
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Training Mode Comparison (Standard vs LOSO)
    if loso_results:
        # Compare standard vs loso for key models
        standard_results = [r for r in filtered_results if r.get('training_mode') == 'standard']
        
        if standard_results:
            # Get best results for comparison
            comparison_data = {}
            for result_set, mode_name in [(loso_results, 'LOSO'), (standard_results, 'Standard')]:
                for r in result_set:
                    model_type = r.get('model_type', 'Unknown')
                    if model_type not in comparison_data:
                        comparison_data[model_type] = {}
                    
                    accuracy = r.get('test_accuracy', 0.0)
                    if mode_name not in comparison_data[model_type] or accuracy > comparison_data[model_type].get(mode_name, {}).get('accuracy', 0):
                        comparison_data[model_type][mode_name] = {
                            'accuracy': accuracy,
                            'f1_score': r.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0.0)
                        }
            
            # Plot comparison for models that have both modes
            models_with_both = [m for m in comparison_data.keys() if 'LOSO' in comparison_data[m] and 'Standard' in comparison_data[m]]
            
            if models_with_both:
                x = np.arange(len(models_with_both))
                width = 0.35
                
                loso_accs = [comparison_data[m]['LOSO']['accuracy'] for m in models_with_both]
                standard_accs = [comparison_data[m]['Standard']['accuracy'] for m in models_with_both]
                
                axes[1, 0].bar(x - width/2, loso_accs, width, label='LOSO', alpha=0.8, color='skyblue')
                axes[1, 0].bar(x + width/2, standard_accs, width, label='Standard', alpha=0.8, color='lightgreen')
                
                axes[1, 0].set_title('Training Mode Comparison (Accuracy)')
                axes[1, 0].set_xlabel('Model Type')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels(models_with_both, rotation=45, ha='right')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_ylim(0, 1.05)
        
        # Plot 3: Optimization Mode Comparison (Full vs Arduino)
        full_results = [r for r in loso_results if r.get('optimization_mode') == 'full']
        arduino_results = [r for r in loso_results if r.get('optimization_mode') == 'arduino']
        
        if full_results and arduino_results:
            opt_comparison_data = {}
            for result_set, opt_mode in [(full_results, 'Full'), (arduino_results, 'Arduino')]:
                for r in result_set:
                    model_type = r.get('model_type', 'Unknown')
                    if model_type not in opt_comparison_data:
                        opt_comparison_data[model_type] = {}
                    
                    accuracy = r.get('test_accuracy', 0.0)
                    if opt_mode not in opt_comparison_data[model_type] or accuracy > opt_comparison_data[model_type].get(opt_mode, {}).get('accuracy', 0):
                        opt_comparison_data[model_type][opt_mode] = {
                            'accuracy': accuracy,
                            'f1_score': r.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0.0)
                        }
            
            models_with_both_opt = [m for m in opt_comparison_data.keys() if 'Full' in opt_comparison_data[m] and 'Arduino' in opt_comparison_data[m]]
            
            if models_with_both_opt:
                x = np.arange(len(models_with_both_opt))
                
                full_accs = [opt_comparison_data[m]['Full']['accuracy'] for m in models_with_both_opt]
                arduino_accs = [opt_comparison_data[m]['Arduino']['accuracy'] for m in models_with_both_opt]
                
                axes[1, 1].bar(x - width/2, full_accs, width, label='Full', alpha=0.8, color='lightcoral')
                axes[1, 1].bar(x + width/2, arduino_accs, width, label='Arduino', alpha=0.8, color='orange')
                
                axes[1, 1].set_title('Optimization Mode Comparison (Accuracy)')
                axes[1, 1].set_xlabel('Model Type')
                axes[1, 1].set_ylabel('Accuracy')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(models_with_both_opt, rotation=45, ha='right')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_ylim(0, 1.05)
    
    # Hide unused plots
    for i in range(2):
        for j in range(2):
            if not axes[i, j].has_data():
                axes[i, j].text(0.5, 0.5, 'No data available', ha='center', va='center', 
                               transform=axes[i, j].transAxes, fontsize=12)
                axes[i, j].set_title(f'Plot {i+1}.{j+1}', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(plots_dir, 'model_performance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive plots saved to: {plots_dir}")


def generate_final_hyperparams_table(best_params_data, output_dir):
    """Generate final hyperparameters table for dissertation."""
    
    # Create table for each model type showing best configuration
    final_params = {}
    
    for params in best_params_data:
        model_type = params.get('model_type', 'Unknown')
        training_mode = params.get('training_mode', 'unknown')
        optimization_mode = params.get('optimization_mode', 'unknown')
        
        # Use LOSO full mode as the primary configuration
        if training_mode == 'loso' and optimization_mode == 'full':
            if model_type not in final_params:
                final_params[model_type] = params
            elif params.get('best_value', 0) > final_params[model_type].get('best_value', 0):
                final_params[model_type] = params
    
    # Create markdown table
    table_rows = []
    table_rows.append("| Model | Best Validation Accuracy | Key Hyperparameters |")
    table_rows.append("|-------|--------------------------|---------------------|")
    
    for model_type, params in final_params.items():
        best_value = params.get('best_value', 0.0)
        best_params = params.get('best_params', {})
        
        # Format key parameters (limit to most important ones)
        key_params = []
        important_params = ['learning_rate', 'batch_size', 'n_estimators', 'max_depth', 
                           'num_leaves', 'dropout', 'filters', 'heads']
        
        for param in important_params:
            if param in best_params:
                value = best_params[param]
                if isinstance(value, float):
                    key_params.append(f"{param}={value:.4f}")
                else:
                    key_params.append(f"{param}={value}")
        
        if len(key_params) > 3:  # Limit to top 3 for readability
            key_params = key_params[:3] + ['...']
        
        params_str = ', '.join(key_params) if key_params else 'Default'
        
        table_rows.append(f"| {model_type} | {best_value:.4f} | {params_str} |")
    
    # Save table
    table_path = os.path.join(output_dir, 'final_hyperparams_table.md')
    with open(table_path, 'w') as f:
        f.write("# Final Hyperparameters Table\n\n")
        f.write("Best hyperparameters found through Optuna optimization (LOSO full mode):\n\n")
        f.write('\n'.join(table_rows))
        f.write("\n\n")
        f.write("Note: These hyperparameters were determined through HPO and are fixed for final model training and deployment.\n")
    
    print(f"Final hyperparameters table saved to: {table_path}")
    
    return final_params


def main():
    """Main aggregation function."""
    print("="*60)
    print("BSL GESTURE RECOGNITION - ARTIFACTS AGGREGATION")
    print("="*60)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/aggregated_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Find all artifacts
    print("\n1. Discovering artifacts...")
    artifacts = find_output_files()
    
    for category, files in artifacts.items():
        print(f"   {category}: {len(files)} files")
    
    # 2. Load evaluation results
    print("\n2. Loading evaluation results...")
    evaluation_results = load_evaluation_results(artifacts['evaluation_results'])
    print(f"   Loaded {len(evaluation_results)} evaluation reports")
    
    # 3. Load optimization data
    print("\n3. Loading optimization data...")
    best_params_data = load_best_params(artifacts['best_params'])
    trials_df = load_trials_data(artifacts['trials'])
    print(f"   Loaded {len(best_params_data)} best parameter sets")
    print(f"   Loaded {len(trials_df)} trial records")
    
    # 4. Generate main results table
    print("\n4. Generating main results table...")
    main_table = generate_main_results_table(evaluation_results)
    if not main_table.empty:
        main_table_path = os.path.join(output_dir, 'main_results_table.csv')
        main_table.to_csv(main_table_path, index=False)
        print(f"   Main results table saved: {main_table_path}")
        
        # Also save as markdown for dissertation
        markdown_path = os.path.join(output_dir, 'main_results_table.md')
        with open(markdown_path, 'w') as f:
            f.write("# Main Results Table (LOSO Cross-Validation)\n\n")
            f.write(main_table.to_markdown(index=False, floatfmt='.4f'))
        print(f"   Markdown table saved: {markdown_path}")
    
    # 5. Hyperparameter analysis
    print("\n5. Analyzing hyperparameter optimization...")
    hp_analysis = generate_hyperparameter_analysis(best_params_data, trials_df)
    
    # Save analysis
    hp_analysis_path = os.path.join(output_dir, 'hyperparameter_analysis.json')
    with open(hp_analysis_path, 'w') as f:
        json.dump(hp_analysis, f, indent=2, default=str)
    print(f"   Hyperparameter analysis saved: {hp_analysis_path}")
    
    # 6. Generate final hyperparameters table
    print("\n6. Generating final hyperparameters table...")
    final_params = generate_final_hyperparams_table(best_params_data, output_dir)
    
    # 7. Create comprehensive plots
    print("\n7. Creating comprehensive visualizations...")
    create_comprehensive_plots(evaluation_results, trials_df, output_dir)
    
    # 8. Generate summary report
    print("\n8. Generating summary report...")
    summary = {
        'generation_timestamp': timestamp,
        'total_evaluation_reports': len(evaluation_results),
        'total_optimization_studies': len(best_params_data),
        'total_trials': len(trials_df),
        'model_types_evaluated': list(set([r.get('model_type', 'Unknown') for r in evaluation_results])),
        'best_overall_accuracy': max([r.get('test_accuracy', 0.0) for r in evaluation_results]) if evaluation_results else 0.0,
        'artifacts_summary': {category: len(files) for category, files in artifacts.items()}
    }
    
    summary_path = os.path.join(output_dir, 'aggregation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("AGGREGATION COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Aggregated analysis saved to: {output_dir}")
    print(f"📈 Total models evaluated: {len(set([r.get('model_type', 'Unknown') for r in evaluation_results]))}")
    print(f"🏆 Best accuracy achieved: {summary['best_overall_accuracy']:.4f}")
    print(f"📁 Summary report: {summary_path}")
    

if __name__ == "__main__":
    main()
