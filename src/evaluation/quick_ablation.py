#!/usr/bin/env python3
"""
Quick ablation study for ADANN_LightGBM model.
Tests only the most critical parameters: augmentation switch and epochs.
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import subprocess
import glob
from pathlib import Path

# Add project root to path
sys.path.append('.')

class QuickAblationStudy:
    """Quick ablation study for ADANN_LightGBM model."""
    
    def __init__(self, output_dir="outputs/quick_ablation"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = {
            'augmentation_ablation': [],
            'epochs_ablation': [],
            'summary': {}
        }
    
    def run_augmentation_ablation(self):
        """Test the impact of data augmentation (simplified)."""
        print("🔬 Running quick augmentation ablation study...")
        
        # Test only two configurations: with and without augmentation
        configs = [
            {'enabled': False, 'name': 'No Augmentation'},
            {'enabled': True, 'name': 'With Augmentation'}
        ]
        
        for config in configs:
            print(f"  Testing: {config['name']}")
            
            # Create temporary config
            self._create_temp_config(config)
            
            # Run training with reduced parameters for speed
            try:
                result = self._run_quick_training_experiment(config['name'])
                self.results['augmentation_ablation'].append({
                    'config': config,
                    'result': result
                })
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                self.results['augmentation_ablation'].append({
                    'config': config,
                    'result': {'error': str(e)}
                })
            
            # Clean up
            self._cleanup_temp_config()
    
    def run_epochs_ablation(self):
        """Test the impact of training epochs (simplified)."""
        print("🔬 Running quick epochs ablation study...")
        
        # Test only three epoch values for speed
        epochs_list = [50, 100, 150]
        
        for epochs in epochs_list:
            print(f"  Testing: {epochs} epochs")
            
            try:
                result = self._run_quick_training_experiment(f"{epochs}_epochs", epochs=epochs)
                self.results['epochs_ablation'].append({
                    'epochs': epochs,
                    'result': result
                })
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                self.results['epochs_ablation'].append({
                    'epochs': epochs,
                    'result': {'error': str(e)}
                })
    
    def _create_temp_config(self, config):
        """Create temporary config file for ablation test."""
        import yaml
        
        # Load original config
        with open('configs/config.yaml', 'r') as f:
            full_config = yaml.safe_load(f)
        
        # Modify augmentation settings
        full_config['data']['augmentation']['enabled'] = config['enabled']
        
        # Save temporary config
        with open('configs/temp_config.yaml', 'w') as f:
            yaml.dump(full_config, f)
    
    def _cleanup_temp_config(self):
        """Clean up temporary config file."""
        if os.path.exists('configs/temp_config.yaml'):
            os.remove('configs/temp_config.yaml')
    
    def _run_quick_training_experiment(self, name, epochs=100):
        """Run a quick training experiment with reduced parameters."""
        # Use subprocess to run training with reduced trials for speed
        cmd = [
            'python', 'run.py',
            '--model_type', 'ADANN_LightGBM',
            '--loso',
            '--epochs', str(epochs),
            '--n_trials', '25',  # Reduced for speed
            '--output_suffix', f'quick_ablation_{name}'
        ]
        
        # Run the command with shorter timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)  # 15 min timeout
        
        if result.returncode != 0:
            raise Exception(f"Training failed: {result.stderr}")
        
        # Parse results from output
        return self._parse_training_output(result.stdout)
    
    def _parse_training_output(self, output):
        """Parse training output to extract metrics."""
        lines = output.split('\n')
        
        # Look for accuracy and F1 scores
        accuracy = None
        f1_score = None
        
        for line in lines:
            if 'Average Accuracy' in line:
                try:
                    accuracy = float(line.split(':')[-1].strip().replace('%', ''))
                except:
                    pass
            elif 'Macro F1-Score' in line:
                try:
                    f1_score = float(line.split(':')[-1].strip().replace('%', ''))
                except:
                    pass
        
        return {
            'accuracy': accuracy,
            'f1_score': f1_score,
            'raw_output': output
        }
    
    def create_ablation_plots(self):
        """Create visualization plots for ablation results."""
        print("📊 Creating quick ablation study plots...")
        
        # Augmentation ablation plot
        if self.results['augmentation_ablation']:
            self._plot_augmentation_results()
        
        # Epochs ablation plot
        if self.results['epochs_ablation']:
            self._plot_epochs_results()
    
    def _plot_augmentation_results(self):
        """Plot augmentation ablation results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        configs = []
        accuracies = []
        f1_scores = []
        
        for result in self.results['augmentation_ablation']:
            if 'error' not in result['result']:
                configs.append(result['config']['name'])
                accuracies.append(result['result']['accuracy'])
                f1_scores.append(result['result']['f1_score'])
        
        if accuracies:
            x = range(len(configs))
            ax1.bar(x, accuracies, color=['red', 'green'], alpha=0.7)
            ax1.set_title('Accuracy: Augmentation Impact')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(configs, rotation=45)
            
            ax2.bar(x, f1_scores, color=['red', 'green'], alpha=0.7)
            ax2.set_title('F1-Score: Augmentation Impact')
            ax2.set_ylabel('F1-Score (%)')
            ax2.set_xticks(x)
            ax2.set_xticklabels(configs, rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/augmentation_ablation.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_epochs_results(self):
        """Plot epochs ablation results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        epochs = []
        accuracies = []
        f1_scores = []
        
        for result in self.results['epochs_ablation']:
            if 'error' not in result['result']:
                epochs.append(result['epochs'])
                accuracies.append(result['result']['accuracy'])
                f1_scores.append(result['result']['f1_score'])
        
        if accuracies:
            ax1.plot(epochs, accuracies, 'o-', color='blue', linewidth=2, markersize=8)
            ax1.set_title('Accuracy vs Training Epochs')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Accuracy (%)')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(epochs, f1_scores, 'o-', color='red', linewidth=2, markersize=8)
            ax2.set_title('F1-Score vs Training Epochs')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('F1-Score (%)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/epochs_ablation.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results(self):
        """Save ablation study results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_path = f"{self.output_dir}/quick_ablation_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create summary table
        summary_data = []
        
        # Augmentation summary
        for result in self.results['augmentation_ablation']:
            if 'error' not in result['result']:
                summary_data.append({
                    'Test': 'Augmentation',
                    'Config': result['config']['name'],
                    'Accuracy': result['result']['accuracy'],
                    'F1_Score': result['result']['f1_score']
                })
        
        # Epochs summary
        for result in self.results['epochs_ablation']:
            if 'error' not in result['result']:
                summary_data.append({
                    'Test': 'Epochs',
                    'Config': f"{result['epochs']} epochs",
                    'Accuracy': result['result']['accuracy'],
                    'F1_Score': result['result']['f1_score']
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_path = f"{self.output_dir}/quick_ablation_summary_{timestamp}.csv"
            df.to_csv(summary_path, index=False)
            
            print(f"📊 Quick ablation summary saved: {summary_path}")
        
        print(f"📄 Detailed results saved: {results_path}")
        return results_path, summary_path if summary_data else None

def main():
    """Main function to run quick ablation study."""
    print("="*60)
    print("ADANN_LightGBM QUICK ABLATION STUDY")
    print("="*60)
    
    # Initialize ablation study
    ablation = QuickAblationStudy()
    
    # Run ablation experiments
    ablation.run_augmentation_ablation()
    ablation.run_epochs_ablation()
    
    # Create visualizations
    ablation.create_ablation_plots()
    
    # Save results
    results_path, summary_path = ablation.save_results()
    
    print("\n" + "="*60)
    print("QUICK ABLATION STUDY COMPLETE")
    print("="*60)
    print(f"📊 Results saved to: {results_path}")
    if summary_path:
        print(f"📈 Summary saved to: {summary_path}")
    print(f"📁 Plots saved to: {ablation.output_dir}/")

if __name__ == "__main__":
    main()
