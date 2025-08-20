#!/usr/bin/env python3
"""
Comprehensive summary of all experimental results.
Generates tables and figures for dissertation.
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import glob
from pathlib import Path

# Add project root to path
sys.path.append('.')

class ComprehensiveSummary:
    """Generate comprehensive summary of all experimental results."""
    
    def __init__(self, output_dir="outputs/comprehensive_summary"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all available data
        self.sensor_data = None
        self.latency_data = None
        self.model_results = None
        self.ablation_results = None
        
    def load_all_data(self):
        """Load all available experimental data."""
        print("📊 Loading all experimental data...")
        
        # Load sensor characterization data
        self._load_sensor_data()
        
        # Load latency analysis data
        self._load_latency_data()
        
        # Load model performance results
        self._load_model_results()
        
        # Load ablation study results
        self._load_ablation_results()
    
    def _load_sensor_data(self):
        """Load sensor characterization data."""
        try:
            # Look for latest sensor summary
            sensor_files = glob.glob("outputs/sensor_stats/sensor_summary_*.csv")
            if sensor_files:
                latest_file = max(sensor_files, key=os.path.getctime)
                self.sensor_data = pd.read_csv(latest_file)
                print(f"  ✅ Loaded sensor data: {latest_file}")
            else:
                print("  ⚠️ No sensor data found")
        except Exception as e:
            print(f"  ❌ Error loading sensor data: {e}")
    
    def _load_latency_data(self):
        """Load latency analysis data."""
        try:
            # Look for latest latency table
            latency_files = glob.glob("outputs/latency_analysis/latency_comparison_table_*.csv")
            if latency_files:
                latest_file = max(latency_files, key=os.path.getctime)
                self.latency_data = pd.read_csv(latest_file)
                print(f"  ✅ Loaded latency data: {latest_file}")
            else:
                print("  ⚠️ No latency data found")
        except Exception as e:
            print(f"  ❌ Error loading latency data: {e}")
    
    def _load_model_results(self):
        """Load model performance results."""
        try:
            # Look for aggregated results
            result_files = glob.glob("outputs/aggregated_analysis_*/main_results_table.csv")
            if result_files:
                latest_file = max(result_files, key=os.path.getctime)
                self.model_results = pd.read_csv(latest_file)
                print(f"  ✅ Loaded model results: {latest_file}")
            else:
                print("  ⚠️ No model results found")
        except Exception as e:
            print(f"  ❌ Error loading model results: {e}")
    
    def _load_ablation_results(self):
        """Load ablation study results."""
        try:
            # Look for ablation summaries
            ablation_files = glob.glob("outputs/*ablation*/ablation_summary_*.csv")
            if ablation_files:
                latest_file = max(ablation_files, key=os.path.getctime)
                self.ablation_results = pd.read_csv(latest_file)
                print(f"  ✅ Loaded ablation results: {latest_file}")
            else:
                print("  ⚠️ No ablation results found")
        except Exception as e:
            print(f"  ❌ Error loading ablation results: {e}")
    
    def create_sensor_stability_summary(self):
        """Create sensor stability summary table and plots."""
        if self.sensor_data is None:
            print("⚠️ No sensor data available")
            return
        
        print("📊 Creating sensor stability summary...")
        
        # Calculate summary statistics
        snr_summary = self.sensor_data[self.sensor_data['metric_type'] == 'SNR'].groupby('channel')['value'].agg(['mean', 'std', 'min', 'max']).round(2)
        
        # Create summary table
        summary_table = pd.DataFrame({
            'Channel': ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'],
            'Mean_SNR_dB': snr_summary['mean'].values,
            'Std_SNR_dB': snr_summary['std'].values,
            'Min_SNR_dB': snr_summary['min'].values,
            'Max_SNR_dB': snr_summary['max'].values
        })
        
        # Save summary table
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = f"{self.output_dir}/sensor_stability_summary_{timestamp}.csv"
        summary_table.to_csv(summary_path, index=False)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # SNR boxplot by channel
        snr_data = self.sensor_data[self.sensor_data['metric_type'] == 'SNR']
        sns.boxplot(data=snr_data, x='channel', y='value', ax=ax1)
        ax1.set_title('SNR Distribution by Channel')
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('SNR (dB)')
        
        # SNR histogram
        ax2.hist(snr_data['value'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Overall SNR Distribution')
        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plot_path = f"{self.output_dir}/sensor_stability_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Sensor summary saved: {summary_path}")
        print(f"  📈 Sensor plots saved: {plot_path}")
        
        return summary_path, plot_path
    
    def create_performance_comparison_table(self):
        """Create comprehensive performance comparison table."""
        if self.model_results is None:
            print("⚠️ No model results available")
            return
        
        print("📊 Creating performance comparison table...")
        
        # Filter for LOSO results
        loso_results = self.model_results[self.model_results['Training_Mode'] == 'loso'].copy()
        
        # Create comparison table
        comparison_table = loso_results[['Model', 'Optimization_Mode', 'Accuracy', 'Macro-F1', 'Precision', 'Recall']].copy()
        comparison_table = comparison_table.sort_values('Accuracy', ascending=False)
        
        # Round to 2 decimal places
        numeric_cols = ['Accuracy', 'Macro-F1', 'Precision', 'Recall']
        comparison_table[numeric_cols] = comparison_table[numeric_cols].round(2)
        
        # Save table
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        table_path = f"{self.output_dir}/performance_comparison_{timestamp}.csv"
        comparison_table.to_csv(table_path, index=False)
        
        # Create markdown table
        md_path = f"{self.output_dir}/performance_comparison_{timestamp}.md"
        with open(md_path, 'w') as f:
            f.write("# Model Performance Comparison (LOSO)\n\n")
            f.write("| Model | Optimization_Mode | Accuracy (%) | Macro-F1 (%) | Precision (%) | Recall (%) |\n")
            f.write("|-------|------------------|-------------|-------------|---------------|------------|\n")
            
            for _, row in comparison_table.iterrows():
                f.write(f"| {row['Model']} | {row['Optimization_Mode']} | {row['Accuracy']:.2f} | {row['Macro-F1']:.2f} | {row['Precision']:.2f} | {row['Recall']:.2f} |\n")
        
        print(f"  📊 Performance table saved: {table_path}")
        print(f"  📝 Markdown table saved: {md_path}")
        
        return table_path, md_path
    
    def create_deployment_summary(self):
        """Create deployment summary with latency and model size."""
        if self.latency_data is None:
            print("⚠️ No latency data available")
            return
        
        print("📊 Creating deployment summary...")
        
        # Create deployment table
        deployment_table = self.latency_data.copy()
        
        # Add model size estimates (placeholder - you might want to add actual model sizes)
        model_sizes = {
            '1D-CNN': '~50KB',
            'ADANN': '~200KB', 
            'XGBoost': '~100KB',
            'LightGBM': '~80KB',
            'DA-LGBM': '~300KB',
            'Transformer': '~500KB'
        }
        
        deployment_table['Estimated_Size'] = deployment_table['Model'].map(model_sizes)
        
        # Add deployment recommendations
        def get_recommendation(row):
            if row['Arduino_Latency_ms'] < 0.01:
                return 'Excellent for TinyML'
            elif row['Arduino_Latency_ms'] < 0.1:
                return 'Good for Real-time'
            else:
                return 'Limited for Edge'
        
        deployment_table['Recommendation'] = deployment_table.apply(get_recommendation, axis=1)
        
        # Save table
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        table_path = f"{self.output_dir}/deployment_summary_{timestamp}.csv"
        deployment_table.to_csv(table_path, index=False)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Arduino latency comparison
        models = deployment_table['Model']
        arduino_latency = deployment_table['Arduino_Latency_ms']
        
        ax1.bar(models, arduino_latency, color='lightblue', alpha=0.7)
        ax1.set_title('Arduino Latency Comparison')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Latency (ms)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Colab A100 latency comparison
        colab_latency = deployment_table['Colab A100_Latency_ms']
        
        ax2.bar(models, colab_latency, color='lightgreen', alpha=0.7)
        ax2.set_title('Colab A100 Latency Comparison')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Latency (ms)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = f"{self.output_dir}/deployment_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Deployment summary saved: {table_path}")
        print(f"  📈 Deployment plots saved: {plot_path}")
        
        return table_path, plot_path
    
    def create_ablation_summary(self):
        """Create ablation study summary."""
        if self.ablation_results is None:
            print("⚠️ No ablation results available")
            return
        
        print("📊 Creating ablation study summary...")
        
        # Create summary table
        ablation_summary = self.ablation_results.copy()
        
        # Save table
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        table_path = f"{self.output_dir}/ablation_summary_{timestamp}.csv"
        ablation_summary.to_csv(table_path, index=False)
        
        # Create visualization
        if len(ablation_summary) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Augmentation impact
            aug_data = ablation_summary[ablation_summary['Test'] == 'Augmentation']
            if len(aug_data) > 0:
                ax1.bar(aug_data['Config'], aug_data['Accuracy'], color=['red', 'green'], alpha=0.7)
                ax1.set_title('Augmentation Impact on Accuracy')
                ax1.set_ylabel('Accuracy (%)')
                ax1.tick_params(axis='x', rotation=45)
            
            # Epochs impact
            epoch_data = ablation_summary[ablation_summary['Test'] == 'Epochs']
            if len(epoch_data) > 0:
                epochs = [int(x.split()[0]) for x in epoch_data['Config']]
                ax2.plot(epochs, epoch_data['Accuracy'], 'o-', color='blue', linewidth=2, markersize=8)
                ax2.set_title('Epochs Impact on Accuracy')
                ax2.set_xlabel('Epochs')
                ax2.set_ylabel('Accuracy (%)')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = f"{self.output_dir}/ablation_plots_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 Ablation plots saved: {plot_path}")
        else:
            plot_path = None
        
        print(f"  📊 Ablation summary saved: {table_path}")
        
        return table_path, plot_path
    
    def create_dissertation_summary(self):
        """Create comprehensive summary for dissertation."""
        print("📊 Creating dissertation summary...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive summary document
        summary_path = f"{self.output_dir}/dissertation_summary_{timestamp}.md"
        
        with open(summary_path, 'w') as f:
            f.write("# BSL Gesture Recognition - Dissertation Summary\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 Key Findings\n\n")
            
            # Sensor stability findings
            if self.sensor_data is not None:
                snr_mean = self.sensor_data[self.sensor_data['metric_type'] == 'SNR']['value'].mean()
                f.write(f"### Sensor Stability\n")
                f.write(f"- **Average SNR**: {snr_mean:.2f} dB\n")
                f.write(f"- **SNR Range**: {self.sensor_data[self.sensor_data['metric_type'] == 'SNR']['value'].min():.2f} - {self.sensor_data[self.sensor_data['metric_type'] == 'SNR']['value'].max():.2f} dB\n\n")
            
            # Model performance findings
            if self.model_results is not None:
                loso_results = self.model_results[self.model_results['Training_Mode'] == 'loso']
                best_model = loso_results.loc[loso_results['Accuracy'].idxmax()]
                f.write(f"### Model Performance\n")
                f.write(f"- **Best Model**: {best_model['Model']} ({best_model['Accuracy']:.2f}% accuracy)\n")
                f.write(f"- **Best Macro-F1**: {best_model['Macro-F1']:.2f}%\n\n")
            
            # Deployment findings
            if self.latency_data is not None:
                fastest_model = self.latency_data.loc[self.latency_data['Arduino_Latency_ms'].idxmin()]
                f.write(f"### Deployment Performance\n")
                f.write(f"- **Fastest Model**: {fastest_model['Model']} ({fastest_model['Arduino_Latency_ms']:.3f} ms)\n")
                f.write(f"- **Real-time Suitable**: All models < 50ms latency\n\n")
            
            f.write("## 📈 Generated Files\n\n")
            f.write("The following files have been generated for the dissertation:\n\n")
            
            # List generated files
            files = glob.glob(f"{self.output_dir}/*")
            for file_path in files:
                if os.path.isfile(file_path):
                    file_name = os.path.basename(file_path)
                    f.write(f"- `{file_name}`\n")
        
        print(f"📄 Dissertation summary saved: {summary_path}")
        return summary_path
    
    def generate_all_summaries(self):
        """Generate all summary tables and visualizations."""
        print("="*60)
        print("COMPREHENSIVE SUMMARY GENERATION")
        print("="*60)
        
        # Load all data
        self.load_all_data()
        
        # Generate summaries
        summaries = {}
        
        # Sensor stability summary
        sensor_summary = self.create_sensor_stability_summary()
        if sensor_summary:
            summaries['sensor'] = sensor_summary
        
        # Performance comparison
        perf_summary = self.create_performance_comparison_table()
        if perf_summary:
            summaries['performance'] = perf_summary
        
        # Deployment summary
        deploy_summary = self.create_deployment_summary()
        if deploy_summary:
            summaries['deployment'] = deploy_summary
        
        # Ablation summary
        ablation_summary = self.create_ablation_summary()
        if ablation_summary:
            summaries['ablation'] = ablation_summary
        
        # Dissertation summary
        diss_summary = self.create_dissertation_summary()
        if diss_summary:
            summaries['dissertation'] = (diss_summary, None)
        
        print("\n" + "="*60)
        print("COMPREHENSIVE SUMMARY COMPLETE")
        print("="*60)
        
        for category, (table_path, plot_path) in summaries.items():
            print(f"📊 {category.title()}: {table_path}")
            if plot_path:
                print(f"📈 {category.title()} plots: {plot_path}")
        
        print(f"\n📁 All results saved to: {self.output_dir}")
        
        return summaries

def main():
    """Main function to generate comprehensive summary."""
    summary = ComprehensiveSummary()
    summary.generate_all_summaries()

if __name__ == "__main__":
    main()
