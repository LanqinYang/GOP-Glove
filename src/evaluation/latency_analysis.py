"""
Latency analysis and three-platform comparison for BSL gesture recognition models.
This script aggregates latency benchmark results across Arduino, CPU, and A100 platforms.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


class LatencyAnalyzer:
    """Class for comprehensive latency analysis across platforms."""
    
    def __init__(self, test_dir="src/test/latency_benchmark_test", output_dir="outputs/latency_analysis"):
        self.test_dir = test_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Platform mapping
        self.platform_mapping = {
            'CPU': 'Colab CPU',
            'A100': 'Colab A100',
            'arduino': 'Arduino'  # Files with arduino in name
        }
        
        # Model display names
        self.model_display_names = {
            '1D_CNN': '1D-CNN',
            'Transformer_Encoder': 'Transformer',
            'LightGBM': 'LightGBM',
            'XGBoost': 'XGBoost',
            'ADANN': 'ADANN',
            'ADANN_LightGBM': 'DA-LGBM'
        }
        
        self.results = {}
    
    def load_latency_data(self):
        """Load latency benchmark results from all platforms."""
        print("Loading latency benchmark data...")
        
        # Find all CSV files in the test directory
        csv_files = glob.glob(os.path.join(self.test_dir, "**/*.csv"), recursive=True)
        
        platform_data = {}
        
        for csv_file in csv_files:
            try:
                # Determine platform from path and filename
                path_parts = Path(csv_file).parts
                filename = Path(csv_file).stem
                platform = None
                
                # First check filename for arduino (highest priority)
                if 'arduino' in filename.lower():
                    platform = 'Arduino'
                # Then check path for specific platforms
                elif 'A100' in path_parts:
                    platform = 'Colab A100'
                elif 'CPU' in path_parts:
                    platform = 'Colab CPU'
                # Final fallback checks
                elif 'a100' in filename.lower():
                    platform = 'Colab A100'
                else:
                    platform = 'Unknown'
                
                # Load data
                df = pd.read_csv(csv_file)
                
                # Handle unit conversion - Arduino data might be in microseconds but labeled as ms
                if platform == 'Arduino':
                    # Check if latency values suggest microseconds (much larger than expected)
                    if 'mean_latency_ms' in df.columns:
                        sample_latency = df['mean_latency_ms'].iloc[0] if len(df) > 0 else 0
                        # If Transformer latency > 10ms on Arduino, likely microseconds
                        transformer_data = df[df['model_name'] == 'Transformer_Encoder']
                        if not transformer_data.empty and transformer_data['mean_latency_ms'].iloc[0] > 10:
                            print(f"   Converting Arduino latency from μs to ms: {csv_file}")
                            df['mean_latency_ms'] = df['mean_latency_ms'] / 1000
                            df['std_latency_ms'] = df['std_latency_ms'] / 1000
                            # Also adjust throughput accordingly
                            if 'throughput_inf_per_sec' in df.columns:
                                df['throughput_inf_per_sec'] = df['throughput_inf_per_sec'] * 1000
                
                # Add platform information
                df['platform'] = platform
                df['source_file'] = csv_file
                
                # Store by platform
                if platform not in platform_data:
                    platform_data[platform] = []
                platform_data[platform].append(df)
                
                print(f"   Loaded {len(df)} records from {platform}: {Path(csv_file).name}")
                
            except Exception as e:
                print(f"   Warning: Could not load {csv_file}: {e}")
        
        # Combine data for each platform
        combined_data = {}
        for platform, dfs in platform_data.items():
            if dfs:
                combined_data[platform] = pd.concat(dfs, ignore_index=True)
        
        self.results = combined_data
        return combined_data
    
    def create_main_latency_table(self):
        """Create the main latency comparison table for dissertation."""
        print("Creating main latency comparison table...")
        
        # Prepare data for the main table
        table_data = []
        
        # For each model, get single-batch latency across platforms
        all_models = set()
        for platform, df in self.results.items():
            all_models.update(df['model_name'].unique())
        
        for model in sorted(all_models):
            row = {'Model': self.model_display_names.get(model, model)}
            
            # Get single-batch (batch_size=1) latency for each platform
            for platform, df in self.results.items():
                # Filter for this model and batch size 1
                model_data = df[(df['model_name'] == model) & (df['batch_size'] == 1)]
                
                if not model_data.empty:
                    # Use LOSO data preferentially, otherwise standard
                    loso_data = model_data[model_data['training_mode'] == 'loso']
                    if not loso_data.empty:
                        latency = loso_data['mean_latency_ms'].iloc[0]
                        std_latency = loso_data['std_latency_ms'].iloc[0]
                    else:
                        standard_data = model_data[model_data['training_mode'] == 'standard']
                        if not standard_data.empty:
                            latency = standard_data['mean_latency_ms'].iloc[0]
                            std_latency = standard_data['std_latency_ms'].iloc[0]
                        else:
                            latency = model_data['mean_latency_ms'].iloc[0]
                            std_latency = model_data['std_latency_ms'].iloc[0]
                    
                    # Format latency with appropriate precision
                    if latency < 1:
                        row[f'{platform}_Latency_ms'] = f"{latency:.3f}"
                        row[f'{platform}_Std_ms'] = f"{std_latency:.3f}"
                    else:
                        row[f'{platform}_Latency_ms'] = f"{latency:.2f}"
                        row[f'{platform}_Std_ms'] = f"{std_latency:.2f}"
                    
                    row[f'{platform}_Latency_raw'] = latency  # For sorting
                else:
                    row[f'{platform}_Latency_ms'] = 'N/A'
                    row[f'{platform}_Std_ms'] = 'N/A'
                    row[f'{platform}_Latency_raw'] = float('inf')
            
            table_data.append(row)
        
        # Create DataFrame
        main_table = pd.DataFrame(table_data)
        
        # Sort by Arduino latency (most relevant for deployment)
        arduino_col = None
        for col in main_table.columns:
            if 'Arduino' in col and '_Latency_raw' in col:
                arduino_col = col
                break
        
        if arduino_col:
            main_table = main_table.sort_values(arduino_col)
        
        # Remove raw latency columns (used only for sorting)
        display_columns = [col for col in main_table.columns if not col.endswith('_raw')]
        main_table_display = main_table[display_columns]
        
        return main_table_display, main_table
    
    def create_comprehensive_latency_plots(self):
        """Create comprehensive latency analysis plots."""
        print("Creating comprehensive latency plots...")
        
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: Platform Comparison (Single Batch)
        self._plot_platform_comparison(plots_dir)
        
        # Plot 2: Batch Size Scaling
        self._plot_batch_scaling(plots_dir)
        
        # Plot 3: Latency vs Accuracy Trade-off
        self._plot_latency_accuracy_tradeoff(plots_dir)
        
        # Plot 4: Model Size vs Latency
        self._plot_model_size_analysis(plots_dir)
        
        print(f"   Plots saved to: {plots_dir}")
    
    def _plot_platform_comparison(self, plots_dir):
        """Create platform comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Latency Comparison Across Platforms (Single Batch)', fontsize=16, fontweight='bold')
        
        # Prepare data for plotting
        plot_data = []
        
        for platform, df in self.results.items():
            # Filter for batch size 1 and LOSO mode (primary comparison)
            single_batch = df[(df['batch_size'] == 1) & (df['training_mode'] == 'loso')]
            if single_batch.empty:
                # Fallback to standard mode
                single_batch = df[(df['batch_size'] == 1) & (df['training_mode'] == 'standard')]
            
            for _, row in single_batch.iterrows():
                plot_data.append({
                    'Model': self.model_display_names.get(row['model_name'], row['model_name']),
                    'Platform': platform,
                    'Latency_ms': row['mean_latency_ms'],
                    'Std_ms': row['std_latency_ms']
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        if not plot_df.empty:
            # Plot 1: Grouped bar chart
            models = plot_df['Model'].unique()
            platforms = plot_df['Platform'].unique()
            
            x = np.arange(len(models))
            width = 0.25
            
            colors = ['skyblue', 'lightcoral', 'lightgreen']
            
            for i, platform in enumerate(platforms):
                platform_data = plot_df[plot_df['Platform'] == platform]
                latencies = []
                errors = []
                
                for model in models:
                    model_data = platform_data[platform_data['Model'] == model]
                    if not model_data.empty:
                        latencies.append(model_data['Latency_ms'].iloc[0])
                        errors.append(model_data['Std_ms'].iloc[0])
                    else:
                        latencies.append(0)
                        errors.append(0)
                
                bars = ax1.bar(x + i*width, latencies, width, 
                              label=platform, alpha=0.8, 
                              color=colors[i % len(colors)],
                              yerr=errors, capsize=3)
                
                # Add value labels on bars
                for bar, lat in zip(bars, latencies):
                    if lat > 0:
                        height = bar.get_height()
                        if height < 1:
                            label = f'{height:.3f}'
                        else:
                            label = f'{height:.1f}'
                        ax1.text(bar.get_x() + bar.get_width()/2., height + bar.get_height()*0.01,
                                label, ha='center', va='bottom', fontsize=8, rotation=90)
            
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Latency (ms)')
            ax1.set_title('Latency by Model and Platform')
            ax1.set_xticks(x + width)
            ax1.set_xticklabels(models, rotation=45, ha='right')
            ax1.legend()
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Throughput comparison
            throughput_data = []
            for platform, df in self.results.items():
                single_batch = df[(df['batch_size'] == 1) & (df['training_mode'] == 'loso')]
                if single_batch.empty:
                    single_batch = df[(df['batch_size'] == 1) & (df['training_mode'] == 'standard')]
                
                for _, row in single_batch.iterrows():
                    if 'throughput_inf_per_sec' in row and not pd.isna(row['throughput_inf_per_sec']):
                        throughput_data.append({
                            'Model': self.model_display_names.get(row['model_name'], row['model_name']),
                            'Platform': platform,
                            'Throughput': row['throughput_inf_per_sec']
                        })
            
            if throughput_data:
                throughput_df = pd.DataFrame(throughput_data)
                
                for i, platform in enumerate(platforms):
                    platform_data = throughput_df[throughput_df['Platform'] == platform]
                    throughputs = []
                    
                    for model in models:
                        model_data = platform_data[platform_data['Model'] == model]
                        if not model_data.empty:
                            throughputs.append(model_data['Throughput'].iloc[0])
                        else:
                            throughputs.append(0)
                    
                    bars = ax2.bar(x + i*width, throughputs, width,
                                  label=platform, alpha=0.8,
                                  color=colors[i % len(colors)])
                    
                    # Add value labels
                    for bar, thr in zip(bars, throughputs):
                        if thr > 0:
                            height = bar.get_height()
                            if height > 1000:
                                label = f'{height/1000:.1f}k'
                            else:
                                label = f'{height:.0f}'
                            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                    label, ha='center', va='bottom', fontsize=8, rotation=90)
                
                ax2.set_xlabel('Model')
                ax2.set_ylabel('Throughput (inferences/sec)')
                ax2.set_title('Throughput by Model and Platform')
                ax2.set_xticks(x + width)
                ax2.set_xticklabels(models, rotation=45, ha='right')
                ax2.legend()
                ax2.set_yscale('log')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(plots_dir, 'platform_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_batch_scaling(self, plots_dir):
        """Create batch size scaling analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Batch Size Scaling Analysis', fontsize=16, fontweight='bold')
        
        # Only analyze platforms that have batch size data
        platforms_with_batching = []
        for platform, df in self.results.items():
            if len(df['batch_size'].unique()) > 1:
                platforms_with_batching.append(platform)
        
        if not platforms_with_batching:
            # Create placeholder if no batch data
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No batch scaling data available',
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Batch Scaling Analysis')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(plots_dir, 'batch_scaling.png'), dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Plot for each platform (up to 4)
        for idx, platform in enumerate(platforms_with_batching[:4]):
            ax = axes[idx // 2, idx % 2]
            
            df = self.results[platform]
            models = df['model_name'].unique()
            
            for model in models:
                model_data = df[df['model_name'] == model]
                # Use LOSO data preferentially
                loso_data = model_data[model_data['training_mode'] == 'loso']
                if not loso_data.empty:
                    plot_data = loso_data
                else:
                    plot_data = model_data[model_data['training_mode'] == 'standard']
                
                if not plot_data.empty:
                    batch_sizes = plot_data['batch_size'].values
                    latencies = plot_data['mean_latency_ms'].values
                    
                    ax.plot(batch_sizes, latencies, 'o-', label=self.model_display_names.get(model, model),
                           linewidth=2, markersize=6)
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Latency (ms)')
            ax.set_title(f'{platform} - Batch Scaling')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        # Hide unused subplots
        for idx in range(len(platforms_with_batching), 4):
            axes[idx // 2, idx % 2].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(plots_dir, 'batch_scaling.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latency_accuracy_tradeoff(self, plots_dir):
        """Create latency vs accuracy trade-off analysis."""
        # This would require accuracy data from evaluation results
        # For now, create a placeholder
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Latency vs Accuracy Trade-off Analysis', fontsize=16, fontweight='bold')
        
        ax.text(0.5, 0.5, 'Accuracy data integration pending\n(Requires evaluation results)',
               ha='center', va='center', transform=ax.transAxes, fontsize=14,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance vs Inference Speed')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'latency_accuracy_tradeoff.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_size_analysis(self, plots_dir):
        """Create model size vs latency analysis."""
        # This would require model size data
        # For now, create a placeholder with estimated sizes
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Model Size vs Latency Analysis', fontsize=16, fontweight='bold')
        
        # Estimated model sizes (in MB) - these would come from actual model files
        estimated_sizes = {
            '1D_CNN': 0.5,
            'Transformer_Encoder': 2.0,
            'LightGBM': 0.1,
            'XGBoost': 0.2,
            'ADANN': 1.5,
            'ADANN_LightGBM': 1.6
        }
        
        # Get Arduino latencies for comparison
        arduino_platform = None
        for platform in self.results.keys():
            if 'Arduino' in platform:
                arduino_platform = platform
                break
        
        if arduino_platform:
            df = self.results[arduino_platform]
            single_batch = df[df['batch_size'] == 1]
            
            sizes = []
            latencies = []
            labels = []
            
            for _, row in single_batch.iterrows():
                model = row['model_name']
                if model in estimated_sizes:
                    sizes.append(estimated_sizes[model])
                    latencies.append(row['mean_latency_ms'])
                    labels.append(self.model_display_names.get(model, model))
            
            if sizes and latencies:
                scatter = ax.scatter(sizes, latencies, s=100, alpha=0.7, c=range(len(sizes)), cmap='viridis')
                
                # Add labels
                for i, label in enumerate(labels):
                    ax.annotate(label, (sizes[i], latencies[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=10)
                
                ax.set_xlabel('Estimated Model Size (MB)')
                ax.set_ylabel('Latency (ms)')
                ax.set_title('Model Size vs Inference Latency (Arduino)')
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'Arduino latency data not available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_size_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_deployment_recommendations(self, main_table):
        """Generate deployment recommendations based on latency analysis."""
        print("Generating deployment recommendations...")
        
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'analysis_summary': {},
            'deployment_recommendations': {},
            'platform_specific_notes': {}
        }
        
        # Analyze Arduino performance (most deployment-relevant)
        arduino_col = None
        for col in main_table.columns:
            if 'Arduino' in col and 'Latency_ms' in col:
                arduino_col = col
                break
        
        if arduino_col:
            # Parse latency values
            arduino_latencies = []
            model_names = []
            
            for _, row in main_table.iterrows():
                if row[arduino_col] != 'N/A':
                    try:
                        latency = float(row[arduino_col])
                        arduino_latencies.append(latency)
                        model_names.append(row['Model'])
                    except ValueError:
                        pass
            
            if arduino_latencies:
                best_idx = np.argmin(arduino_latencies)
                worst_idx = np.argmax(arduino_latencies)
                
                recommendations['analysis_summary'] = {
                    'fastest_model': model_names[best_idx],
                    'fastest_latency_ms': arduino_latencies[best_idx],
                    'slowest_model': model_names[worst_idx],
                    'slowest_latency_ms': arduino_latencies[worst_idx],
                    'latency_range_ms': [min(arduino_latencies), max(arduino_latencies)],
                    'mean_latency_ms': np.mean(arduino_latencies)
                }
                
                # Deployment recommendations
                recommendations['deployment_recommendations'] = {
                    'real_time_suitable': [name for name, lat in zip(model_names, arduino_latencies) if lat < 5.0],
                    'interactive_suitable': [name for name, lat in zip(model_names, arduino_latencies) if lat < 50.0],
                    'batch_processing_only': [name for name, lat in zip(model_names, arduino_latencies) if lat >= 50.0],
                    'recommended_for_tinyml': model_names[best_idx] if arduino_latencies[best_idx] < 10.0 else 'None'
                }
        
        # Platform-specific notes
        recommendations['platform_specific_notes'] = {
            'Arduino': 'Optimized for low-power edge deployment with INT8 quantization',
            'Colab A100': 'High-throughput inference for batch processing and development',
            'Colab CPU': 'CPU-based inference for development and comparison',
            'General': 'Latency measurements include preprocessing and post-processing overhead'
        }
        
        return recommendations
    
    def save_results(self, main_table, recommendations):
        """Save all analysis results."""
        print("Saving latency analysis results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main table
        table_path = os.path.join(self.output_dir, f'latency_comparison_table_{timestamp}.csv')
        main_table.to_csv(table_path, index=False)
        print(f"   Main table saved: {table_path}")
        
        # Save as markdown for dissertation
        markdown_path = os.path.join(self.output_dir, f'latency_comparison_table_{timestamp}.md')
        with open(markdown_path, 'w') as f:
            f.write("# Three-Platform Latency Comparison Table\n\n")
            f.write("Inference latency (ms) for single-batch processing across deployment platforms:\n\n")
            f.write(main_table.to_markdown(index=False))
            f.write("\n\n")
            f.write("**Note**: Latencies measured with LOSO-trained models where available, ")
            f.write("otherwise standard training mode. All measurements include preprocessing overhead.\n")
        print(f"   Markdown table saved: {markdown_path}")
        
        # Save recommendations
        rec_path = os.path.join(self.output_dir, f'deployment_recommendations_{timestamp}.json')
        with open(rec_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"   Recommendations saved: {rec_path}")
        
        # Save raw aggregated data
        raw_data_path = os.path.join(self.output_dir, f'aggregated_latency_data_{timestamp}.json')
        serializable_results = {}
        for platform, df in self.results.items():
            serializable_results[platform] = df.to_dict('records')
        
        with open(raw_data_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        print(f"   Raw data saved: {raw_data_path}")
        
        return table_path, markdown_path, rec_path


def main():
    """Main function to run latency analysis."""
    print("="*60)
    print("BSL GESTURE RECOGNITION - LATENCY ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = LatencyAnalyzer()
    
    # Load data
    latency_data = analyzer.load_latency_data()
    
    if not latency_data:
        print("No latency data found. Please check the test directory.")
        return
    
    print(f"\nLoaded data from {len(latency_data)} platforms:")
    for platform, df in latency_data.items():
        print(f"   {platform}: {len(df)} measurements")
    
    # Create main table
    main_table_display, main_table_full = analyzer.create_main_latency_table()
    
    # Create plots
    analyzer.create_comprehensive_latency_plots()
    
    # Generate recommendations
    recommendations = analyzer.generate_deployment_recommendations(main_table_full)
    
    # Save results
    table_path, markdown_path, rec_path = analyzer.save_results(main_table_display, recommendations)
    
    # Print summary
    print("\n" + "="*60)
    print("LATENCY ANALYSIS COMPLETE")
    print("="*60)
    
    # Display main table
    print("\nMain Latency Comparison Table:")
    print(main_table_display.to_string(index=False))
    
    # Display key recommendations
    if 'analysis_summary' in recommendations:
        summary = recommendations['analysis_summary']
        print(f"\n📊 Analysis Summary:")
        print(f"   Fastest Model: {summary.get('fastest_model', 'N/A')} ({summary.get('fastest_latency_ms', 'N/A')} ms)")
        print(f"   Slowest Model: {summary.get('slowest_model', 'N/A')} ({summary.get('slowest_latency_ms', 'N/A')} ms)")
        
        deploy_rec = recommendations.get('deployment_recommendations', {})
        real_time = deploy_rec.get('real_time_suitable', [])
        if real_time:
            print(f"   Real-time Suitable: {', '.join(real_time)}")
        
        tinyml_rec = deploy_rec.get('recommended_for_tinyml', 'None')
        print(f"   TinyML Recommended: {tinyml_rec}")
    
    print(f"\n📁 Results saved to: {analyzer.output_dir}")
    print(f"📄 Main table: {table_path}")
    print(f"📝 Markdown table: {markdown_path}")
    print(f"💡 Recommendations: {rec_path}")


if __name__ == "__main__":
    main()
