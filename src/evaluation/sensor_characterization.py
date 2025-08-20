"""
Sensor characterization and stability analysis for BSL gesture recognition.
This script analyzes sensor data quality, stability metrics, and cross-session/cross-user variability.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, ScalarFormatter, LogLocator, LogFormatter, LogFormatterMathtext, NullLocator
import seaborn as sns
import glob
import re
from pathlib import Path
from datetime import datetime
from scipy import signal
from scipy.stats import variation, theilslopes
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set plotting style for publication quality (journal-friendly)
plt.style.use('default')
# Global font settings for figures (Times New Roman, compact sizes for papers)
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
# Use a restrained palette (colorblind-friendly blues/teals)
sns.set_palette(["#4c72b0", "#55a868", "#c44e52"])  # blue, green, red (muted)


class SensorCharacterizer:
    """Class for comprehensive sensor stability and quality analysis."""
    
    def __init__(self, data_dir="datasets/gesture_csv", output_dir="outputs/sensor_stats"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'snr_analysis': {},
            'drift_analysis': {},
            'hysteresis_analysis': {},
            'cross_session_analysis': {},
            'cross_user_analysis': {},
            'stability_metrics': {}
        }
    
    def load_sensor_data(self):
        """Load all sensor data files and organize by user and gesture."""
        print("Loading sensor data...")
        
        data_collection = {}
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        for csv_file in csv_files:
            try:
                # Extract metadata from filename
                filename = Path(csv_file).stem
                user_match = re.search(r'user_(\d+)', filename)
                gesture_match = re.search(r'gesture_(\d+)', filename)
                session_match = re.search(r'session_(\d+)', filename)
                
                if user_match and gesture_match:
                    user_id = int(user_match.group(1))
                    gesture_id = int(gesture_match.group(1))
                    session_id = int(session_match.group(1)) if session_match else 1
                    
                    # Load data
                    with open(csv_file, 'r') as f:
                        lines = f.readlines()
                    
                    sensor_data = []
                    for line in lines:
                        if 'timestamp' in line or line.startswith('#'):
                            continue
                        parts = line.strip().split(',')
                        if len(parts) >= 6:
                            try:
                                values = [float(parts[i]) for i in range(1, 6)]  # Skip timestamp
                                sensor_data.append(values)
                            except ValueError:
                                continue
                    
                    if sensor_data:
                        data_array = np.array(sensor_data)
                        
                        # Store in organized structure
                        if user_id not in data_collection:
                            data_collection[user_id] = {}
                        if session_id not in data_collection[user_id]:
                            data_collection[user_id][session_id] = {}
                        
                        data_collection[user_id][session_id][gesture_id] = {
                            'data': data_array,
                            'file_path': csv_file,
                            'n_samples': len(data_array)
                        }
                        
            except Exception as e:
                print(f"Warning: Could not process {csv_file}: {e}")
        
        print(f"Loaded data for {len(data_collection)} users")
        return data_collection
    
    def calculate_snr(self, signal_data, noise_window=20):
        """Calculate Signal-to-Noise Ratio for sensor channels."""
        snr_values = []
        
        for channel in range(signal_data.shape[1]):
            channel_data = signal_data[:, channel]
            
            # Method 1: Signal vs noise variance approach
            # Remove DC component to focus on signal variation
            channel_data_ac = channel_data - np.mean(channel_data)
            
            # Signal power: variance of the AC component (actual signal variation)
            signal_power = np.var(channel_data_ac)
            
            # Noise estimation: use high-frequency components or residual after smoothing
            # Apply smoothing to get signal trend, noise is the residual
            from scipy.signal import savgol_filter
            try:
                # Use Savitzky-Golay filter to extract smooth signal
                window_length = min(21, len(channel_data) // 4)
                if window_length % 2 == 0:
                    window_length += 1
                if window_length >= 3:
                    smooth_signal = savgol_filter(channel_data, window_length, 2)
                    noise_residual = channel_data - smooth_signal
                    noise_power = np.var(noise_residual)
                else:
                    # Fallback for very short signals
                    noise_power = np.var(channel_data) * 0.1  # Assume 10% is noise
            except:
                # Simple fallback: use difference-based noise estimation
                noise_estimate = np.diff(channel_data)
                noise_power = np.var(noise_estimate) / 2  # Difference amplifies noise by sqrt(2)
            
            # Ensure we have reasonable values
            if signal_power <= 0:
                signal_power = np.var(channel_data)
            if noise_power <= 0:
                noise_power = signal_power * 0.01  # Assume at least 1% noise
            
            # SNR in dB
            snr_db = 10 * np.log10(signal_power / noise_power)
            
            # Ensure reasonable SNR range (0-60 dB)
            snr_db = np.clip(snr_db, 0, 60)
            
            snr_values.append(snr_db)
        
        return np.array(snr_values)
    
    def analyze_drift(self, signal_data, window_size=20):
        """Analyze sensor drift characteristics."""
        drift_metrics = []
        
        for channel in range(signal_data.shape[1]):
            channel_data = signal_data[:, channel]
            
            # Smooth trend using moving average (same length for residual computation)
            moving_avg = np.convolve(channel_data, np.ones(window_size)/window_size, mode='same')
            
            # Robust trend slope using Theil–Sen estimator to reduce outlier influence
            time_steps = np.arange(len(moving_avg))
            try:
                trend_slope, intercept, _, _ = theilslopes(moving_avg, time_steps)
            except Exception:
                # Fallback to OLS if robust fit fails
                trend_slope = np.polyfit(time_steps, moving_avg, 1)[0]
            
            # Detrended residuals and their variance (post-detrending variance)
            residuals = channel_data - moving_avg
            drift_variance = np.var(residuals)
            
            # Maximum drift measured on smoothed trend
            max_drift = np.max(moving_avg) - np.min(moving_avg)
            
            drift_metrics.append({
                'trend_slope': trend_slope,
                'drift_variance': drift_variance,
                'max_drift': max_drift,
                'coefficient_of_variation': variation(channel_data)
            })
        
        return drift_metrics
    
    def analyze_hysteresis(self, data_collection):
        """Analyze hysteresis effects by comparing repeated gestures."""
        print("Analyzing hysteresis effects...")
        
        hysteresis_results = {}
        
        for user_id, user_data in data_collection.items():
            for session_id, session_data in user_data.items():
                # Group gestures by type
                gesture_groups = {}
                for gesture_id, gesture_data in session_data.items():
                    if gesture_id not in gesture_groups:
                        gesture_groups[gesture_id] = []
                    gesture_groups[gesture_id].append(gesture_data['data'])
                
                # Analyze hysteresis for gestures with multiple repetitions
                session_hysteresis = {}
                for gesture_id, gesture_list in gesture_groups.items():
                    if len(gesture_list) >= 2:  # Need at least 2 repetitions
                        # Calculate pairwise differences between repetitions
                        hysteresis_values = []
                        
                        for i in range(len(gesture_list)):
                            for j in range(i+1, len(gesture_list)):
                                data1, data2 = gesture_list[i], gesture_list[j]
                                
                                # Resample to same length for comparison
                                min_len = min(len(data1), len(data2))
                                data1_resampled = signal.resample(data1, min_len)
                                data2_resampled = signal.resample(data2, min_len)
                                
                                # Calculate RMSE between repetitions (hysteresis metric)
                                rmse_per_channel = []
                                for ch in range(data1_resampled.shape[1]):
                                    rmse = np.sqrt(mean_squared_error(data1_resampled[:, ch], 
                                                                     data2_resampled[:, ch]))
                                    rmse_per_channel.append(rmse)
                                
                                hysteresis_values.append(rmse_per_channel)
                        
                        if hysteresis_values:
                            session_hysteresis[gesture_id] = {
                                'mean_hysteresis': np.mean(hysteresis_values, axis=0),
                                'std_hysteresis': np.std(hysteresis_values, axis=0),
                                'n_comparisons': len(hysteresis_values)
                            }
                
                if session_hysteresis:
                    hysteresis_results[f"user_{user_id}_session_{session_id}"] = session_hysteresis
        
        self.results['hysteresis_analysis'] = hysteresis_results
        return hysteresis_results
    
    def analyze_cross_session_variability(self, data_collection):
        """Analyze variability across sessions for same user."""
        print("Analyzing cross-session variability...")
        
        cross_session_results = {}
        
        for user_id, user_data in data_collection.items():
            if len(user_data) < 2:  # Need at least 2 sessions
                continue
            
            user_results = {}
            session_ids = list(user_data.keys())
            
            # Compare each gesture across sessions
            for gesture_id in range(11):  # 0-10 gestures
                gesture_session_data = []
                available_sessions = []
                
                for session_id in session_ids:
                    if gesture_id in user_data[session_id]:
                        gesture_session_data.append(user_data[session_id][gesture_id]['data'])
                        available_sessions.append(session_id)
                
                if len(gesture_session_data) >= 2:
                    # Calculate cross-session consistency metrics
                    consistency_metrics = []
                    
                    for i in range(len(gesture_session_data)):
                        for j in range(i+1, len(gesture_session_data)):
                            data1, data2 = gesture_session_data[i], gesture_session_data[j]
                            
                            # Feature-based comparison (mean, std, range)
                            features1 = self._extract_basic_features(data1)
                            features2 = self._extract_basic_features(data2)
                            
                            # Calculate feature differences
                            feature_diff = np.abs(features1 - features2)
                            consistency_metrics.append(feature_diff)
                    
                    if consistency_metrics:
                        user_results[gesture_id] = {
                            'mean_feature_difference': np.mean(consistency_metrics, axis=0),
                            'std_feature_difference': np.std(consistency_metrics, axis=0),
                            'sessions_available': available_sessions,
                            'n_comparisons': len(consistency_metrics)
                        }
            
            if user_results:
                cross_session_results[user_id] = user_results
        
        self.results['cross_session_analysis'] = cross_session_results
        return cross_session_results
    
    def analyze_cross_user_variability(self, data_collection):
        """Analyze variability across different users.

        CV is computed per (user, gesture, channel) as std(detrended)/abs(mean(original)).
        Then take the median across channels to form a per-user CV for that gesture.
        Across users, compute per-gesture median and 95% bootstrap CI.
        """
        print("Analyzing cross-user variability...")
        
        cross_user_results = {}
        user_ids = list(data_collection.keys())
        
        if len(user_ids) < 2:
            print("Warning: Need at least 2 users for cross-user analysis")
            return {}
        
        def compute_channel_cv(raw_values: np.ndarray) -> float:
            # Detrend via moving average; CV uses residual std divided by absolute mean of raw signal
            if raw_values.size == 0:
                return np.nan
            trend = np.convolve(raw_values, np.ones(20)/20, mode='same')
            residual = raw_values - trend
            denom = np.abs(np.mean(raw_values))
            if denom <= 1e-8:
                return np.nan
            return float(np.std(residual, ddof=1) / denom)
        
        # For each gesture, aggregate per-user CVs (median across channels)
        rng = np.random.default_rng(42)
        for gesture_id in range(11):
            user_cv_values = []
            users_included = []
            for user_id in user_ids:
                session_channel_cvs = []
                for session_id, session_data in data_collection[user_id].items():
                    if gesture_id in session_data:
                        data = session_data[gesture_id]['data']  # shape: [N, 5]
                        # Per-channel CV on raw counts with detrending residual std
                        channel_cvs = [compute_channel_cv(data[:, ch]) for ch in range(data.shape[1])]
                        channel_cvs = [cv for cv in channel_cvs if np.isfinite(cv)]
                        if channel_cvs:
                            session_channel_cvs.append(np.median(channel_cvs))
                if session_channel_cvs:
                    users_included.append(user_id)
                    user_cv_values.append(float(np.median(session_channel_cvs)))
            if len(user_cv_values) >= 2:
                # Bootstrap 95% CI on the median across users
                samples = np.array(user_cv_values, dtype=float)
                n_boot = 2000 if len(samples) < 200 else 1000
                boot_stats = []
                for _ in range(n_boot):
                    resample = rng.choice(samples, size=len(samples), replace=True)
                    boot_stats.append(np.median(resample))
                ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])
                cross_user_results[gesture_id] = {
                    'user_cvs': user_cv_values,
                    'median_cv': float(np.median(samples)),
                    'ci95_low': float(ci_low),
                    'ci95_high': float(ci_high),
                    'users_included': users_included,
                    'n_users': len(users_included)
                }
        
        self.results['cross_user_analysis'] = cross_user_results
        return cross_user_results
    
    def _extract_basic_features(self, data):
        """Extract basic statistical features from sensor data."""
        features = []
        
        for channel in range(data.shape[1]):
            channel_data = data[:, channel]
            
            # Basic statistical features
            features.extend([
                np.mean(channel_data),       # Mean
                np.std(channel_data),        # Standard deviation
                np.min(channel_data),        # Minimum
                np.max(channel_data),        # Maximum
            ])
        
        return np.array(features)
    
    def generate_comprehensive_analysis(self, data_collection):
        """Run all analyses and generate comprehensive results."""
        print("="*60)
        print("SENSOR CHARACTERIZATION ANALYSIS")
        print("="*60)
        
        # Run all analyses
        all_snr_results = []
        all_drift_results = []
        
        print("\n1. Analyzing SNR and drift for all data...")
        for user_id, user_data in data_collection.items():
            for session_id, session_data in user_data.items():
                for gesture_id, gesture_data in session_data.items():
                    data = gesture_data['data']
                    
                    # SNR analysis
                    snr_values = self.calculate_snr(data)
                    all_snr_results.append({
                        'user_id': user_id,
                        'session_id': session_id,
                        'gesture_id': gesture_id,
                        'snr_values': snr_values,
                        'mean_snr': np.mean(snr_values)
                    })
                    
                    # Drift analysis
                    drift_metrics = self.analyze_drift(data)
                    all_drift_results.append({
                        'user_id': user_id,
                        'session_id': session_id,
                        'gesture_id': gesture_id,
                        'drift_metrics': drift_metrics
                    })
        
        self.results['snr_analysis'] = all_snr_results
        self.results['drift_analysis'] = all_drift_results
        
        # Run specialized analyses
        self.analyze_hysteresis(data_collection)
        self.analyze_cross_session_variability(data_collection)
        self.analyze_cross_user_variability(data_collection)
        
        # Calculate overall stability metrics
        self._calculate_stability_metrics()
    
    def _calculate_stability_metrics(self):
        """Calculate overall stability metrics."""
        print("\n2. Calculating overall stability metrics...")
        
        # SNR statistics
        all_snr_values = []
        for result in self.results['snr_analysis']:
            all_snr_values.extend(result['snr_values'])
        
        snr_stats = {
            'mean_snr': np.mean(all_snr_values),
            'std_snr': np.std(all_snr_values),
            'min_snr': np.min(all_snr_values),
            'max_snr': np.max(all_snr_values),
            'snr_percentiles': np.percentile(all_snr_values, [25, 50, 75])
        }
        
        # Drift statistics
        all_drift_slopes = []
        all_drift_variances = []
        all_max_drifts = []
        
        for result in self.results['drift_analysis']:
            for metric in result['drift_metrics']:
                all_drift_slopes.append(metric['trend_slope'])
                all_drift_variances.append(metric['drift_variance'])
                all_max_drifts.append(metric['max_drift'])
        
        drift_stats = {
            'mean_drift_slope': np.mean(all_drift_slopes),
            'std_drift_slope': np.std(all_drift_slopes),
            'mean_drift_variance': np.mean(all_drift_variances),
            'mean_max_drift': np.mean(all_max_drifts)
        }
        
        self.results['stability_metrics'] = {
            'snr_statistics': snr_stats,
            'drift_statistics': drift_stats,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def create_visualization_plots(self):
        """Create comprehensive visualization plots."""
        print("\n3. Creating visualization plots...")
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: SNR Analysis
        self._plot_snr_analysis(plots_dir)
        
        # Plot 2: Drift Analysis
        self._plot_drift_analysis(plots_dir)
        
        # Plot 3: Cross-User Variability
        self._plot_cross_user_variability(plots_dir)
        
        # Plot 4: Cross-Session Consistency
        self._plot_cross_session_consistency(plots_dir)
        
        print(f"   Plots saved to: {plots_dir}")
    
    def _plot_snr_analysis(self, plots_dir):
        """Create SNR analysis plots: (a) by channel, (b) by gesture."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        

        # Prepare SNR data
        snr_data = []
        channels = []
        gestures = []

        for result in self.results['snr_analysis']:
            for ch, snr_val in enumerate(result['snr_values']):
                if np.isfinite(snr_val):
                    snr_data.append(snr_val)
                    channels.append(ch)
                    gestures.append(result['gesture_id'])

        # (a) SNR by channel (boxplot)
        snr_by_channel = [[] for _ in range(5)]
        for snr, ch in zip(snr_data, channels):
            snr_by_channel[ch].append(snr)

        box_a = axes[0].boxplot(
            snr_by_channel,
            labels=[f'Ch{i}' for i in range(5)],
            patch_artist=True,
            medianprops=dict(color='#1f1f1f'),
            boxprops=dict(color='#1f1f1f'),
            whiskerprops=dict(color='#1f1f1f'),
            capprops=dict(color='#1f1f1f')
        )
        for patch in box_a['boxes']:
            patch.set_facecolor('#4c72b0')
        axes[0].set_title('SNR by channel')
        axes[0].set_ylabel('SNR (dB)')
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        axes[0].text(0.02, 0.98, '(a)', transform=axes[0].transAxes, ha='left', va='top', fontweight='bold')

        # (b) SNR by gesture (boxplot)
        unique_gestures = sorted(list(set(gestures)))
        snr_by_gesture = [[] for _ in range(len(unique_gestures))]
        for snr, gesture in zip(snr_data, gestures):
            gesture_idx = unique_gestures.index(gesture)
            snr_by_gesture[gesture_idx].append(snr)

        gesture_labels = ['Static' if g == 10 else str(g) for g in unique_gestures]
        box_b = axes[1].boxplot(
            snr_by_gesture,
            labels=gesture_labels,
            patch_artist=True,
            medianprops=dict(color='#1f1f1f'),
            boxprops=dict(color='#1f1f1f'),
            whiskerprops=dict(color='#1f1f1f'),
            capprops=dict(color='#1f1f1f')
        )
        for patch in box_b['boxes']:
            patch.set_facecolor('#55a868')
        axes[1].set_title('SNR by gesture')
        axes[1].set_ylabel('SNR (dB)')
        axes[1].set_xlabel('Gesture')
        if len(unique_gestures) > 6:
            axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        axes[1].yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        axes[1].text(0.02, 0.98, '(b)', transform=axes[1].transAxes, ha='left', va='top', fontweight='bold')

        # Identify lowest-SNR gestures (based on median)
        gesture_medians = []
        for values in snr_by_gesture:
            if len(values) > 0:
                gesture_medians.append(np.median(values))
            else:
                gesture_medians.append(np.nan)
        low_idx = np.argsort([m if np.isfinite(m) else np.inf for m in gesture_medians])[:3]
        low_gestures_list = [gesture_labels[i] for i in low_idx if np.isfinite(gesture_medians[i])]
 
        # Reference threshold lines and near-0 dB jittered hollow markers (grey)
        axes[0].axhline(10, color='#b3b3b3', linestyle='--', linewidth=0.8, zorder=0)
        axes[0].axhline(20, color='#b3b3b3', linestyle='--', linewidth=0.8, zorder=0)
        near_zero_thresh = 3.0
        jitter_rng = np.random.default_rng(7)
        for snr, ch in zip(snr_data, channels):
            if snr <= near_zero_thresh:
                jitter = float(jitter_rng.uniform(-0.06, 0.06))
                axes[0].scatter(ch + 1 + jitter, snr, facecolors='none', edgecolors='#c44e52', s=20, zorder=3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'snr_analysis.png'), dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'snr_analysis.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'snr_analysis.svg'), bbox_inches='tight')
        plt.close()
    
    def _plot_drift_analysis(self, plots_dir):
        """Create drift analysis plots: (a) slope by channel, (b) variance by channel (log scale)."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Prepare drift data
        trend_slopes = []
        drift_variances = []
        channels = []

        for result in self.results['drift_analysis']:
            for ch, metrics in enumerate(result['drift_metrics']):
                trend_slopes.append(metrics['trend_slope'])
                drift_variances.append(metrics['drift_variance'])
                channels.append(ch)

        # (a) Trend slopes by channel (boxplot)
        slopes_by_channel = [[] for _ in range(5)]
        for slope, ch in zip(trend_slopes, channels):
            slopes_by_channel[ch].append(slope)

        box_a = axes[0].boxplot(
            slopes_by_channel,
            labels=[f'Ch{i}' for i in range(5)],
            patch_artist=True,
            medianprops=dict(color='#1f1f1f'),
            boxprops=dict(color='#1f1f1f'),
            whiskerprops=dict(color='#1f1f1f'),
            capprops=dict(color='#1f1f1f')
        )
        for patch in box_a['boxes']:
            patch.set_facecolor('#4c72b0')
        axes[0].set_title('Drift slope')
        axes[0].set_ylabel('Slope (a.u./sample)')
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        axes[0].text(0.02, 0.98, '(a)', transform=axes[0].transAxes, ha='left', va='top', fontweight='bold')

        # (b) Drift variances by channel (log scale boxplot)
        variances_by_channel = [[] for _ in range(5)]
        for var, ch in zip(drift_variances, channels):
            # Ensure strictly positive for log scale
            safe_var = var if var > 0 else 1e-8
            variances_by_channel[ch].append(safe_var)

        box_b = axes[1].boxplot(
            variances_by_channel,
            labels=[f'Ch{i}' for i in range(5)],
            patch_artist=True,
            medianprops=dict(color='#1f1f1f'),
            boxprops=dict(color='#1f1f1f'),
            whiskerprops=dict(color='#1f1f1f'),
            capprops=dict(color='#1f1f1f')
        )
        for patch in box_b['boxes']:
            patch.set_facecolor('#55a868')
        axes[1].set_title('Drift variance (log scale)')
        axes[1].set_ylabel('Variance')
        axes[1].set_yscale('log')
        # Major ticks at decades and mathtext formatter 10^n
        axes[1].yaxis.set_major_locator(LogLocator(base=10.0))
        axes[1].yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        # Remove minor ticks and dense gridlines
        axes[1].yaxis.set_minor_locator(NullLocator())
        axes[1].grid(True, alpha=0.3, which='major')
        axes[1].text(0.02, 0.98, '(b)', transform=axes[1].transAxes, ha='left', va='top', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'drift_analysis.png'), dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'drift_analysis.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'drift_analysis.svg'), bbox_inches='tight')
        plt.close()
    
    def _plot_cross_user_variability(self, plots_dir):
        """Create cross-user variability plot: (a) CV by gesture only, filter NaN/empty."""
        if not self.results['cross_user_analysis']:
            print("   Skipping cross-user plots (no data)")
            return

        # Prepare data using new CV statistics (median across users) with 95% CI
        gestures = []
        median_cvs = []
        ci_lows = []
        ci_highs = []
        for gesture_id, data in self.results['cross_user_analysis'].items():
            if data.get('n_users', 0) < 2:
                continue
            m = data.get('median_cv', np.nan)
            lo = data.get('ci95_low', np.nan)
            hi = data.get('ci95_high', np.nan)
            if not (np.isfinite(m) and np.isfinite(lo) and np.isfinite(hi)):
                continue
            gestures.append(gesture_id)
            median_cvs.append(float(m))
            ci_lows.append(float(lo))
            ci_highs.append(float(hi))

        if not gestures:
            print("   No valid gestures for cross-user CV after filtering; skipping plot")
            return

        # Sort by gesture id for stable ordering
        order = np.argsort(gestures)
        gestures = [gestures[i] for i in order]
        median_cvs = [median_cvs[i] for i in order]
        ci_lows = [ci_lows[i] for i in order]
        ci_highs = [ci_highs[i] for i in order]

        gesture_labels = ['Static' if g == 10 else str(g) for g in gestures]

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x_positions = np.arange(len(gestures))
        bars = ax.bar(x_positions, median_cvs, color='#4c72b0')
        # Add 95% bootstrap CI as error bars
        yerr_lower = np.array(median_cvs) - np.array(ci_lows)
        yerr_upper = np.array(ci_highs) - np.array(median_cvs)
        ax.errorbar(x_positions, median_cvs, yerr=[yerr_lower, yerr_upper], fmt='none', ecolor='#4c72b0', elinewidth=1, capsize=3)
        ax.set_xlabel('Gesture')
        ax.set_ylabel('Coefficient of Variation (CV, unitless)')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(gesture_labels, rotation=45 if len(gestures) > 6 else 0)
        ax.grid(True, axis='y', alpha=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, ha='left', va='top', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'cross_user_variability.png'), dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'cross_user_variability.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'cross_user_variability.svg'), bbox_inches='tight')
        plt.close()
    
    def _plot_cross_session_consistency(self, plots_dir):
        """Create cross-session consistency plots.""" 
        if not self.results['cross_session_analysis']:
            print("   Skipping cross-session plots (no data)")
            return
        
        # This would be implemented similarly to cross-user analysis
        # For brevity, creating a placeholder
        print("   Cross-session consistency plots: placeholder implementation")
    
    def save_results(self):
        """Save all analysis results to files."""
        print("\n4. Saving analysis results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_path = os.path.join(self.output_dir, f'sensor_characterization_{timestamp}.json')
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(self.results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"   Detailed results saved: {results_path}")
        
        # Save summary CSV
        summary_data = []
        
        # SNR summary
        for result in self.results['snr_analysis']:
            for ch, snr_val in enumerate(result['snr_values']):
                summary_data.append({
                    'metric_type': 'SNR',
                    'user_id': result['user_id'],
                    'session_id': result['session_id'],
                    'gesture_id': result['gesture_id'],
                    'channel': ch,
                    'value': snr_val
                })
        
        # Drift summary
        for result in self.results['drift_analysis']:
            for ch, metrics in enumerate(result['drift_metrics']):
                summary_data.append({
                    'metric_type': 'Drift_Slope',
                    'user_id': result['user_id'],
                    'session_id': result['session_id'],
                    'gesture_id': result['gesture_id'],
                    'channel': ch,
                    'value': metrics['trend_slope']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, f'sensor_summary_{timestamp}.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print(f"   Summary CSV saved: {summary_path}")
        
        return results_path, summary_path
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj


def main():
    """Main function to run sensor characterization analysis."""
    print("="*60)
    print("BSL GESTURE RECOGNITION - SENSOR CHARACTERIZATION")
    print("="*60)
    
    # Initialize characterizer
    characterizer = SensorCharacterizer()
    
    # Load data
    data_collection = characterizer.load_sensor_data()
    
    if not data_collection:
        print("No data found. Please check the data directory.")
        return
    
    # Run comprehensive analysis
    characterizer.generate_comprehensive_analysis(data_collection)
    
    # Create visualizations
    characterizer.create_visualization_plots()
    
    # Save results
    results_path, summary_path = characterizer.save_results()
    
    # Print summary
    stability_metrics = characterizer.results['stability_metrics']
    print("\n" + "="*60)
    print("SENSOR CHARACTERIZATION COMPLETE")
    print("="*60)
    print(f"📊 Mean SNR: {stability_metrics['snr_statistics']['mean_snr']:.2f} dB")
    print(f"📈 SNR Range: {stability_metrics['snr_statistics']['min_snr']:.2f} - {stability_metrics['snr_statistics']['max_snr']:.2f} dB")
    print(f"📉 Mean Drift Slope: {stability_metrics['drift_statistics']['mean_drift_slope']:.6f}")
    print(f"📁 Results saved to: {characterizer.output_dir}")
    print(f"📄 Detailed results: {results_path}")
    print(f"📊 Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
