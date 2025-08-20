#!/usr/bin/env python3
"""
Static Stability Analysis for BSL Gesture Recognition.
Based on the Original vs Repaired visualization pattern from the image.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

class StaticStabilityAnalyzer:
    """Static stability analysis based on Original vs Repaired visualization."""
    
    def __init__(self, data_dir="datasets/gesture_csv", output_dir="outputs/static_stability"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Channel names unified to Ch0–Ch4 (colorblind-friendly palette)
        self.channel_names = ['Ch0', 'Ch1', 'Ch2', 'Ch3', 'Ch4']
        # Colorblind-friendly colors (ggplot2-like): blue, orange, green, vermillion, reddish purple
        self.channel_colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']
        # Distinct line styles and markers to improve distinguishability in grayscale
        self.line_styles = ['solid', 'dashed', 'dotted', 'dashdot', (0, (5, 2, 1, 2))]
        self.line_markers = ['o', 's', '^', 'D', 'x']
        # Sampling rate used to convert cluster size to tau (seconds) for Allan deviation
        self.fs = 50.0
        
        # Analysis results
        self.results = {
            'stability_metrics': {},
            'comparison_data': {},
            'visualization_data': {}
        }
    
    def load_static_data(self):
        """Load static gesture data (gesture_10_Static)."""
        print("📊 Loading static gesture data...")
        
        # Find all static gesture files
        static_pattern = os.path.join(self.data_dir, "*gesture_10_Static*.csv")
        static_files = glob.glob(static_pattern)
        
        if not static_files:
            print("❌ No static gesture files found!")
            return None
        
        print(f"✅ Found {len(static_files)} static gesture files")
        
        # Load and organize data
        static_data = {}
        for file_path in static_files:
            # Extract user and sample info from filename
            filename = os.path.basename(file_path)
            match = re.search(r'user_(\d+)_gesture_10_Static_sample_(\d+)', filename)
            if match:
                user_id = match.group(1)
                sample_id = match.group(2)
                
                # Load CSV data
                try:
                    # Read CSV and skip comment lines properly
                    df = pd.read_csv(file_path, comment='#', skiprows=2)
                    if len(df) > 0:
                        # Check if columns exist and handle potential whitespace
                        expected_columns = ['thumb', 'index', 'middle', 'ring', 'pinky']
                        actual_columns = [col.strip() for col in df.columns]
                        
                        # Find matching columns
                        sensor_columns = []
                        for expected_col in expected_columns:
                            if expected_col in actual_columns:
                                sensor_columns.append(expected_col)
                            else:
                                print(f"⚠️ Column '{expected_col}' not found in {filename}")
                                print(f"   Available columns: {actual_columns}")
                                break
                        
                        if len(sensor_columns) == 5:
                            sensor_data = df[sensor_columns].values
                        static_data[f"user_{user_id}_sample_{sample_id}"] = {
                            'data': sensor_data,
                            'user_id': user_id,
                            'sample_id': sample_id,
                            'file_path': file_path
                        }
                except Exception as e:
                    print(f"⚠️ Error loading {filename}: {e}")
        
        print(f"✅ Successfully loaded {len(static_data)} static samples")
        return static_data
    
    def create_original_vs_repaired_plots(self, static_data, num_samples=3):
        """Create Original vs Repaired comparison plots like in the image."""
        print("🎨 Creating Original vs Repaired comparison plots...")
        
        # Select samples to plot
        sample_keys = list(static_data.keys())[:num_samples]
        
        # Create figure with subplots
        fig, axes = plt.subplots(num_samples, 2, figsize=(15, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample_key in enumerate(sample_keys):
            sample_data = static_data[sample_key]
            sensor_data = sample_data['data']
            
            # Original data (with boundary markers)
            ax_original = axes[i, 0]
            self._plot_sensor_data(ax_original, sensor_data, "Original", sample_key, add_boundaries=True)
            
            # Repaired data (cleaned)
            ax_repaired = axes[i, 1]
            repaired_data = self._repair_sensor_data(sensor_data)
            self._plot_sensor_data(ax_repaired, repaired_data, "Repaired", sample_key, add_boundaries=False)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.output_dir, f"static_stability_comparison_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Comparison plots saved: {plot_path}")
        return plot_path
    
    def _plot_sensor_data(self, ax, data, title_prefix, sample_key, add_boundaries=True):
        """Plot sensor data for one sample."""
        time_points = np.arange(len(data))
        
        # Plot each channel
        for ch in range(5):
            ax.plot(time_points, data[:, ch], 
                   color=self.channel_colors[ch], 
                   linewidth=1.5, 
                   label=self.channel_names[ch])
        
        # Add boundary markers (red dashed lines) for original
        if add_boundaries:
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(x=len(data)-1, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # Customize plot
        ax.set_title(f"{title_prefix} - {sample_key}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (samples)", fontsize=10)
        ax.set_ylabel("Signal Value", fontsize=10)
        ax.set_ylim(0, 600)  # Match image scale
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        
        # Add stability metrics as text
        stability_text = self._calculate_stability_metrics(data)
        ax.text(0.02, 0.98, stability_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=8, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _repair_sensor_data(self, data):
        """Apply repair/cleaning to sensor data."""
        # Simple repair: remove outliers and smooth
        repaired_data = data.copy()
        
        # Remove outliers (values > 3 std from mean)
        for ch in range(5):
            channel_data = repaired_data[:, ch]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            
            # Replace outliers with mean
            outliers = np.abs(channel_data - mean_val) > 3 * std_val
            repaired_data[outliers, ch] = mean_val
        
        # Apply light smoothing
        window_size = 3
        for ch in range(5):
            channel_data = repaired_data[:, ch]
            smoothed = np.convolve(channel_data, np.ones(window_size)/window_size, mode='same')
            repaired_data[:, ch] = smoothed
        
        return repaired_data
    
    def _calculate_stability_metrics(self, data):
        """Calculate stability metrics for display."""
        metrics = []
        
        # Overall stability
        overall_std = np.std(data)
        overall_cv = overall_std / np.mean(data) * 100
        
        # Channel-wise stability
        channel_stability = []
        for ch in range(5):
            ch_std = np.std(data[:, ch])
            ch_cv = ch_std / np.mean(data[:, ch]) * 100
            channel_stability.append(ch_cv)
        
        avg_channel_cv = np.mean(channel_stability)
        
        metrics.append(f"Overall CV: {overall_cv:.1f}%")
        metrics.append(f"Avg Ch CV: {avg_channel_cv:.1f}%")
        metrics.append(f"Max Ch CV: {max(channel_stability):.1f}%")
        
        return "\n".join(metrics)

    # ========================= New: Main Figure A (2x2 panels) =========================
    def _compute_user_channel_baselines(self, static_data):
        """Compute per-user, per-channel baseline (median) across all their static samples."""
        user_channel_values = {}
        for sample_key, sample_info in static_data.items():
            user_id = sample_info['user_id']
            data = sample_info['data']
            if user_id not in user_channel_values:
                user_channel_values[user_id] = [[] for _ in range(5)]
            for ch in range(5):
                user_channel_values[user_id][ch].extend(data[:, ch].tolist())

        user_channel_baseline = {}
        for user_id, lists_per_ch in user_channel_values.items():
            medians = []
            for ch in range(5):
                arr = np.array(lists_per_ch[ch])
                medians.append(float(np.median(arr)) if arr.size > 0 else 0.0)
            user_channel_baseline[user_id] = medians
        return user_channel_baseline

    def _mad(self, x):
        """Median Absolute Deviation (MAD)."""
        x = np.asarray(x)
        med = np.median(x)
        return np.median(np.abs(x - med))

    def _robust_slope(self, y):
        """Compute a robust slope by removing outliers via MAD then fitting a line."""
        y = np.asarray(y)
        n = y.size
        if n < 2:
            return 0.0
        x = np.arange(n)
        med = np.median(y)
        mad = self._mad(y)
        threshold = med if np.isnan(med) else med
        scale = mad if mad > 1e-8 else 1.0
        mask = np.abs(y - threshold) <= 3.0 * scale
        if mask.sum() < 2:
            return 0.0
        coeffs = np.polyfit(x[mask], y[mask], 1)
        return float(coeffs[0])

    def _allan_deviation(self, y, m_values):
        """Compute Allan deviation for a 1D series for specified cluster sizes m_values.
        Returns array of sigma for each m in m_values. Uses non-overlapping clusters.
        """
        y = np.asarray(y)
        n = y.size
        sigmas = []
        for m in m_values:
            if m < 1 or 2 * m > n:
                sigmas.append(np.nan)
                continue
            # Compute cluster means
            num_clusters = n // m
            trimmed = y[:num_clusters * m]
            clusters = trimmed.reshape(num_clusters, m)
            means = clusters.mean(axis=1)
            # Successive differences
            diffs = np.diff(means)
            if diffs.size == 0:
                sigmas.append(np.nan)
                continue
            avar = 0.5 * np.mean(diffs ** 2)
            sigmas.append(np.sqrt(avar))
        return np.array(sigmas)

    def _compute_within_user_cv_df(self, static_data):
        """Compute robust within-user CV (CVr) per channel for each static segment.
        CVr = 100 * (1.4826 * MAD) / median, computed on each sample's channel series.
        """
        records = []
        for sample_key, sample_info in static_data.items():
            user_id = sample_info['user_id']
            data = sample_info['data']
            for ch in range(5):
                y = data[:, ch]
                med = np.median(y)
                mad = self._mad(y)
                # Avoid explosion when median is near zero; epsilon lower bound
                denom = max(abs(med), 1.0)
                cvr = float(100.0 * (1.4826 * mad) / denom)
                records.append({
                    'user_id': user_id,
                    'sample_key': sample_key,
                    'channel': f'Ch{ch}',
                    'cv': cvr
                })
        import pandas as pd  # local import for serialization safety
        return pd.DataFrame.from_records(records)

    def _compute_spike_rates(self, static_data):
        """Compute spike/dropout rate per sample and channel using |x - med| > 3*MAD."""
        channel_to_rates = {f'Ch{ch}': [] for ch in range(5)}
        for sample_key, sample_info in static_data.items():
            data = sample_info['data']
            for ch in range(5):
                y = data[:, ch]
                med = np.median(y)
                mad = self._mad(y)
                scale = mad if mad > 1e-8 else 1.0
                spikes = np.abs(y - med) > 3.0 * scale
                rate = float(np.sum(spikes) / y.size)
                channel_to_rates[f'Ch{ch}'].append(rate)
        # Aggregate stats
        stats = {}
        for ch, rates in channel_to_rates.items():
            arr = np.array(rates)
            stats[ch] = {
                'median': float(np.median(arr)) if arr.size else 0.0,
                'q1': float(np.percentile(arr, 25)) if arr.size else 0.0,
                'q3': float(np.percentile(arr, 75)) if arr.size else 0.0,
                'values': arr.tolist()
            }
        return stats

    def _select_single_user_five_channels(self, cv_df, static_data):
        """Select one representative static sample (single user segment) using median aggregated CVr."""
        if cv_df.empty:
            return None
        grouped = cv_df.groupby('sample_key')['cv'].mean()
        target = grouped.median()
        sample_key = (grouped - target).abs().sort_values().index[0]
        return sample_key, static_data[sample_key]

    def _select_representative_trajectories(self, static_data, cv_df):
        """Select two representative (user, sample, channel) combos: one low CV, one high CV."""
        if cv_df.empty:
            return []
        # Sort by CV and pick 25th and 75th percentile examples
        q25 = cv_df['cv'].quantile(0.25)
        q75 = cv_df['cv'].quantile(0.75)
        low_row = cv_df.iloc[(cv_df['cv'] - q25).abs().argsort()].head(1).iloc[0]
        high_row = cv_df.iloc[(cv_df['cv'] - q75).abs().argsort()].head(1).iloc[0]
        selections = []
        for row in [low_row, high_row]:
            sample_key = row['sample_key']
            ch_idx = int(row['channel'][2:])
            sample_info = static_data[sample_key]
            y = sample_info['data'][:, ch_idx]
            # Metrics for annotation
            med = np.median(y)
            mad = self._mad(y)
            scale = mad if mad > 1e-8 else 1.0
            spikes = np.abs(y - med) > 3.0 * scale
            spike_rate = float(np.sum(spikes) / y.size)
            slope = self._robust_slope(y)
            # CV as within-user centered relative to baseline
            # Use per-user baseline for denominator
            selections.append({
                'sample_key': sample_key,
                'user_id': sample_info['user_id'],
                'channel': row['channel'],
                'channel_index': ch_idx,
                'series': y,
                'cv': float(row['cv']),
                'slope': slope,
                'spike_rate': spike_rate
            })
        return selections

    def _compute_allan_across_channels(self, static_data):
        """Compute Allan deviation curves aggregated per channel (median across samples)."""
        # Determine common m values based on minimum length
        lengths = [v['data'].shape[0] for v in static_data.values()]
        if not lengths:
            return None
        min_len = int(np.min(lengths))
        if min_len < 8:
            return None
        m_values = []
        m = 1
        while 2 * m <= min_len:
            m_values.append(m)
            m *= 2
        if len(m_values) < 2:
            return None

        channel_sigmas = {}
        for ch in range(5):
            all_sigmas = []
            for sample_key, sample_info in static_data.items():
                y = sample_info['data'][:, ch]
                # Center to remove bias; Allan does not require detrending here
                y_centered = y - np.median(y)
                sigmas = self._allan_deviation(y_centered, m_values)
                all_sigmas.append(sigmas)
            all_sigmas = np.array(all_sigmas)  # shape: num_samples x len(m_values)
            median_sigma = np.nanmedian(all_sigmas, axis=0)
            channel_sigmas[f'Ch{ch}'] = median_sigma.tolist()
        return {'m_values': m_values, 'channel_sigmas': channel_sigmas}

    def create_static_main_figure_A(self, static_data):
        """Create the Main Figure A (2x2):
        (a) Within-user centered CV by channel (boxplot)
        (b) Static drift via Allan deviation (log-log) if available; otherwise robust slope boxplot
        (c) Spike/dropout rate per channel (bar with IQR)
        (d) Two example trajectories (0-100 samples) with metrics
        """
        print("🎨 Creating Main Figure A (Static stability 2×2)...")

        # Unified styling (grayscale, consistent fonts)
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 10,
            'axes.labelsize': 10,
            'legend.fontsize': 8,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9
        })

        # Panel (a): Within-user centered CV distribution
        cv_df = self._compute_within_user_cv_df(static_data)

        # Panel (b): Drift via Allan deviation or fallback to slope boxplot
        allan_result = self._compute_allan_across_channels(static_data)

        # Panel (c): Spike/dropout rate stats
        spike_stats = self._compute_spike_rates(static_data)

        # Panel (d): Example trajectories selection
        selections = self._select_representative_trajectories(static_data, cv_df)

        # Build figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

        # (a) CV boxplot
        ax_a = axes[0, 0]
        sns.boxplot(data=cv_df, x='channel', y='cv', ax=ax_a, palette=self.channel_colors)
        ax_a.set_title('A1 (a) Static CVr by Channel (within-user, robust)')
        ax_a.set_xlabel('Channel')
        ax_a.set_ylabel('CVr (%)')
        ax_a.grid(True, axis='y', alpha=0.3)
        ax_a.text(0.02, 0.98, '(a)', transform=ax_a.transAxes, va='top', ha='left', fontweight='bold')

        # (b) Allan deviation or slope
        ax_b = axes[0, 1]
        if allan_result is not None:
            m_values = np.array(allan_result['m_values'])
            tau = m_values / self.fs
            for idx, ch in enumerate([f'Ch{i}' for i in range(5)]):
                sigma = np.array(allan_result['channel_sigmas'][ch])
                # Light smoothing (moving average) on log-sigma to reduce visual noise
                log_sigma = np.log10(sigma)
                if log_sigma.size >= 3:
                    kernel = np.ones(3) / 3.0
                    log_sigma = np.convolve(log_sigma, kernel, mode='same')
                x_vals = np.log10(tau)
                finite_mask = np.isfinite(log_sigma) & np.isfinite(x_vals)
                # Plot with distinct linestyle and marker
                mark_every = max(1, int(np.count_nonzero(finite_mask) / 12))
                ax_b.plot(
                    x_vals[finite_mask], log_sigma[finite_mask],
                    label=ch,
                    color=self.channel_colors[idx],
                    linestyle=self.line_styles[idx],
                    marker=self.line_markers[idx],
                    markersize=3,
                    markevery=mark_every,
                    linewidth=1.6,
                )
                # Direct label at the last valid point
                if np.any(finite_mask):
                    last_idx = np.where(finite_mask)[0][-1]
                    ax_b.text(
                        x_vals[last_idx], log_sigma[last_idx], f" {ch}",
                        fontsize=8, va='center', ha='left', color=self.channel_colors[idx],
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.6, linewidth=0)
                    )
            ax_b.set_title('A1 (b) Static Drift (Allan deviation)')
            ax_b.set_xlabel('log10 τ (s)')
            ax_b.set_ylabel('log10 Allan deviation')
            ax_b.grid(True, alpha=0.3)
            ax_b.legend(fontsize=8)
            ax_b.text(0.02, 0.98, '(b)', transform=ax_b.transAxes, va='top', ha='left', fontweight='bold')
        else:
            # Fallback: robust slope boxplot per channel
            slope_records = []
            for sample_key, sample_info in static_data.items():
                data = sample_info['data']
                for ch in range(5):
                    slope = self._robust_slope(data[:, ch])
                    slope_records.append({'channel': f'Ch{ch}', 'slope': slope})
            slope_df = pd.DataFrame.from_records(slope_records)
            sns.boxplot(data=slope_df, x='channel', y='slope', ax=ax_b, palette=self.channel_colors)
            ax_b.set_title('A1 (b) Static Drift (robust slope)')
            ax_b.set_xlabel('Channel')
            ax_b.set_ylabel('Slope (a.u./sample)')
            ax_b.grid(True, axis='y', alpha=0.3)
            ax_b.text(0.02, 0.98, '(b)', transform=ax_b.transAxes, va='top', ha='left', fontweight='bold')

        # (c) Spike/dropout rate
        ax_c = axes[1, 0]
        channels = [f'Ch{i}' for i in range(5)]
        # Non-symmetric error bars using Q1/Q3 relative to median (convert to %)
        medians = np.array([spike_stats[ch]['median'] * 100.0 for ch in channels])
        q1 = np.array([spike_stats[ch]['q1'] * 100.0 for ch in channels])
        q3 = np.array([spike_stats[ch]['q3'] * 100.0 for ch in channels])
        lower = np.clip(medians - q1, 0.0, None)
        upper = np.clip(q3 - medians, 0.0, None)
        yerr = np.vstack([lower, upper])
        bars = ax_c.bar(channels, medians, yerr=yerr, color=self.channel_colors, alpha=0.85, capsize=4)
        ax_c.set_title('A1 (c) Spike/Dropout Rate (|x − med| > 3·MAD)')
        ax_c.set_ylabel('% of samples')
        ax_c.set_xlabel('Channel')
        ax_c.set_ylim(0, max(5.0, float(medians.max()) * 1.3) if medians.size else 10.0)
        ax_c.grid(True, axis='y', alpha=0.3)
        ax_c.text(0.02, 0.98, '(c)', transform=ax_c.transAxes, va='top', ha='left', fontweight='bold')

        # (d) Example trajectories: single user with five sensors overlay
        ax_d = axes[1, 1]
        ax_d.set_title('A1 (d) Example Static Trajectories (single user, Ch0–Ch4)')
        single_sel = self._select_single_user_five_channels(cv_df, static_data)
        table_lines = []
        if single_sel is not None:
            sample_key, sample_info = single_sel
            data = sample_info['data']
            clip_len = min(100, data.shape[0])
            x_vals = np.arange(clip_len)
            # Append sample identity to title for clarity
            ax_d.set_title(f"A1 (d) Example Static Trajectories (single user, Ch0–Ch4)\n{sample_key}")
            for ch in range(5):
                y = data[:clip_len, ch]
                ax_d.plot(
                    x_vals, y,
                    color=self.channel_colors[ch],
                    linestyle=self.line_styles[ch],
                    marker=self.line_markers[ch],
                    markersize=2.5,
                    linewidth=1.8,
                    label=f'Ch{ch}',
                )
                # Metrics across the whole segment for annotation
                full_y = data[:, ch]
                med = np.median(full_y)
                mad = self._mad(full_y)
                denom = max(abs(med), 1.0)
                cvr = float(100.0 * (1.4826 * mad) / denom)
                slope = self._robust_slope(full_y)
                spikes = np.abs(full_y - med) > 3.0 * (mad if mad > 1e-8 else 1.0)
                spike_rate = float(np.sum(spikes) / full_y.size)
                table_lines.append(f"Ch{ch}: CV {cvr:.1f}% | slope {slope:.3g} | spikes {spike_rate:.2%}")
            ax_d.set_xlabel('Sample index')
            ax_d.set_ylabel('Signal Value')
            ax_d.grid(True, alpha=0.3)
            # Legend at center-left to reduce occlusion
            ax_d.legend(fontsize=8, loc='center left', bbox_to_anchor=(0.02, 0.5), ncol=1, framealpha=0.9)
            # Annotation box at center-right
            ax_d.text(0.95, 0.5, "\n".join(table_lines), transform=ax_d.transAxes,
                      ha='right', va='center', fontsize=8,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax_d.text(0.02, 0.98, '(d)', transform=ax_d.transAxes, va='top', ha='left', fontweight='bold')

        # Use constrained_layout above for spacing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save multi-format outputs: PDF, SVG, and high-DPI PNG
        base = os.path.join(self.output_dir, f"static_main_figure_A_{timestamp}")
        out_png = f"{base}.png"
        out_pdf = f"{base}.pdf"
        out_svg = f"{base}.svg"
        plt.savefig(out_pdf, bbox_inches='tight')
        plt.savefig(out_svg, bbox_inches='tight')
        plt.savefig(out_png, dpi=800, bbox_inches='tight')
        plt.close()

        # Store results for export
        self.results['visualization_data']['figure_A_paths'] = {
            'png': out_png,
            'pdf': out_pdf,
            'svg': out_svg
        }
        self.results['stability_metrics']['within_user_cv'] = cv_df.groupby('channel')['cv'].apply(list).to_dict()
        self.results['stability_metrics']['spike_rate'] = spike_stats
        if allan_result is not None:
            self.results['stability_metrics']['allan_deviation'] = allan_result
        else:
            self.results['stability_metrics']['allan_deviation'] = None
        self.results['stability_metrics']['example_trajectories'] = [
            {
                'sample_key': sel['sample_key'],
                'user_id': sel['user_id'],
                'channel': sel['channel'],
                'cv': sel['cv'],
                'slope': sel['slope'],
                'spike_rate': sel['spike_rate']
            } for sel in selections[:2]
        ]

        print(f"📈 Main Figure A saved: {out_png}")
        return out_png
    
    def analyze_static_stability(self, static_data):
        """Comprehensive static stability analysis."""
        print("🔬 Analyzing static stability...")
        
        stability_results = {
            'overall_metrics': {},
            'channel_metrics': {},
            'sample_metrics': {}
        }
        
        # Collect all data for analysis
        all_data = []
        channel_data = [[] for _ in range(5)]
        
        for sample_key, sample_info in static_data.items():
            data = sample_info['data']
            all_data.append(data)
            
            # Channel-wise analysis
            for ch in range(5):
                channel_data[ch].extend(data[:, ch])
            
            # Sample-wise metrics
            sample_metrics = {
                'mean': np.mean(data),
                'std': np.std(data),
                'cv': np.std(data) / np.mean(data) * 100,
                'range': np.max(data) - np.min(data),
                'user_id': sample_info['user_id']
            }
            stability_results['sample_metrics'][sample_key] = sample_metrics
        
        # Overall metrics
        all_data_array = np.vstack(all_data)
        stability_results['overall_metrics'] = {
            'mean': np.mean(all_data_array),
            'std': np.std(all_data_array),
            'cv': np.std(all_data_array) / np.mean(all_data_array) * 100,
            'range': np.max(all_data_array) - np.min(all_data_array)
        }
        
        # Channel-wise metrics
        for ch in range(5):
            ch_data = np.array(channel_data[ch])
            stability_results['channel_metrics'][f'Ch{ch}'] = {
                'mean': np.mean(ch_data),
                'std': np.std(ch_data),
                'cv': np.std(ch_data) / np.mean(ch_data) * 100,
                'range': np.max(ch_data) - np.min(ch_data)
            }
        
        self.results['stability_metrics'] = stability_results
        return stability_results
    
    def create_stability_summary_plots(self):
        """Create summary plots for static stability analysis."""
        print("📊 Creating stability summary plots...")
        
        if not self.results['stability_metrics']:
            print("⚠️ No stability metrics available")
            return
        
        metrics = self.results['stability_metrics']
        
        # Create summary figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Static Stability Analysis Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Channel-wise CV
        channels = list(metrics['channel_metrics'].keys())
        cv_values = [metrics['channel_metrics'][ch]['cv'] for ch in channels]
        
        axes[0, 0].bar(channels, cv_values, color=self.channel_colors, alpha=0.7)
        axes[0, 0].set_title('Coefficient of Variation by Channel')
        axes[0, 0].set_ylabel('CV (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Sample-wise CV distribution
        sample_cvs = [metrics['sample_metrics'][key]['cv'] for key in metrics['sample_metrics']]
        axes[0, 1].hist(sample_cvs, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Distribution of Sample CVs')
        axes[0, 1].set_xlabel('CV (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Channel-wise signal ranges
        ranges = [metrics['channel_metrics'][ch]['range'] for ch in channels]
        axes[1, 0].bar(channels, ranges, color=self.channel_colors, alpha=0.7)
        axes[1, 0].set_title('Signal Range by Channel')
        axes[1, 0].set_ylabel('Range')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: User-wise stability comparison
        user_cvs = {}
        for sample_key, sample_metrics in metrics['sample_metrics'].items():
            user_id = sample_metrics['user_id']
            if user_id not in user_cvs:
                user_cvs[user_id] = []
            user_cvs[user_id].append(sample_metrics['cv'])
        
        user_ids = list(user_cvs.keys())
        user_avg_cvs = [np.mean(user_cvs[uid]) for uid in user_ids]
        
        axes[1, 1].bar(user_ids, user_avg_cvs, alpha=0.7, color='lightcoral')
        axes[1, 1].set_title('Average CV by User')
        axes[1, 1].set_xlabel('User ID')
        axes[1, 1].set_ylabel('Average CV (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_plot_path = os.path.join(self.output_dir, f"static_stability_summary_{timestamp}.png")
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Summary plots saved: {summary_plot_path}")
        return summary_plot_path
    
    def save_results(self):
        """Save analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_path = os.path.join(self.output_dir, f"static_stability_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            import json
            json.dump(self.results, f, indent=2, default=str)
        
        # Create summary CSV
        if self.results['stability_metrics']:
            metrics = self.results['stability_metrics']
            
            # Channel summary
            channel_summary = []
            for ch, ch_metrics in metrics['channel_metrics'].items():
                channel_summary.append({
                    'Channel': ch,
                    'Mean': ch_metrics['mean'],
                    'Std': ch_metrics['std'],
                    'CV_Percent': ch_metrics['cv'],
                    'Range': ch_metrics['range']
                })
            
            # Sample summary
            sample_summary = []
            for sample_key, sample_metrics in metrics['sample_metrics'].items():
                sample_summary.append({
                    'Sample': sample_key,
                    'User_ID': sample_metrics['user_id'],
                    'Mean': sample_metrics['mean'],
                    'Std': sample_metrics['std'],
                    'CV_Percent': sample_metrics['cv'],
                    'Range': sample_metrics['range']
                })
            
            # Save CSVs
            channel_csv_path = os.path.join(self.output_dir, f"channel_stability_{timestamp}.csv")
            pd.DataFrame(channel_summary).to_csv(channel_csv_path, index=False)
            
            sample_csv_path = os.path.join(self.output_dir, f"sample_stability_{timestamp}.csv")
            pd.DataFrame(sample_summary).to_csv(sample_csv_path, index=False)
            
            print(f"📊 Channel summary: {channel_csv_path}")
            print(f"📊 Sample summary: {sample_csv_path}")
        
        print(f"📄 Detailed results: {results_path}")
        return results_path

def main():
    """Main function to run static stability analysis."""
    print("="*60)
    print("STATIC STABILITY ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = StaticStabilityAnalyzer()
    
    # Load static data
    static_data = analyzer.load_static_data()
    
    if not static_data:
        print("❌ No static data available for analysis")
        return
    
    # Create Main Figure A (replacing previous comparison plots; no 'Repaired')
    figure_a_path = analyzer.create_static_main_figure_A(static_data)
    
    # Analyze stability
    stability_results = analyzer.analyze_static_stability(static_data)
    
    # Optionally, create legacy summary plots if needed (disabled by default)
    summary_plot = None
    
    # Save results
    results_path = analyzer.save_results()
    
    # Print summary
    if stability_results['overall_metrics']:
        overall = stability_results['overall_metrics']
        print("\n" + "="*60)
        print("STATIC STABILITY ANALYSIS COMPLETE")
        print("="*60)
        print(f"📊 Overall CV: {overall['cv']:.2f}%")
        print(f"📈 Signal Range: {overall['range']:.2f}")
        print(f"📉 Mean Signal: {overall['mean']:.2f}")
        print(f"📁 Results saved to: {analyzer.output_dir}")
        print(f"📄 Detailed results: {results_path}")
        print(f"🎨 Main Figure A: {figure_a_path}")
        print(f"📊 Summary plots: {summary_plot}")

if __name__ == "__main__":
    main()
