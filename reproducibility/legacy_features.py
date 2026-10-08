"""Exact legacy ADANN descriptor implementation; see REPRODUCTION_STATUS.md."""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from src.training.feature_utils import extract_wavelet_energies

class EnhancedFeatureExtractor:
    """增强特征提取器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_comprehensive_features(self, sample):
        """提取综合特征集（数值稳定版，避免NaN/Inf）"""
        features = []

        for ch in range(sample.shape[1]):
            channel_data = sample[:, ch]

            # 时域特征 (18个)：mean, std, var, min, max, ptp, median, skew, kurtosis, rms, mav, wl, zc, ssc, q1, q3, aav, outlier_cnt
            
            # 基础统计特征
            mean_val = float(np.mean(channel_data)) if len(channel_data) > 0 else 0.0
            std_val = float(np.std(channel_data)) if len(channel_data) > 0 else 0.0
            var_val = float(np.var(channel_data)) if len(channel_data) > 0 else 0.0
            min_val = float(np.min(channel_data)) if len(channel_data) > 0 else 0.0
            max_val = float(np.max(channel_data)) if len(channel_data) > 0 else 0.0
            ptp_val = float(np.ptp(channel_data)) if len(channel_data) > 0 else 0.0
            median_val = float(np.median(channel_data)) if len(channel_data) > 0 else 0.0

            # 偏度和峰度
            try:
                ch_skew = skew(channel_data) if len(channel_data) > 2 else 0.0
                ch_skew = 0.0 if (np.isnan(ch_skew) or np.isinf(ch_skew)) else float(ch_skew)
            except Exception:
                ch_skew = 0.0

            try:
                ch_kurt = kurtosis(channel_data) if len(channel_data) > 2 else 0.0
                ch_kurt = 0.0 if (np.isnan(ch_kurt) or np.isinf(ch_kurt)) else float(ch_kurt)
            except Exception:
                ch_kurt = 0.0

            # RMS和MAV
            rms_val = float(np.sqrt(np.mean(channel_data**2))) if len(channel_data) > 0 else 0.0
            mav_val = float(np.mean(np.abs(channel_data))) if len(channel_data) > 0 else 0.0

            # 波形长度和过零率
            if len(channel_data) > 1:
                wl_val = float(np.sum(np.abs(np.diff(channel_data))))
                zero_crossings = int(len(np.where(np.diff(np.sign(channel_data)))[0]))
            else:
                wl_val = 0.0
                zero_crossings = 0

            # 斜率符号变化 (SSC)
            ssc = 0
            if len(channel_data) > 2:
                for i in range(1, len(channel_data) - 1):
                    if (channel_data[i] > channel_data[i-1] and channel_data[i] > channel_data[i+1]) or \
                       (channel_data[i] < channel_data[i-1] and channel_data[i] < channel_data[i+1]):
                        ssc += 1

            # 分位数
            q1 = float(np.percentile(channel_data, 25)) if len(channel_data) > 0 else 0.0
            q3 = float(np.percentile(channel_data, 75)) if len(channel_data) > 0 else 0.0

            # 平均绝对变化率 (AAV)
            if len(channel_data) > 1:
                aav = float(np.mean(np.abs(np.diff(channel_data))))
            else:
                aav = 0.0

            # 异常值计数 (outlier_cnt) - 使用IQR方法
            if len(channel_data) > 0:
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_cnt = int(np.sum((channel_data < lower_bound) | (channel_data > upper_bound)))
            else:
                outlier_cnt = 0

            # 添加时域特征 (18个)
            features.extend([
                mean_val, std_val, var_val, min_val, max_val, ptp_val, median_val,
                ch_skew, ch_kurt, rms_val, mav_val, wl_val, zero_crossings, ssc,
                q1, q3, aav, outlier_cnt
            ])

            # 频域特征 (12个)：spectral_centroid, dominant_freq, total_power, spectral_spread, spectral_entropy, low_freq_power, mid_freq_power, high_freq_power, 2nd_moment, 3rd_moment, peak_factor, coeff_var
            try:
                freqs, psd = signal.periodogram(channel_data, fs=250)
                total_power = float(np.sum(psd))
                
                if total_power > 1e-12:
                    psd_norm = psd / total_power
                    spectral_centroid = float(np.sum(freqs * psd_norm))
                    spectral_spread = float(np.sqrt(np.sum(((freqs - spectral_centroid)**2) * psd_norm)))
                    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
                    
                    # 2nd和3rd moment
                    second_moment = float(np.sum((freqs - spectral_centroid)**2 * psd_norm))
                    third_moment = float(np.sum((freqs - spectral_centroid)**3 * psd_norm))
                else:
                    spectral_centroid = 0.0
                    spectral_spread = 0.0
                    spectral_entropy = 0.0
                    second_moment = 0.0
                    third_moment = 0.0

                dominant_freq = float(freqs[np.argmax(psd)]) if psd.size > 0 else 0.0
                low_freq_power = float(np.sum(psd[freqs <= 50])) if psd.size > 0 else 0.0
                mid_freq_power = float(np.sum(psd[(freqs > 50) & (freqs <= 100)])) if psd.size > 0 else 0.0
                high_freq_power = float(np.sum(psd[freqs > 100])) if psd.size > 0 else 0.0

                # Peak factor
                peak_factor = float(np.max(np.abs(channel_data)) / rms_val) if rms_val > 1e-10 else 0.0

                # Coefficient of variation
                coeff_var = float(std_val / mean_val) if abs(mean_val) > 1e-6 else 0.0

                features.extend([
                    spectral_centroid, dominant_freq, total_power, spectral_spread,
                    spectral_entropy, low_freq_power, mid_freq_power, high_freq_power,
                    second_moment, third_moment, peak_factor, coeff_var
                ])
            except Exception:
                features.extend([0.0] * 12)

            # 小波特征 (8个)：不同时间尺度上分析信号的能量分布
            try:
                features.extend(extract_wavelet_energies(channel_data))
            except Exception:
                features.extend([0.0] * 8)

        features = np.array(features, dtype=np.float32)
        # 全局兜底，防止任何残留的NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        return features
