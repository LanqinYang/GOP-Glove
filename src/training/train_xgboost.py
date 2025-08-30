"""
BSL Gesture Recognition - XGBoost Model Definition
"""
import numpy as np
import xgboost as xgb
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import scipy.signal as signal

# Constants
N_FEATURES = 5
N_CLASSES = 11

class XgboostModelCreator:
    def __init__(self):
        pass

    def _extract_features(self, X, arduino_mode):
        """提取190维特征（38个/通道）"""
        features = []
        for sample in X:
            sample_features = []
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
                sample_features.extend([
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
                    coeff_var = float(std_val / mean_val) if abs(mean_val) > 1e-10 else 0.0

                    sample_features.extend([
                        spectral_centroid, dominant_freq, total_power, spectral_spread,
                        spectral_entropy, low_freq_power, mid_freq_power, high_freq_power,
                        second_moment, third_moment, peak_factor, coeff_var
                    ])
                except Exception:
                    sample_features.extend([0.0] * 12)

                # 小波特征 (8个)：不同时间尺度上分析信号的能量分布
                try:
                    from scipy.signal import cwt, ricker
                    scales = np.arange(1, 9)  # 8个尺度
                    coeffs = cwt(channel_data, ricker, scales)
                    for i in range(8):
                        energy = float(np.sum(coeffs[i]**2))
                        energy = 0.0 if (np.isnan(energy) or np.isinf(energy)) else energy
                        sample_features.append(energy)
                except Exception:
                    sample_features.extend([0.0] * 8)

            features.append(sample_features)
        
        features = np.array(features, dtype=np.float32)
        # 全局兜底，防止任何残留的NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        return features

    def extract_and_scale_features(self, X_data, fit=False, scaler=None, arduino_mode=False):
        X_features = self._extract_features(X_data, arduino_mode)
        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_features)
            return X_scaled, scaler
        else:
            X_scaled = scaler.transform(X_features)
            # Ensure a tuple is always returned, consistent with other model creators
            return X_scaled, scaler

    def define_hyperparams(self, trial, arduino_mode=False):
        if arduino_mode:
            # 更激进的规模约束以保证导出头文件<256KB
            return {
                'n_estimators': trial.suggest_int('n_estimators', 10, 20),
                'max_depth': trial.suggest_int('max_depth', 2, 3),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                'gamma': trial.suggest_float('gamma', 0.1, 5.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 2, 6),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0)
            }
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                
                            # 数据增强参数 - 基于论文研究的保守设置
            'augment_factor': trial.suggest_int('augment_factor', 1, 2),
            'jitter_noise_level': trial.suggest_float('jitter_noise_level', 0.005, 0.015),
            'time_warp_max_speed': trial.suggest_int('time_warp_max_speed', 2, 3),
                            'scale_min': trial.suggest_float('scale_min', 0.95, 0.98),
                'scale_max': trial.suggest_float('scale_max', 1.02, 1.05),
            'augment_prob': trial.suggest_float('augment_prob', 0.3, 0.6)
            }
            
            # 构建scale_range
            params['scale_range'] = [params['scale_min'], params['scale_max']]
            
            return params

    def create_model(self, params, arduino_mode=False, callbacks=None):
        return xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=N_CLASSES,
            eval_metric='mlogloss',
            seed=42,
            callbacks=callbacks,
            **params
        ) 