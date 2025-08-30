"""
BSL Gesture Recognition - LightGBM Model Definition
基于增强特征工程的LightGBM手势识别模型

核心特性:
1. 综合特征提取 (时域、频域、非线性特征)
2. 滑动窗口特征聚合
3. Optuna超参数优化
4. 类别平衡损失函数
5. Arduino兼容的轻量级版本
"""

import numpy as np
import lightgbm as lgb
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import optuna

# Constants
SEQUENCE_LENGTH = 100
N_FEATURES = 5
N_CLASSES = 11
# Sampling rate used for frequency features should match data collection
FS_HZ = 50.0  # Hz, aligned with configs and pipeline (100 samples ≈ 2s at 50Hz)

class LightgbmModelWrapper:
    """LightGBM模型包装器，兼容Keras风格的pipeline"""
    
    def __init__(self, lgb_model, model_creator, params):
        self.lgb_model = lgb_model
        self.model_creator = model_creator
        self.params = params
        self.history = None
        self.trained = False
        
    def fit(self, X_train, y_train, validation_data=None, epochs=10, callbacks=None, verbose=1, **kwargs):
        """兼容Keras风格的fit方法"""
        if verbose:
            print(f"🚀 Training LightGBM model...")
        
        try:
            # 训练模型 (使用验证集和早停)
            if validation_data is not None:
                X_val, y_val = validation_data
                # 使用验证集训练以启用早停
                self.lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_names=['valid'],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]  # 静默训练
                )
            else:
                # 无验证集时的普通训练
                self.lgb_model.fit(X_train, y_train)
            
            # 计算训练准确率
            train_pred = self.lgb_model.predict(X_train)
            accuracy = np.mean(train_pred == y_train)
            
            # 创建简化的history对象
            self.history = {
                'accuracy': [accuracy],
                'loss': [0.0],  # LightGBM不直接提供损失值
                'val_accuracy': [],
                'val_loss': []
            }
            
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self.lgb_model.predict(X_val)
                val_accuracy = np.mean(val_pred == y_val)
                self.history['val_accuracy'] = [val_accuracy]
                self.history['val_loss'] = [0.0]
                if verbose:
                    print(f"   验证准确率: {val_accuracy:.4f}")
            
            self.trained = True
            if verbose:
                print(f"✅ LightGBM training completed with accuracy: {accuracy:.4f}")
            
            return self.history
            
        except Exception as e:
            if verbose:
                print(f"❌ LightGBM training failed: {e}")
            raise e
    
    def predict(self, X):
        """预测方法"""
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            return self.lgb_model.predict(X)
        except Exception as e:
            print(f"⚠️ LightGBM prediction failed: {e}")
            raise e
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            return self.lgb_model.predict_proba(X)
        except Exception as e:
            print(f"⚠️ LightGBM prediction failed: {e}")
            raise e
    
    def save(self, filepath):
        """保存模型 - 兼容Keras格式"""
        if self.trained:
            import pickle
            # 转换路径格式
            lgb_filepath = filepath.replace('.keras', '.pkl')
            
            # 保存LightGBM模型和相关信息
            model_data = {
                'lgb_model': self.lgb_model,
                'params': self.params,
                'trained': self.trained,
                'model_creator': self.model_creator
            }
            
            with open(lgb_filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ LightGBM model saved to {lgb_filepath}")
        else:
            print("⚠️ Model not trained, cannot save.")
    
    def load(self, filepath):
        """加载模型"""
        import pickle
        
        # 转换路径格式
        lgb_filepath = filepath.replace('.keras', '.pkl')
        
        try:
            with open(lgb_filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.lgb_model = model_data['lgb_model']
            self.params = model_data['params']
            self.trained = model_data['trained']
            
            print(f"✅ LightGBM model loaded from {lgb_filepath}")
        except Exception as e:
            print(f"❌ Failed to load LightGBM model: {e}")
            raise e

class EnhancedFeatureExtractor:
    """增强特征提取器 - 结合传统ML和深度学习特征"""
    
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
                freqs, psd = signal.periodogram(channel_data, fs=FS_HZ)
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

                features.extend([
                    spectral_centroid, dominant_freq, total_power, spectral_spread,
                    spectral_entropy, low_freq_power, mid_freq_power, high_freq_power,
                    second_moment, third_moment, peak_factor, coeff_var
                ])
            except Exception:
                features.extend([0.0] * 12)

            # 小波特征 (8个)：不同时间尺度上分析信号的能量分布
            try:
                from scipy.signal import cwt, ricker
                scales = np.arange(1, 9)  # 8个尺度
                coeffs = cwt(channel_data, ricker, scales)
                for i in range(8):
                    energy = float(np.sum(coeffs[i]**2))
                    energy = 0.0 if (np.isnan(energy) or np.isinf(energy)) else energy
                    features.append(energy)
            except Exception:
                features.extend([0.0] * 8)

        features = np.array(features, dtype=np.float32)
        # 全局兜底，防止任何残留的NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        return features
    
    def extract_sliding_window_features(self, sample, window_size=50, step=25):
        """滑动窗口特征提取"""
        n_timesteps = sample.shape[0]
        all_features = []
        
        # 滑动窗口
        for start in range(0, n_timesteps - window_size + 1, step):
            end = start + window_size
            window_data = sample[start:end]
            features = self.extract_comprehensive_features(window_data)
            all_features.append(features)
        
        if len(all_features) == 0:
            # 如果窗口太大，使用整个样本
            return self.extract_comprehensive_features(sample)
        
        # 聚合窗口特征：均值、标准差、最大值、最小值
        all_features = np.array(all_features)
        aggregated = np.concatenate([
            np.mean(all_features, axis=0),
            np.std(all_features, axis=0),
            np.max(all_features, axis=0),
            np.min(all_features, axis=0)
        ])
        
        return aggregated

class LightgbmModelCreator:
    """LightGBM模型创建器"""
    
    def __init__(self):
        self.model_name = "LightGBM"
        self.model_abbreviation = "LGB"
        self.feature_extractor = EnhancedFeatureExtractor()
    
    def extract_and_scale_features(self, X_data, fit=False, scaler=None, arduino_mode=False):
        """特征提取和标准化"""
        # 提取特征
        X_features = []
        for sample in X_data:
            
            if arduino_mode:
                # Arduino模式：简化特征
                features = self._extract_arduino_features(sample)
            else:
                # 完整特征提取
                features = self.feature_extractor.extract_sliding_window_features(sample)
            X_features.append(features)
        
        X_features = np.array(X_features)
        
        # 标准化
        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_features)
            return X_scaled, scaler
        else:
            X_scaled = scaler.transform(X_features)
            return X_scaled, scaler
    
    def _extract_arduino_features(self, sample):
        """Arduino-optimized lightweight features: mean, std, min, max per channel (4 each)."""
        features = []
        for ch in range(sample.shape[1]):
            channel_data = sample[:, ch]
            mu = float(np.mean(channel_data)) if len(channel_data) > 0 else 0.0
            sd = float(np.std(channel_data)) if len(channel_data) > 0 else 0.0
            mn = float(np.min(channel_data)) if len(channel_data) > 0 else 0.0
            mx = float(np.max(channel_data)) if len(channel_data) > 0 else 0.0
            features.extend([mu, sd, mn, mx])
        return np.array(features, dtype=np.float32)
    
    def define_hyperparams(self, trial, arduino_mode=False):
        """定义超参数搜索空间"""
        if arduino_mode:
            # Arduino模式：使用固定、可复现且轻量的模型复杂度，保证 standard/loso 一致
            return {
                'n_estimators': trial.suggest_categorical('n_estimators', [120]),
                'num_leaves': trial.suggest_categorical('num_leaves', [31]),
                'max_depth': trial.suggest_categorical('max_depth', [8]),
                'learning_rate': trial.suggest_categorical('learning_rate', [0.10]),
                'feature_fraction': trial.suggest_categorical('feature_fraction', [0.80]),
                'bagging_fraction': trial.suggest_categorical('bagging_fraction', [0.80]),
                'bagging_freq': trial.suggest_categorical('bagging_freq', [5]),
                'min_child_samples': trial.suggest_categorical('min_child_samples', [20]),
                'reg_alpha': trial.suggest_categorical('reg_alpha', [0.10]),
                'reg_lambda': trial.suggest_categorical('reg_lambda', [0.10])
            }
        else:
            # 完整参数空间
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_split_gain': trial.suggest_float('min_split_gain', 0, 1),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }
    
    def create_model(self, params, arduino_mode=False, callbacks=None):
        """创建LightGBM模型"""
        lgb_params = {
            'objective': 'multiclass',
            'num_class': N_CLASSES,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': params.get('n_estimators', 200),
            'num_leaves': params.get('num_leaves', 31),
            'learning_rate': params.get('learning_rate', 0.1),
            'feature_fraction': params.get('feature_fraction', 0.8),
            'bagging_fraction': params.get('bagging_fraction', 0.8),
            'bagging_freq': params.get('bagging_freq', 5),
            'min_child_samples': params.get('min_child_samples', 20),
            'reg_alpha': params.get('reg_alpha', 0.1),
            'reg_lambda': params.get('reg_lambda', 0.1),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        # 添加额外参数（如果存在）
        if 'max_depth' in params:
            lgb_params['max_depth'] = params['max_depth']
        if 'min_split_gain' in params:
            lgb_params['min_split_gain'] = params['min_split_gain']
        if 'subsample' in params:
            lgb_params['subsample'] = params['subsample']
        if 'colsample_bytree' in params:
            lgb_params['colsample_bytree'] = params['colsample_bytree']
        
        # 创建LightGBM模型
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        
        # 返回包装器
        return LightgbmModelWrapper(lgb_model, self, params)
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': 'LightGBM',
            'abbreviation': 'LGB',
            'description': '基于增强特征工程的LightGBM手势识别模型',
            'features': [
                '综合特征提取 (时域、频域、非线性)',
                '滑动窗口特征聚合',
                '类别平衡优化',
                'Optuna超参数优化',
                'Arduino兼容轻量级版本'
            ],
            'arduino_compatible': True,
            'feature_engineering': 'Advanced'
        }
    
    def get_augmentation_params(self, trial=None):
        """获取数据增强参数 - 兼容Optuna优化"""
        if trial is not None:
            # Optuna优化模式 - 轻度增强策略
            return {
                'augment_factor': trial.suggest_int('augment_factor', 1, 2),  # 轻度增强
                'jitter_noise_level': trial.suggest_float('jitter_noise_level', 0.003, 0.008),  # 更小的噪声范围
                'time_warp_max_speed': trial.suggest_int('time_warp_max_speed', 1, 3),  # 保护时序特征
                'scale_range': [
                    trial.suggest_float('scale_min', 0.97, 0.99),
                    trial.suggest_float('scale_max', 1.01, 1.03)
                ],
                'augment_prob': trial.suggest_float('augment_prob', 0.2, 0.4)  # 降低增强概率
            }
        else:
            # 默认轻度增强配置
            return {
                'augment_factor': 1,
                'jitter_noise_level': 0.005,
                'time_warp_max_speed': 2,
                'scale_range': [0.98, 1.02],
                'augment_prob': 0.3
            }