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
        """提取综合特征集"""
        features = []
        
        # 对每个通道提取特征
        for ch in range(sample.shape[1]):
            channel_data = sample[:, ch]
            
            # 1. 时域统计特征 (14个) - 修复NaN问题
            # 安全的统计特征计算
            try:
                ch_skew = skew(channel_data) if len(channel_data) > 2 else 0
                ch_skew = 0 if np.isnan(ch_skew) or np.isinf(ch_skew) else ch_skew
            except:
                ch_skew = 0
            
            try:
                ch_kurtosis = kurtosis(channel_data) if len(channel_data) > 2 else 0
                ch_kurtosis = 0 if np.isnan(ch_kurtosis) or np.isinf(ch_kurtosis) else ch_kurtosis
            except:
                ch_kurtosis = 0
            
            # 安全的差分计算
            if len(channel_data) > 1:
                mean_abs_diff = np.mean(np.abs(np.diff(channel_data)))
                zero_crossings = len(np.where(np.diff(np.sign(channel_data)))[0])
            else:
                mean_abs_diff = 0
                zero_crossings = 0
            
            features.extend([
                np.mean(channel_data),           # 均值
                np.std(channel_data),            # 标准差
                np.var(channel_data),            # 方差
                np.min(channel_data),            # 最小值
                np.max(channel_data),            # 最大值
                np.median(channel_data),         # 中位数
                np.percentile(channel_data, 25), # 25%分位数
                np.percentile(channel_data, 75), # 75%分位数
                ch_skew,                         # 偏度
                ch_kurtosis,                     # 峰度
                np.sum(np.abs(channel_data)),    # 绝对值和
                np.sqrt(np.mean(channel_data**2)), # RMS
                mean_abs_diff,                   # 平均绝对差分
                zero_crossings                   # 过零率
            ])
            
            # 2. 频域特征 (6个) - 修复NaN问题
            try:
                freqs, psd = signal.periodogram(channel_data, fs=250)  # 假设250Hz采样率
                total_power = np.sum(psd)
                
                if total_power > 1e-10:  # 避免除零
                    dominant_freq = freqs[np.argmax(psd)]
                    spectral_centroid = np.sum(freqs * psd) / total_power
                    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_power)
                    low_freq_ratio = np.sum(psd[freqs <= 50]) / total_power
                    high_freq_ratio = np.sum(psd[freqs >= 50]) / total_power
                else:
                    dominant_freq = 0
                    spectral_centroid = 0
                    spectral_spread = 0
                    low_freq_ratio = 0
                    high_freq_ratio = 0
                
                features.extend([
                    total_power,           # 总功率
                    dominant_freq,         # 主频
                    spectral_centroid,     # 质心频率
                    spectral_spread,       # 频谱扩散
                    low_freq_ratio,        # 低频能量比
                    high_freq_ratio        # 高频能量比
                ])
            except Exception as e:
                # 如果频域分析失败，使用默认值
                features.extend([0, 0, 0, 0, 0.5, 0.5])
            
            # 3. 非线性特征 (4个) - 修复NaN问题
            # Hjorth参数
            diff1 = np.diff(channel_data)
            diff2 = np.diff(diff1)
            var_zero = np.var(channel_data)
            var_d1 = np.var(diff1)
            var_d2 = np.var(diff2)
            
            # 安全的Hjorth参数计算
            activity = var_zero
            
            if var_zero > 1e-10:
                mobility = np.sqrt(var_d1 / var_zero)
            else:
                mobility = 0
            
            if var_d1 > 1e-10 and mobility > 1e-10:
                complexity = np.sqrt(var_d2 / var_d1) / mobility
            else:
                complexity = 0
            
            # 平均变化率
            if len(channel_data) > 1:
                mean_change_rate = np.mean(np.abs(channel_data[1:] - channel_data[:-1]))
            else:
                mean_change_rate = 0
            
            features.extend([
                activity,         # 活动性
                mobility,         # 机动性
                complexity,       # 复杂性
                mean_change_rate  # 平均变化率
            ])
        
        # 最终的NaN和Inf检查
        features = np.array(features)
        
        # 替换NaN和Inf值
        nan_mask = np.isnan(features) | np.isinf(features)
        if np.any(nan_mask):
            features[nan_mask] = 0
        
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
        """Arduino优化的轻量级特征提取"""
        features = []
        for ch in range(sample.shape[1]):
            channel_data = sample[:, ch]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.median(channel_data),
                skew(channel_data),
                kurtosis(channel_data)
            ])
        return np.array(features)
    
    def define_hyperparams(self, trial, arduino_mode=False):
        """定义超参数搜索空间"""
        if arduino_mode:
            # Arduino优化参数
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'num_leaves': trial.suggest_int('num_leaves', 10, 50),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5)
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