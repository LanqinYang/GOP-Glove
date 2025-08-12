"""
BSL Gesture Recognition - XGBoost Model Definition
"""
import numpy as np
import xgboost as xgb
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

# Constants
N_FEATURES = 5
N_CLASSES = 11

class XgboostModelCreator:
    def __init__(self):
        pass

    def _extract_features(self, X, arduino_mode):
        features = []
        for sample in X:
            sample_features = []
            for channel in range(N_FEATURES):
                data = sample[:, channel]
                if arduino_mode:
                    # 极简特征以缩小模型（每通道2维：均值、标准差）
                    sample_features.extend([np.mean(data), np.std(data)])
                else:
                    sample_features.extend([
                        np.mean(data), np.std(data), np.min(data), np.max(data),
                        np.median(data), skew(data), kurtosis(data), np.var(data)
                    ])
                    window_size = 10
                    if len(data) >= window_size:
                        rolling_means = [np.mean(data[i:i+window_size]) for i in range(len(data) - window_size + 1)]
                        sample_features.extend([np.mean(rolling_means), np.std(rolling_means)])
                    else:
                        sample_features.extend([0, 0])
            features.append(sample_features)
        return np.array(features)

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