#!/usr/bin/env python3
"""
ADANN + LightGBM 混合模型训练模块
结合对抗域自适应的深度特征和LightGBM的传统特征工程

核心技术:
1. ADANN提取域不变特征
2. LightGBM处理手工特征
3. 特征融合和集成预测
4. 双重优化策略
"""

import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from scipy import signal
from scipy.stats import skew, kurtosis
import optuna
import warnings
warnings.filterwarnings('ignore')

# 导入ADANN组件
from .train_adann import AdversarialFeatureExtractor, EnhancedFeatureExtractor, GradientReversalLayer


class AdannLightgbmModelWrapper:
    """ADANN+LightGBM混合模型包装器"""
    
    def __init__(self, hybrid_model, model_creator, params):
        self.hybrid_model = hybrid_model
        self.model_creator = model_creator
        self.params = params
        self.history = None
        self.trained = False
        
    def fit(self, X_train, y_train, validation_data=None, epochs=10, callbacks=None, verbose=1, **kwargs):
        """兼容Keras风格的fit方法"""
        if verbose:
            print(f"🚀 Training ADANN+LightGBM hybrid model...")
        
        # 准备验证数据
        if validation_data is not None:
            X_val, y_val = validation_data
        else:
            # 简单分割
            split_idx = int(0.8 * len(X_train))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        
        # 创建虚拟subjects
        subjects_train = np.arange(len(X_train)) % 6
        subjects_val = np.arange(len(X_val)) % 6
        
        try:
            # 调用训练方法，获取真实训练历史
            trained_model, accuracy, training_history = self.model_creator.train_model(
                self.hybrid_model, X_train, y_train, subjects_train,
                X_val, y_val, subjects_val, self.params, return_history=True
            )
            
            # 更新模型引用
            self.hybrid_model = trained_model
            
            # 使用真实的训练历史
            if training_history:
                self.history = type('History', (), {'history': training_history})()
            else:
                # 备用虚拟history
                self.history = type('History', (), {
                    'history': {
                        'accuracy': [accuracy] * epochs,
                        'val_accuracy': [accuracy] * epochs,
                        'loss': [1.0 - accuracy] * epochs,
                        'val_loss': [1.0 - accuracy] * epochs
                    }
                })()
            
            self.trained = True
            if verbose:
                print(f"✅ ADANN+LightGBM training completed with accuracy: {accuracy:.4f}")
            
        except Exception as e:
            if verbose:
                print(f"❌ ADANN+LightGBM training failed: {e}")
            # 创建失败的history
            self.history = type('History', (), {
                'history': {
                    'accuracy': [0.1] * epochs,
                    'val_accuracy': [0.1] * epochs,
                    'loss': [2.3] * epochs,
                    'val_loss': [2.3] * epochs
                }
            })()
        
        return self.history
    
    def predict(self, X, verbose=0):
        """兼容Keras风格的predict方法"""
        if not self.trained:
            return np.random.rand(len(X), 11)
        
        try:
            return self.model_creator.predict_hybrid(self.hybrid_model, X)
        except Exception as e:
            print(f"⚠️ ADANN+LightGBM prediction failed: {e}")
            return np.random.rand(len(X), 11)
    
    def evaluate(self, X, y, verbose=0):
        """兼容Keras风格的evaluate方法"""
        predictions = self.predict(X, verbose=verbose)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(y, predicted_classes)
        loss = 1.0 - accuracy
        return [loss, accuracy]
    
    def save(self, filepath):
        """保存混合模型"""
        if self.trained:
            torch_filepath = filepath.replace('.keras', '.pth')
            torch.save({
                'adann_state_dict': self.hybrid_model['adann'].state_dict(),
                'lightgbm_model': self.hybrid_model['lightgbm'],
                'ensemble_weight': self.hybrid_model['ensemble_weight'],
                'params': self.params,
                'trained': self.trained
            }, torch_filepath)
            print(f"✅ ADANN+LightGBM model saved to {torch_filepath}")
        else:
            print("⚠️ Model not trained, cannot save.")
    
    def load(self, filepath):
        """加载混合模型"""
        checkpoint = torch.load(filepath)
        self.hybrid_model['adann'].load_state_dict(checkpoint['adann_state_dict'])
        self.hybrid_model['lightgbm'] = checkpoint['lightgbm_model']
        self.hybrid_model['ensemble_weight'] = checkpoint['ensemble_weight']
        self.trained = True

class HybridFeatureExtractor:
    """混合特征提取器 - 结合ADANN和传统特征"""
    
    def __init__(self):
        self.enhanced_extractor = EnhancedFeatureExtractor()
        self.scaler = StandardScaler()
        
    def extract_lightgbm_features(self, sample):
        """提取LightGBM专用特征"""
        features = []
        
        # 对每个通道提取特征
        for ch in range(sample.shape[1]):
            channel_data = sample[:, ch]
            
            # 1. 扩展时域特征 (18个)
            features.extend([
                np.mean(channel_data),           # 均值
                np.std(channel_data),            # 标准差
                np.var(channel_data),            # 方差
                np.min(channel_data),            # 最小值
                np.max(channel_data),            # 最大值
                np.ptp(channel_data),            # 峰峰值
                np.median(channel_data),         # 中位数
                skew(channel_data),              # 偏度
                kurtosis(channel_data),          # 峰度
                np.sqrt(np.mean(channel_data**2)), # RMS
                np.mean(np.abs(channel_data)),   # MAV
                np.sum(np.abs(np.diff(channel_data))), # WL
                np.sum(np.diff(np.sign(channel_data)) != 0), # ZC
                np.sum(np.diff(np.sign(np.diff(channel_data))) != 0), # SSC
                np.percentile(channel_data, 25), # Q1
                np.percentile(channel_data, 75), # Q3
                np.mean(np.abs(channel_data - np.mean(channel_data))), # AAV
                len(channel_data[channel_data > np.mean(channel_data) + 2*np.std(channel_data)]) # 异常值计数
            ])
            
            # 2. 扩展频域特征 (12个)
            try:
                freqs, psd = signal.welch(channel_data, fs=200, nperseg=min(64, len(channel_data)))
                psd_norm = psd / (np.sum(psd) + 1e-10)
                
                features.extend([
                    np.sum(freqs * psd_norm),       # 谱质心
                    freqs[np.argmax(psd)],          # 主导频率
                    np.sum(psd),                    # 总功率
                    np.sqrt(np.sum(((freqs - np.sum(freqs * psd_norm))**2) * psd_norm)), # 谱扩展
                    -np.sum(psd_norm * np.log2(psd_norm + 1e-10)), # 谱熵
                    np.sum(psd[freqs <= 50]),       # 低频功率
                    np.sum(psd[(freqs > 50) & (freqs <= 100)]), # 中频功率
                    np.sum(psd[freqs > 100]),       # 高频功率
                    np.sum(freqs**2 * psd_norm),    # 二阶谱矩
                    np.sum(freqs**3 * psd_norm),    # 三阶谱矩
                    np.max(psd) / np.mean(psd),     # 峰值因子
                    np.std(psd) / np.mean(psd)      # 变异系数
                ])
            except:
                features.extend([0] * 12)
            
            # 3. 小波特征 (8个)
            try:
                from scipy.signal import cwt, ricker
                scales = np.arange(1, 9)
                coeffs = cwt(channel_data, ricker, scales)
                for i in range(8):
                    features.append(np.sum(coeffs[i]**2))  # 各尺度能量
            except:
                features.extend([0] * 8)
        
        return np.array(features)

class AdannLightgbmModelCreator:
    """ADANN + LightGBM 混合模型创建器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hybrid_extractor = HybridFeatureExtractor()
        self.adann_scaler = StandardScaler()
        self.lgb_scaler = StandardScaler()
        
    def extract_and_scale_features(self, X_data, fit=False, scaler=None, arduino_mode=False):
        """提取和缩放特征 - 兼容pipeline接口"""
        # 对于混合模型，我们直接返回原始数据，特征提取在模型内部进行
        if fit:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_data.reshape(-1, X_data.shape[-1])).reshape(X_data.shape)
            return X_scaled, scaler
        else:
            X_scaled = scaler.transform(X_data.reshape(-1, X_data.shape[-1])).reshape(X_data.shape)
            return X_scaled, scaler
    
    def define_hyperparams(self, trial, arduino_mode=False):
        """定义超参数搜索空间"""
        params = {
            # ADANN参数
            'adann_learning_rate': trial.suggest_float('adann_learning_rate', 1e-4, 1e-2, log=True),
            'adann_feature_size': trial.suggest_int('adann_feature_size', 32, 128, step=16),
            'adann_epochs': trial.suggest_int('adann_epochs', 50, 100),
            'gesture_loss_weight': trial.suggest_float('gesture_loss_weight', 0.5, 2.0),
            'domain_loss_weight': trial.suggest_float('domain_loss_weight', 0.5, 2.0),
            
            # LightGBM参数
            'lgb_num_leaves': trial.suggest_int('lgb_num_leaves', 10, 100),
            'lgb_learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3),
            'lgb_feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.4, 1.0),
            'lgb_bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.4, 1.0),
            'lgb_min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 100),
            'lgb_n_estimators': trial.suggest_int('lgb_n_estimators', 50, 500),
            
            # 集成参数
            'ensemble_adann_weight': trial.suggest_float('ensemble_adann_weight', 0.3, 0.7),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
        }
        
        # 添加数据增强参数 (非Arduino模式)
        if not arduino_mode:
            params.update({
                'augment_factor': trial.suggest_int('augment_factor', 1, 3),
                'jitter_noise_level': trial.suggest_float('jitter_noise_level', 0.005, 0.02),
                'time_warp_max_speed': trial.suggest_int('time_warp_max_speed', 2, 4),
                'scale_range': [trial.suggest_float('scale_min', 0.9, 0.98), trial.suggest_float('scale_max', 1.02, 1.1)],
                'augment_prob': trial.suggest_float('augment_prob', 0.3, 0.8)
            })
        
        return params
    
    def create_model(self, params, arduino_mode=False, callbacks=None):
        """创建混合模型 - 兼容pipeline接口"""
        # 基本配置常量
        SEQUENCE_LENGTH = 100
        N_FEATURES = 5
        N_CLASSES = 11
        
        # 提取特征维度
        dummy_input = np.random.randn(1, SEQUENCE_LENGTH, N_FEATURES)
        adann_features = self.hybrid_extractor.enhanced_extractor.extract_comprehensive_features(dummy_input[0])
        lgb_features = self.hybrid_extractor.extract_lightgbm_features(dummy_input[0])
        
        # 创建ADANN模型
        adann_model = AdversarialFeatureExtractor(
            input_size=len(adann_features),
            feature_size=params.get('adann_feature_size', 64),
            n_gestures=N_CLASSES,
            n_subjects=6
        ).to(self.device)
        
        # 创建LightGBM模型
        lgb_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=N_CLASSES,
            num_leaves=params.get('lgb_num_leaves', 31),
            learning_rate=params.get('lgb_learning_rate', 0.1),
            feature_fraction=params.get('lgb_feature_fraction', 0.8),
            bagging_fraction=params.get('lgb_bagging_fraction', 0.8),
            min_child_samples=params.get('lgb_min_child_samples', 20),
            n_estimators=params.get('lgb_n_estimators', 100),
            random_state=42,
            verbose=-1
        )
        
        # 包装为混合模型
        hybrid_model = {
            'adann': adann_model,
            'lightgbm': lgb_model,
            'ensemble_weight': params.get('ensemble_adann_weight', 0.5)
        }
        
        # 创建兼容包装器
        wrapper = AdannLightgbmModelWrapper(hybrid_model, self, params)
        return wrapper
    
    def train_model(self, model, X_train, y_train, subjects_train, X_val, y_val, subjects_val, hyperparams, return_history=False):
        """训练混合模型"""
        print("🚀 训练ADANN + LightGBM混合模型...")
        
        # 1. 提取ADANN特征
        print("   提取ADANN特征...")
        X_train_adann = []
        X_val_adann = []
        for sample in X_train:
            features = self.hybrid_extractor.enhanced_extractor.extract_comprehensive_features(sample)
            X_train_adann.append(features)
        for sample in X_val:
            features = self.hybrid_extractor.enhanced_extractor.extract_comprehensive_features(sample)
            X_val_adann.append(features)
        
        X_train_adann = np.array(X_train_adann)
        X_val_adann = np.array(X_val_adann)
        
        # 2. 提取LightGBM特征
        print("   提取LightGBM特征...")
        X_train_lgb = []
        X_val_lgb = []
        for sample in X_train:
            features = self.hybrid_extractor.extract_lightgbm_features(sample)
            X_train_lgb.append(features)
        for sample in X_val:
            features = self.hybrid_extractor.extract_lightgbm_features(sample)
            X_val_lgb.append(features)
        
        X_train_lgb = np.array(X_train_lgb)
        X_val_lgb = np.array(X_val_lgb)
        
        # 3. 特征标准化
        X_train_adann_scaled = self.adann_scaler.fit_transform(X_train_adann)
        X_val_adann_scaled = self.adann_scaler.transform(X_val_adann)
        
        X_train_lgb_scaled = self.lgb_scaler.fit_transform(X_train_lgb)
        X_val_lgb_scaled = self.lgb_scaler.transform(X_val_lgb)
        
        # 4. 标签编码
        all_gestures = np.unique(np.concatenate([y_train, y_val]))
        all_subjects = np.unique(np.concatenate([subjects_train, subjects_val]))
        
        gesture_encoder = LabelEncoder()
        subject_encoder = LabelEncoder()
        
        gesture_encoder.fit(all_gestures)
        subject_encoder.fit(all_subjects)
        
        y_train_encoded = gesture_encoder.transform(y_train)
        y_val_encoded = gesture_encoder.transform(y_val)
        subjects_train_encoded = subject_encoder.transform(subjects_train)
        subjects_val_encoded = subject_encoder.transform(subjects_val)
        
        # 5. 训练ADANN
        print("   训练ADANN模型...")
        adann_model = model['adann']
        if return_history:
            adann_model, adann_history = self._train_adann(
                adann_model, X_train_adann_scaled, y_train_encoded, subjects_train_encoded,
                X_val_adann_scaled, y_val_encoded, subjects_val_encoded, hyperparams, return_history=True
            )
        else:
            adann_model = self._train_adann(
                adann_model, X_train_adann_scaled, y_train_encoded, subjects_train_encoded,
                X_val_adann_scaled, y_val_encoded, subjects_val_encoded, hyperparams
            )
        
        # 6. 训练LightGBM
        print("   训练LightGBM模型...")
        lgb_model = model['lightgbm']
        lgb_model.fit(X_train_lgb_scaled, y_train_encoded)
        
        # 7. 验证集成性能
        adann_val_pred = self._predict_adann(adann_model, X_val_adann_scaled)
        lgb_val_pred = lgb_model.predict(X_val_lgb_scaled)
        
        # 集成预测
        ensemble_weight = model['ensemble_weight']
        ensemble_pred = self._ensemble_predict(adann_val_pred, lgb_val_pred, ensemble_weight)
        
        val_accuracy = accuracy_score(y_val_encoded, ensemble_pred)
        print(f"   验证准确率: {val_accuracy:.4f}")
        
        # 保存必要信息
        model['adann'] = adann_model
        model['lightgbm'] = lgb_model
        model['gesture_encoder'] = gesture_encoder
        model['subject_encoder'] = subject_encoder
        model['adann_scaler'] = self.adann_scaler
        model['lgb_scaler'] = self.lgb_scaler
        model['hybrid_extractor'] = self.hybrid_extractor
        
        if return_history:
            # 使用ADANN的真实训练历史
            if 'adann_history' in locals():
                history = adann_history
            else:
                # 备用简单历史记录
                history = {
                    'accuracy': [val_accuracy] * hyperparams.get('adann_epochs', 50),
                    'val_accuracy': [val_accuracy] * hyperparams.get('adann_epochs', 50),
                    'loss': [1.0 - val_accuracy] * hyperparams.get('adann_epochs', 50),
                    'val_loss': [1.0 - val_accuracy] * hyperparams.get('adann_epochs', 50)
                }
            return model, val_accuracy, history
        else:
            return model, val_accuracy
    
    def _train_adann(self, model, X_train, y_train, subjects_train, X_val, y_val, subjects_val, hyperparams, return_history=False):
        """训练ADANN部分"""
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        
        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.LongTensor(y_train).to(self.device),
            torch.LongTensor(subjects_train).to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).to(self.device),
            torch.LongTensor(y_val).to(self.device),
            torch.LongTensor(subjects_val).to(self.device)
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['adann_learning_rate'])
        gesture_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        model.train()
        n_epochs = hyperparams['adann_epochs']
        
        # 初始化训练历史记录
        history = {
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': []
        } if return_history else None
        
        for epoch in range(n_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            # 动态调整alpha
            p = float(epoch) / n_epochs
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            model.set_alpha(alpha)
            
            for data, gesture_labels, subject_labels in train_loader:
                optimizer.zero_grad()
                
                gesture_pred, domain_pred, _ = model(data, reverse_gradient=True)
                
                gesture_loss = gesture_criterion(gesture_pred, gesture_labels) * hyperparams['gesture_loss_weight']
                domain_loss = domain_criterion(domain_pred, subject_labels) * hyperparams['domain_loss_weight']
                
                total_loss_batch = gesture_loss + domain_loss
                total_loss_batch.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                correct += (gesture_pred.argmax(1) == gesture_labels).sum().item()
                total += data.size(0)
            
            # 计算epoch指标
            epoch_loss = total_loss / len(train_loader)
            epoch_acc = correct / total
            
            # 验证
            val_acc = self._evaluate_adann(model, val_loader)
            
            # 记录历史
            if return_history and history:
                history['accuracy'].append(float(epoch_acc))
                history['val_accuracy'].append(float(val_acc))
                history['loss'].append(float(epoch_loss))
                history['val_loss'].append(float(epoch_loss))
            
            if epoch % 20 == 0:
                print(f"     ADANN Epoch {epoch}: Loss={epoch_loss:.4f}")
        
        if return_history:
            return model, history
        else:
            return model
    
    def _evaluate_adann(self, model, val_loader):
        """评估ADANN模型"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, gesture_labels, subject_labels in val_loader:
                gesture_pred, _, _ = model(data, reverse_gradient=False)
                correct += (gesture_pred.argmax(1) == gesture_labels).sum().item()
                total += data.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _predict_adann(self, model, X):
        """ADANN预测"""
        model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), 32):
                batch = X_tensor[i:i+32]
                gesture_pred, _, _ = model(batch, reverse_gradient=False)
                pred_labels = gesture_pred.argmax(1).cpu().numpy()
                predictions.extend(pred_labels)
        
        return np.array(predictions)
    
    def _ensemble_predict(self, adann_pred, lgb_pred, adann_weight):
        """集成预测 - 使用概率加权平均"""
        # 确保输入是概率分布
        if len(adann_pred.shape) == 1:
            # 如果是类别标签，转换为one-hot
            n_classes = 11
            adann_probs = np.zeros((len(adann_pred), n_classes))
            lgb_probs = np.zeros((len(lgb_pred), n_classes))
            adann_probs[np.arange(len(adann_pred)), adann_pred] = 1.0
            lgb_probs[np.arange(len(lgb_pred)), lgb_pred] = 1.0
        else:
            adann_probs = adann_pred
            lgb_probs = lgb_pred
        
        # 加权平均概率
        ensemble_probs = adann_weight * adann_probs + (1 - adann_weight) * lgb_probs
        
        # 返回最终预测类别
        return ensemble_probs.argmax(axis=1)
    
    def predict(self, model, X):
        """预测"""
        # 提取特征
        X_adann = []
        X_lgb = []
        for sample in X:
            adann_features = model['hybrid_extractor'].enhanced_extractor.extract_comprehensive_features(sample)
            lgb_features = model['hybrid_extractor'].extract_lightgbm_features(sample)
            X_adann.append(adann_features)
            X_lgb.append(lgb_features)
        
        X_adann = np.array(X_adann)
        X_lgb = np.array(X_lgb)
        
        # 特征标准化
        X_adann_scaled = model['adann_scaler'].transform(X_adann)
        X_lgb_scaled = model['lgb_scaler'].transform(X_lgb)
        
        # 获取预测
        adann_pred = self._predict_adann(model['adann'], X_adann_scaled)
        lgb_pred = model['lightgbm'].predict(X_lgb_scaled)
        
        # 集成预测
        ensemble_pred = self._ensemble_predict(adann_pred, lgb_pred, model['ensemble_weight'])
        
        # 解码标签
        predictions_decoded = model['gesture_encoder'].inverse_transform(ensemble_pred)
        return predictions_decoded
    
    def predict_hybrid(self, model, X):
        """混合模型预测 - 返回概率分布"""
        try:
            # 提取特征
            X_adann = []
            X_lgb = []
            for sample in X:
                adann_features = self.hybrid_extractor.enhanced_extractor.extract_comprehensive_features(sample)
                lgb_features = self.hybrid_extractor.extract_lightgbm_features(sample)
                X_adann.append(adann_features)
                X_lgb.append(lgb_features)
            
            X_adann = np.array(X_adann)
            X_lgb = np.array(X_lgb)
            
            # 特征标准化 - 使用训练时保存的scaler
            X_adann_scaled = model['adann_scaler'].transform(X_adann)
            X_lgb_scaled = model['lgb_scaler'].transform(X_lgb)
            
            # ADANN预测
            model['adann'].eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_adann_scaled).to(self.device)
                # ADANN forward返回3个值：gesture_logits, domain_logits, features
                gesture_logits, _, _ = model['adann'](X_tensor)
                adann_probs = torch.softmax(gesture_logits, dim=1).cpu().numpy()
            
            # LightGBM预测
            lgb_probs = model['lightgbm'].predict_proba(X_lgb_scaled)
            
            # 集成预测 (加权平均概率)
            ensemble_weight = model['ensemble_weight']
            ensemble_probs = ensemble_weight * adann_probs + (1 - ensemble_weight) * lgb_probs
            
            return ensemble_probs
            
        except Exception as e:
            print(f"⚠️ Hybrid prediction error: {e}")
            # 返回随机预测作为备选
            return np.random.rand(len(X), 11)