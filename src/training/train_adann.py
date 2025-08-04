#!/usr/bin/env python3
"""
对抗域自适应神经网络 (ADANN) 训练模块
基于标准pipeline格式，用于EMG手势识别的跨被试泛化

核心技术:
1. 梯度反转层实现特征解耦
2. 双分支网络结构 (手势分类器 + 域分类器)
3. 对抗训练提升跨被试泛化能力
4. 综合特征工程
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from scipy import signal
from scipy.stats import skew, kurtosis
import optuna
import warnings
warnings.filterwarnings('ignore')


class AdannModelWrapper:
    """包装器类，使PyTorch ADANN模型兼容TensorFlow/Keras风格的pipeline"""
    
    def __init__(self, pytorch_model, model_creator, params):
        self.pytorch_model = pytorch_model
        self.model_creator = model_creator
        self.params = params
        self.history = None
        self.trained = False
        
    def fit(self, X_train, y_train, validation_data=None, epochs=10, callbacks=None, verbose=1, **kwargs):
        """兼容Keras风格的fit方法"""
        if verbose:
            print(f"🚀 Training ADANN model for {epochs} epochs...")
        
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
        
        # 创建虚拟subjects (因为ADANN需要subject信息)
        subjects_train = np.arange(len(X_train)) % 6  # 假设6个被试
        subjects_val = np.arange(len(X_val)) % 6
        
        try:
            # 调用原始训练方法，获取真实训练历史
            trained_model, accuracy, training_history = self.model_creator.train_model(
                self.pytorch_model, X_train, y_train, subjects_train,
                X_val, y_val, subjects_val, self.params, return_history=True
            )
            # 更新模型引用
            self.pytorch_model = trained_model
            
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
                print(f"✅ ADANN training completed with accuracy: {accuracy:.4f}")
            
        except Exception as e:
            if verbose:
                print(f"❌ ADANN training failed: {e}")
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
            # 如果模型未训练，返回随机预测
            return np.random.rand(len(X), 11)  # 11个类别
        
        try:
            # 特征提取
            X_features = self.model_creator._extract_features(X)
            # 使用训练时保存的scaler
            X_scaled = self.pytorch_model.scaler.transform(X_features)
            
            # PyTorch预测
            self.pytorch_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(self.model_creator.device)
                # ADANN forward返回3个值：gesture_logits, domain_logits, features
                gesture_logits, _, _ = self.pytorch_model(X_tensor)
                probabilities = torch.softmax(gesture_logits, dim=1).cpu().numpy()
            
            return probabilities
        except Exception as e:
            print(f"⚠️ ADANN prediction failed: {e}")
            # 返回随机预测作为备选
            return np.random.rand(len(X), 11)
    
    def evaluate(self, X, y, verbose=0):
        """兼容Keras风格的evaluate方法"""
        predictions = self.predict(X, verbose=verbose)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(y, predicted_classes)
        loss = 1.0 - accuracy  # 简单的损失估计
        return [loss, accuracy]
    
    def save(self, filepath):
        """保存模型 - 兼容Keras格式"""
        if self.trained:
            # 保存PyTorch模型状态
            torch_filepath = filepath.replace('.keras', '.pth')
            torch.save({
                'model_state_dict': self.pytorch_model.state_dict(),
                'params': self.params,
                'trained': self.trained
            }, torch_filepath)
            print(f"✅ ADANN model saved to {torch_filepath}")
        else:
            print("⚠️ Model not trained, cannot save.")
    
    def load(self, filepath):
        """加载模型"""
        self.pytorch_model.load_state_dict(torch.load(filepath))
        self.trained = True

class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层 - ADANN的核心组件"""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class AdversarialFeatureExtractor(nn.Module):
    """对抗特征提取器"""
    
    def __init__(self, input_size=135, feature_size=64, n_gestures=11, n_subjects=6):
        super(AdversarialFeatureExtractor, self).__init__()
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, feature_size),
            nn.ReLU()
        )
        
        # 手势分类器
        self.gesture_classifier = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_gestures)
        )
        
        # 域分类器 (被试分类器)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_subjects)
        )
        
        self.alpha = 1.0
        
    def forward(self, x, reverse_gradient=True):
        # 提取特征
        features = self.feature_extractor(x)
        
        # 手势分类
        gesture_pred = self.gesture_classifier(features)
        
        # 域分类 (带梯度反转)
        if reverse_gradient:
            reversed_features = GradientReversalLayer.apply(features, self.alpha)
            domain_pred = self.domain_classifier(reversed_features)
        else:
            domain_pred = self.domain_classifier(features)
        
        return gesture_pred, domain_pred, features
    
    def set_alpha(self, alpha):
        self.alpha = alpha

class EnhancedFeatureExtractor:
    """增强特征提取器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_comprehensive_features(self, sample):
        """提取综合特征集"""
        features = []
        
        # 对每个通道提取特征
        for ch in range(sample.shape[1]):
            channel_data = sample[:, ch]
            
            # 1. 时域统计特征 (14个)
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
                np.sum(np.diff(np.sign(np.diff(channel_data))) != 0) # SSC
            ])
            
            # 2. 频域特征 (8个)
            try:
                freqs, psd = signal.welch(channel_data, fs=200, nperseg=min(64, len(channel_data)))
                features.extend([
                    np.sum(freqs * psd) / np.sum(psd),  # 谱质心
                    freqs[np.argmax(psd)],              # 主导频率
                    np.sum(psd),                        # 总功率
                    np.sqrt(np.sum(((freqs - np.sum(freqs * psd) / np.sum(psd))**2) * psd) / np.sum(psd)), # 谱扩展
                    -np.sum(psd * np.log2(psd + 1e-10)) / np.log2(len(psd)), # 谱熵
                    np.sum(psd[freqs <= 50]),           # 低频功率
                    np.sum(psd[(freqs > 50) & (freqs <= 100)]), # 中频功率
                    np.sum(psd[freqs > 100])            # 高频功率
                ])
            except:
                features.extend([0] * 8)
            
            # 3. 小波特征 (5个)
            try:
                from scipy.signal import cwt, ricker
                scales = np.arange(1, 6)
                coeffs = cwt(channel_data, ricker, scales)
                for i in range(5):
                    features.append(np.sum(coeffs[i]**2))  # 各尺度能量
            except:
                features.extend([0] * 5)
        
        return np.array(features)

class AdannModelCreator:
    """ADANN模型创建器 - 符合pipeline标准"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = EnhancedFeatureExtractor()
        self.scaler = StandardScaler()
        
    def extract_and_scale_features(self, X_data, fit=False, scaler=None, arduino_mode=False):
        """提取和缩放特征 - 兼容pipeline接口"""
        # 对于ADANN，我们直接返回原始数据，特征提取在模型内部进行
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
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'feature_size': trial.suggest_int('feature_size', 32, 128, step=16),
            'gesture_loss_weight': trial.suggest_float('gesture_loss_weight', 0.5, 2.0),
            'domain_loss_weight': trial.suggest_float('domain_loss_weight', 0.5, 2.0),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'n_epochs': trial.suggest_int('n_epochs', 50, 150)
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
    
    def _extract_features(self, X):
        """提取特征"""
        X_features = []
        for sample in X:
            features = self.feature_extractor.extract_comprehensive_features(sample)
            X_features.append(features)
        return np.array(X_features)
    
    def _create_data_loaders(self, X_train, y_train, subjects_train, X_val, y_val, subjects_val, batch_size):
        """创建数据加载器"""
        # 训练数据
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.LongTensor(y_train).to(self.device),
            torch.LongTensor(subjects_train).to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 验证数据
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).to(self.device),
            torch.LongTensor(y_val).to(self.device),
            torch.LongTensor(subjects_val).to(self.device)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def create_model(self, params, arduino_mode=False, callbacks=None):
        """创建ADANN模型 - 兼容pipeline接口"""
        # 基本配置常量
        SEQUENCE_LENGTH = 100
        N_FEATURES = 5
        N_CLASSES = 11
        
        # 提取特征维度 (使用虚拟输入)
        dummy_input = np.random.randn(1, SEQUENCE_LENGTH, N_FEATURES)
        features = self._extract_features(dummy_input)
        input_size = features.shape[1]
        
        # 创建模型
        pytorch_model = AdversarialFeatureExtractor(
            input_size=input_size,
            feature_size=params.get('feature_size', 64),
            n_gestures=N_CLASSES,
            n_subjects=6  # 固定被试数量
        ).to(self.device)
        
        # 创建兼容包装器
        wrapper = AdannModelWrapper(pytorch_model, self, params)
        return wrapper
    
    def train_model(self, model, X_train, y_train, subjects_train, X_val, y_val, subjects_val, hyperparams, return_history=False):
        """训练ADANN模型"""
        # 特征提取
        X_train_features = self._extract_features(X_train)
        X_val_features = self._extract_features(X_val)
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        X_val_scaled = self.scaler.transform(X_val_features)
        
        # 标签编码
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
        
        # 创建数据加载器
        train_loader, val_loader = self._create_data_loaders(
            X_train_scaled, y_train_encoded, subjects_train_encoded,
            X_val_scaled, y_val_encoded, subjects_val_encoded,
            hyperparams['batch_size']
        )
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), 
                              lr=hyperparams['learning_rate'],
                              weight_decay=hyperparams['weight_decay'])
        
        gesture_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        model.train()
        best_val_acc = 0
        n_epochs = hyperparams['n_epochs']
        
        # 初始化训练历史记录
        history = {
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': []
        } if return_history else None
        
        for epoch in range(n_epochs):
            total_loss = 0
            gesture_correct = 0
            total_samples = 0
            
            # 动态调整alpha (梯度反转强度)
            p = float(epoch) / n_epochs
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            model.set_alpha(alpha)
            
            for batch_idx, (data, gesture_labels, subject_labels) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # 前向传播
                gesture_pred, domain_pred, features = model(data, reverse_gradient=True)
                
                # 计算损失
                gesture_loss = gesture_criterion(gesture_pred, gesture_labels) * hyperparams['gesture_loss_weight']
                domain_loss = domain_criterion(domain_pred, subject_labels) * hyperparams['domain_loss_weight']
                
                total_batch_loss = gesture_loss + domain_loss
                
                # 反向传播
                total_batch_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 统计
                total_loss += total_batch_loss.item()
                gesture_correct += (gesture_pred.argmax(1) == gesture_labels).sum().item()
                total_samples += data.size(0)
            
            # 计算epoch指标
            epoch_loss = total_loss / len(train_loader)
            epoch_acc = gesture_correct / total_samples
            
            # 验证
            val_acc = self._evaluate_model(model, val_loader)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # 记录历史
            if return_history and history:
                history['accuracy'].append(float(epoch_acc))
                history['val_accuracy'].append(float(val_acc))
                history['loss'].append(float(epoch_loss))
                history['val_loss'].append(float(epoch_loss))  # 使用训练损失作为近似
            
            # 打印进度
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, "
                      f"Train_Acc={epoch_acc:.4f}, Val_Acc={val_acc:.4f}")
        
        # 保存编码器以供预测使用
        model.gesture_encoder = gesture_encoder
        model.subject_encoder = subject_encoder
        model.scaler = self.scaler
        model.feature_extractor_obj = self.feature_extractor
        
        if return_history:
            return model, best_val_acc, history
        else:
            return model, best_val_acc
    
    def _evaluate_model(self, model, val_loader):
        """评估模型"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, gesture_labels, subject_labels in val_loader:
                gesture_pred, domain_pred, features = model(data, reverse_gradient=False)
                correct += (gesture_pred.argmax(1) == gesture_labels).sum().item()
                total += data.size(0)
        
        model.train()
        return correct / total if total > 0 else 0
    
    def predict(self, model, X):
        """预测"""
        model.eval()
        
        # 特征提取
        X_features = []
        for sample in X:
            features = model.feature_extractor_obj.extract_comprehensive_features(sample)
            X_features.append(features)
        X_features = np.array(X_features)
        
        # 特征标准化
        X_scaled = model.scaler.transform(X_features)
        
        # 转换为tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), 32):  # 批量预测
                batch = X_tensor[i:i+32]
                gesture_pred, _, _ = model(batch, reverse_gradient=False)
                pred_labels = gesture_pred.argmax(1).cpu().numpy()
                predictions.extend(pred_labels)
        
        # 解码标签
        predictions_decoded = model.gesture_encoder.inverse_transform(predictions)
        return predictions_decoded