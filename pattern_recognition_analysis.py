#!/usr/bin/env python3
"""
高级模式识别方法 - 基于gesture_patterns思路的数学优化
核心改进：
1. 多级阈值替代二值化
2. 加权相似度匹配
3. 时序模式检测
4. 自适应个性化阈值
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.stats import pearsonr
import pickle
import warnings
warnings.filterwarnings('ignore')

# 导入现有模块
import sys
sys.path.append('src/training')
from pipeline import load_data

class SlidingWindowNormalizer:
    """滑动窗口归一化器 - 基于最新文献的SWN方法"""
    
    def __init__(self, window_size=50, alpha=0.1):
        self.window_size = window_size
        self.alpha = alpha
        self.global_stats = {}
        
    def fit(self, X):
        """学习全局统计信息"""
        # X shape: (n_samples, time_steps, n_channels)
        X_flat = X.reshape(-1, X.shape[-1])
        self.global_stats['mean'] = np.mean(X_flat, axis=0)
        self.global_stats['std'] = np.std(X_flat, axis=0) + 1e-8
        return self
        
    def transform_sample(self, sample):
        """对单个样本应用SWN"""
        # sample shape: (time_steps, n_channels)
        sample_normalized = np.zeros_like(sample)
        n_timesteps, n_channels = sample.shape
        
        # 初始化局部统计
        local_mean = np.zeros(n_channels)
        local_std = np.ones(n_channels)
        
        for i in range(n_timesteps):
            # 定义滑动窗口范围
            start_idx = max(0, i - self.window_size // 2)
            end_idx = min(n_timesteps, i + self.window_size // 2 + 1)
            
            # 计算窗口内统计信息
            window_data = sample[start_idx:end_idx]
            window_mean = np.mean(window_data, axis=0)
            window_std = np.std(window_data, axis=0) + 1e-8
            
            # 指数移动平均更新
            if i == 0:
                local_mean = window_mean
                local_std = window_std
            else:
                local_mean = self.alpha * window_mean + (1 - self.alpha) * local_mean
                local_std = self.alpha * window_std + (1 - self.alpha) * local_std
            
            # 更保守的组合归一化 (20% 局部 + 80% 全局，减少过度归一化)
            local_norm = (sample[i] - local_mean) / local_std
            global_norm = (sample[i] - self.global_stats['mean']) / self.global_stats['std']
            
            sample_normalized[i] = 0.2 * local_norm + 0.8 * global_norm
            
        return sample_normalized

class AdvancedPatternRecognizer:
    """高级模式识别器 - 基于数学特征的手势识别 + SWN增强"""
    
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        self.X_raw, self.y, self.subjects = load_data(csv_dir)
        self.n_channels = 5
        self.n_classes = 11
        
        # 添加SWN归一化器 - 使用更温和的参数
        self.swn = SlidingWindowNormalizer(window_size=20, alpha=0.05)
        self.use_swn = True  # 启用保守的SWN
        
        # 基于您观察的实际激活模式（保持原有思路）
        self.gesture_patterns = {
            0: [1, 1, 0, 1, 1],  # 手势0：实际观察到的模式
            1: [1, 0, 0, 1, 1],  # 手势1
            2: [1, 0, 0, 1, 1],  # 手势2（与手势1相同，需要其他特征区分）
            3: [1, 0, 0, 1, 1],  # 手势3（与手势1,2相同）
            4: [1, 0, 0, 0, 0],  # 手势4
            5: [0, 1, 0, 1, 1],  # 手势5（动态手势）
            6: [0, 1, 1, 1, 1],  # 手势6
            7: [0, 0, 1, 1, 1],  # 手势7
            8: [0, 0, 0, 1, 1],  # 手势8
            9: [0, 0, 0, 1, 1],  # 手势9（与手势8相同）
            10: [0, 0, 0, 0, 0], # 手势10：休息状态
        }
        
        # 数学特征权重（基于您的数据分析）
        self.feature_weights = {
            'amplitude': 0.3,      # 幅度特征
            'pattern': 0.4,        # 模式匹配
            'temporal': 0.2,       # 时序特征
            'frequency': 0.1       # 频域特征
        }
        
        self.scalers = {}
        self.gesture_profiles = {}
        
    def extract_mathematical_features(self, sample, apply_swn=None):
        """提取数学特征（而非简单阈值）+ SWN增强"""
        # 应用SWN预处理 - 可选
        if apply_swn is None:
            apply_swn = self.use_swn
        
        if apply_swn and hasattr(self, 'swn') and hasattr(self.swn, 'global_stats') and self.swn.global_stats:
            sample = self.swn.transform_sample(sample)
        
        features = {}
        
        # 1. 幅度特征（替代简单的激活/未激活）
        features['rms'] = np.sqrt(np.mean(sample**2, axis=0))  # RMS值
        features['peak'] = np.max(sample, axis=0)              # 峰值
        features['mean'] = np.mean(sample, axis=0)             # 均值
        features['std'] = np.std(sample, axis=0)               # 标准差
        
        # 2. 时序特征（捕捉动态变化）
        features['slope'] = []
        features['energy_change'] = []
        for ch in range(self.n_channels):
            # 计算信号斜率（前半段vs后半段）
            mid = len(sample) // 2
            slope = np.mean(sample[mid:, ch]) - np.mean(sample[:mid, ch])
            features['slope'].append(slope)
            
            # 能量变化率
            energy_1 = np.sum(sample[:mid, ch]**2)
            energy_2 = np.sum(sample[mid:, ch]**2)
            energy_change = (energy_2 - energy_1) / (energy_1 + 1e-8)
            features['energy_change'].append(energy_change)
        
        features['slope'] = np.array(features['slope'])
        features['energy_change'] = np.array(features['energy_change'])
        
        # 3. 频域特征（捕捉肌肉激活频率特征）
        features['dominant_freq'] = []
        features['spectral_centroid'] = []
        for ch in range(self.n_channels):
            # 计算主导频率
            freqs, psd = signal.welch(sample[:, ch], fs=200, nperseg=min(64, len(sample)))
            dominant_freq = freqs[np.argmax(psd)]
            features['dominant_freq'].append(dominant_freq)
            
            # 谱质心
            spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-8)
            features['spectral_centroid'].append(spectral_centroid)
        
        features['dominant_freq'] = np.array(features['dominant_freq'])
        features['spectral_centroid'] = np.array(features['spectral_centroid'])
        
        # 4. 通道间相关性（捕捉协同激活模式）
        features['channel_corr'] = []
        for i in range(self.n_channels):
            for j in range(i+1, self.n_channels):
                corr, _ = pearsonr(sample[:, i], sample[:, j])
                features['channel_corr'].append(corr if not np.isnan(corr) else 0)
        features['channel_corr'] = np.array(features['channel_corr'])
        
        return features
    
    def create_gesture_profile(self, gesture_id, gesture_samples):
        """为每个手势创建数学特征档案"""
        print(f"  创建手势{gesture_id}的特征档案...")
        
        # 提取所有样本的特征
        all_features = []
        for sample in gesture_samples:
            features = self.extract_mathematical_features(sample)
            all_features.append(features)
        
        # 计算特征统计量
        profile = {}
        
        # 幅度特征统计
        rms_values = np.array([f['rms'] for f in all_features])
        profile['rms_mean'] = np.mean(rms_values, axis=0)
        profile['rms_std'] = np.std(rms_values, axis=0)
        
        peak_values = np.array([f['peak'] for f in all_features])
        profile['peak_mean'] = np.mean(peak_values, axis=0)
        profile['peak_std'] = np.std(peak_values, axis=0)
        
        # 时序特征统计
        slope_values = np.array([f['slope'] for f in all_features])
        profile['slope_mean'] = np.mean(slope_values, axis=0)
        profile['slope_std'] = np.std(slope_values, axis=0)
        
        energy_change_values = np.array([f['energy_change'] for f in all_features])
        profile['energy_change_mean'] = np.mean(energy_change_values, axis=0)
        profile['energy_change_std'] = np.std(energy_change_values, axis=0)
        
        # 频域特征统计
        freq_values = np.array([f['dominant_freq'] for f in all_features])
        profile['freq_mean'] = np.mean(freq_values, axis=0)
        profile['freq_std'] = np.std(freq_values, axis=0)
        
        # 相关性特征统计
        corr_values = np.array([f['channel_corr'] for f in all_features])
        profile['corr_mean'] = np.mean(corr_values, axis=0)
        profile['corr_std'] = np.std(corr_values, axis=0)
        
        # 基础激活模式（保持您的原有思路）
        profile['activation_pattern'] = np.array(self.gesture_patterns[gesture_id])
        
        return profile
    
    def fit(self, X_train, y_train):
        """训练模型（创建手势特征档案）+ SWN训练"""
        print("🧠 创建SWN增强的手势特征档案...")
        
        # 训练SWN归一化器
        print("   训练滑动窗口归一化器...")
        self.swn.fit(X_train)
        
        self.gesture_profiles = {}
        
        for gesture_id in range(self.n_classes):
            mask = y_train == gesture_id
            if np.sum(mask) == 0:
                continue
            
            gesture_samples = X_train[mask]
            profile = self.create_gesture_profile(gesture_id, gesture_samples)
            self.gesture_profiles[gesture_id] = profile
        
        print(f"✅ 创建了{len(self.gesture_profiles)}个手势档案")
        
    def calculate_similarity_score(self, sample_features, gesture_profile):
        """计算样本与手势档案的相似度分数"""
        scores = {}
        
        # 1. 幅度相似度
        rms_score = 1.0 - np.mean(np.abs(sample_features['rms'] - gesture_profile['rms_mean']) / 
                                 (gesture_profile['rms_std'] + 1e-8))
        scores['amplitude'] = max(0, rms_score)
        
        # 2. 模式相似度（基于您的激活模式思路）
        # 将RMS值归一化后与预定义模式比较
        normalized_rms = sample_features['rms'] / (np.max(sample_features['rms']) + 1e-8)
        binary_pattern = (normalized_rms > 0.5).astype(int)
        pattern_similarity = 1.0 - np.mean(np.abs(binary_pattern - gesture_profile['activation_pattern']))
        scores['pattern'] = pattern_similarity
        
        # 3. 时序相似度
        slope_score = 1.0 - np.mean(np.abs(sample_features['slope'] - gesture_profile['slope_mean']) / 
                                   (gesture_profile['slope_std'] + 1e-8))
        scores['temporal'] = max(0, slope_score)
        
        # 4. 频域相似度
        freq_score = 1.0 - np.mean(np.abs(sample_features['dominant_freq'] - gesture_profile['freq_mean']) / 
                                  (gesture_profile['freq_std'] + 1e-8))
        scores['frequency'] = max(0, freq_score)
        
        # 加权总分
        total_score = sum(scores[key] * self.feature_weights[key] for key in scores)
        
        return total_score, scores
    
    def predict_sample(self, sample):
        """预测单个样本"""
        sample_features = self.extract_mathematical_features(sample)
        
        best_gesture = -1
        best_score = -np.inf
        all_scores = {}
        
        for gesture_id, profile in self.gesture_profiles.items():
            score, detailed_scores = self.calculate_similarity_score(sample_features, profile)
            all_scores[gesture_id] = score
            
            if score > best_score:
                best_score = score
                best_gesture = gesture_id
        
        return best_gesture, best_score, all_scores
    
    def predict(self, X_test):
        """预测测试数据"""
        predictions = []
        confidence_scores = []
        
        for sample in X_test:
            pred, confidence, _ = self.predict_sample(sample)
            predictions.append(pred)
            confidence_scores.append(confidence)
        
        return np.array(predictions), np.array(confidence_scores)
    
    def evaluate_loso(self):
        """LOSO交叉验证"""
        print("🔍 运行LOSO交叉验证...")
        
        unique_subjects = np.unique(self.subjects)
        fold_results = []
        
        for fold, test_subject in enumerate(unique_subjects):
            print(f"📋 折 {fold+1}/{len(unique_subjects)}: 测试被试 {test_subject}")
            
            # 分割数据
            train_mask = self.subjects != test_subject
            test_mask = self.subjects == test_subject
            
            X_train, y_train = self.X_raw[train_mask], self.y[train_mask]
            X_test, y_test = self.X_raw[test_mask], self.y[test_mask]
            
            print(f"   训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
            
            # 训练模型
            self.fit(X_train, y_train)
            
            # 预测
            y_pred, confidence = self.predict(X_test)
            
            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            fold_results.append(accuracy)
            
            print(f"   折 {fold+1} 准确率: {accuracy:.4f}")
            print(f"   平均置信度: {np.mean(confidence):.4f}")
        
        # 总结结果
        mean_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)
        
        print(f"\n🎯 LOSO交叉验证结果:")
        print(f"   平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"   各折结果: {[f'{acc:.4f}' for acc in fold_results]}")
        
        target_gap = 0.8 - mean_acc
        if target_gap > 0:
            print(f"⚠️  距离80%目标还差 {target_gap:.4f}")
        else:
            print(f"🎉 已超越80%目标！")
        
        return fold_results
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("\n🔬 分析特征重要性...")
        
        # 使用整个数据集分析
        self.fit(self.X_raw, self.y)
        
        # 对每个手势分析特征区分度
        feature_discriminability = {}
        
        for gesture_id in self.gesture_profiles:
            profile = self.gesture_profiles[gesture_id]
            
            # 分析各类特征的标准差（越小说明越稳定，越有区分度）
            rms_stability = np.mean(1.0 / (profile['rms_std'] + 1e-8))
            slope_stability = np.mean(1.0 / (profile['slope_std'] + 1e-8))
            freq_stability = np.mean(1.0 / (profile['freq_std'] + 1e-8))
            
            feature_discriminability[gesture_id] = {
                'rms': rms_stability,
                'slope': slope_stability,
                'freq': freq_stability
            }
        
        print("各手势特征稳定性分析:")
        for gesture_id, features in feature_discriminability.items():
            print(f"  手势{gesture_id}: RMS={features['rms']:.2f}, 时序={features['slope']:.2f}, 频域={features['freq']:.2f}")
    
    def visualize_gesture_profiles(self):
        """可视化手势特征档案"""
        print("\n📊 可视化手势特征档案...")
        
        if not self.gesture_profiles:
            self.fit(self.X_raw, self.y)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. RMS特征热图
        rms_matrix = []
        gesture_labels = []
        for gesture_id in sorted(self.gesture_profiles.keys()):
            rms_matrix.append(self.gesture_profiles[gesture_id]['rms_mean'])
            gesture_labels.append(f'手势{gesture_id}')
        
        rms_matrix = np.array(rms_matrix)
        sns.heatmap(rms_matrix, annot=True, fmt='.1f', 
                   xticklabels=[f'通道{i}' for i in range(5)],
                   yticklabels=gesture_labels, ax=axes[0,0])
        axes[0,0].set_title('RMS特征档案')
        
        # 2. 激活模式热图
        pattern_matrix = []
        for gesture_id in sorted(self.gesture_profiles.keys()):
            pattern_matrix.append(self.gesture_profiles[gesture_id]['activation_pattern'])
        
        pattern_matrix = np.array(pattern_matrix)
        sns.heatmap(pattern_matrix, annot=True, fmt='d', cmap='RdYlBu_r',
                   xticklabels=[f'通道{i}' for i in range(5)],
                   yticklabels=gesture_labels, ax=axes[0,1])
        axes[0,1].set_title('激活模式档案（您的思路）')
        
        # 3. 时序特征
        slope_matrix = []
        for gesture_id in sorted(self.gesture_profiles.keys()):
            slope_matrix.append(self.gesture_profiles[gesture_id]['slope_mean'])
        
        slope_matrix = np.array(slope_matrix)
        sns.heatmap(slope_matrix, annot=True, fmt='.1f', cmap='coolwarm',
                   xticklabels=[f'通道{i}' for i in range(5)],
                   yticklabels=gesture_labels, ax=axes[1,0])
        axes[1,0].set_title('时序变化特征')
        
        # 4. 频域特征
        freq_matrix = []
        for gesture_id in sorted(self.gesture_profiles.keys()):
            freq_matrix.append(self.gesture_profiles[gesture_id]['freq_mean'])
        
        freq_matrix = np.array(freq_matrix)
        sns.heatmap(freq_matrix, annot=True, fmt='.1f', cmap='viridis',
                   xticklabels=[f'通道{i}' for i in range(5)],
                   yticklabels=gesture_labels, ax=axes[1,1])
        axes[1,1].set_title('主导频率特征')
        
        plt.tight_layout()
        plt.savefig('advanced_pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("🚀 SWN增强的高级模式识别 - 基于滑动窗口归一化的跨被试优化")
    print("="*70)
    print("核心改进:")
    print("- 保持gesture_patterns思路")
    print("- 滑动窗口归一化(SWN)增强个体适应性")
    print("- 局部-全局特征融合(70%-30%)")
    print("- 添加幅度、时序、频域特征")
    print("- 多维相似度匹配")
    print("="*70)
    
    # 初始化识别器
    recognizer = AdvancedPatternRecognizer("datasets/gesture_csv")
    
    print(f"数据加载: {len(recognizer.X_raw)} 样本, {len(np.unique(recognizer.subjects))} 被试, {recognizer.n_classes} 手势")
    
    # 分析特征重要性
    recognizer.analyze_feature_importance()
    
    # 可视化手势档案
    recognizer.visualize_gesture_profiles()
    
    # LOSO验证
    results = recognizer.evaluate_loso()
    
    # 保存模型
    with open('advanced_pattern_recognizer.pkl', 'wb') as f:
        pickle.dump(recognizer, f)
    print("💾 模型已保存: advanced_pattern_recognizer.pkl")

if __name__ == "__main__":
    main()