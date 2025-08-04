#!/usr/bin/env python3
"""
EMG数据集数学分析 - 从信号处理角度解决跨被试问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 导入数据加载函数
import sys
sys.path.append('src/training')
from pipeline import load_data

class EMGMathematicalAnalyzer:
    """EMG信号数学分析器"""
    
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        self.X, self.y, self.subjects = load_data(csv_dir)
        print(f"数据加载完成: {len(self.X)} 样本, {len(np.unique(self.subjects))} 被试, {len(np.unique(self.y))} 手势")
    
    def analyze_signal_characteristics(self):
        """分析EMG信号的数学特征"""
        print("\n🔬 EMG信号数学特征分析")
        print("=" * 50)
        
        # 1. 信号幅值分析
        amplitudes = []
        for subject in np.unique(self.subjects):
            subject_mask = self.subjects == subject
            subject_data = self.X[subject_mask]
            amp = np.mean(np.abs(subject_data))
            amplitudes.append(amp)
            print(f"被试 {subject}: 平均幅值 = {amp:.4f}")
        
        amp_cv = np.std(amplitudes) / np.mean(amplitudes)
        print(f"\n幅值变异系数 (CV): {amp_cv:.3f}")
        if amp_cv > 0.3:
            print("⚠️  被试间幅值差异巨大 - 这是跨被试识别困难的主要原因！")
        
        # 2. 频域特征分析
        print("\n📊 频域特征分析:")
        freq_features = self._analyze_frequency_domain()
        
        # 3. 信号相关性分析
        print("\n🔗 被试间信号相关性分析:")
        correlation_matrix = self._analyze_subject_correlation()
        
        return {
            'amplitude_cv': amp_cv,
            'frequency_features': freq_features,
            'correlation_matrix': correlation_matrix
        }
    
    def _analyze_frequency_domain(self):
        """频域分析"""
        fs = 100  # 假设采样率100Hz
        
        subject_spectra = {}
        for subject in np.unique(self.subjects):
            subject_mask = self.subjects == subject
            subject_data = self.X[subject_mask]
            
            # 计算平均功率谱密度
            freqs, psd = signal.welch(subject_data.reshape(-1, subject_data.shape[-1]).T, 
                                    fs=fs, nperseg=64, axis=1)
            avg_psd = np.mean(psd, axis=0)
            subject_spectra[subject] = {'freqs': freqs, 'psd': avg_psd}
            
            # 主要频率成分
            dominant_freq = freqs[np.argmax(avg_psd)]
            print(f"被试 {subject}: 主频 = {dominant_freq:.1f} Hz")
        
        return subject_spectra
    
    def _analyze_subject_correlation(self):
        """分析被试间相关性"""
        subjects_unique = np.unique(self.subjects)
        n_subjects = len(subjects_unique)
        correlation_matrix = np.zeros((n_subjects, n_subjects))
        
        # 计算每个被试的平均信号模式
        subject_patterns = {}
        for i, subject in enumerate(subjects_unique):
            subject_mask = self.subjects == subject
            subject_data = self.X[subject_mask]
            # 使用每个手势的平均模式
            pattern = []
            for gesture in np.unique(self.y):
                gesture_mask = self.y[subject_mask] == gesture
                if np.any(gesture_mask):
                    gesture_pattern = np.mean(subject_data[gesture_mask], axis=0)
                    pattern.append(gesture_pattern.flatten())
            
            if pattern:
                subject_patterns[subject] = np.concatenate(pattern)
        
        # 计算相关性矩阵
        for i, subj1 in enumerate(subjects_unique):
            for j, subj2 in enumerate(subjects_unique):
                if subj1 in subject_patterns and subj2 in subject_patterns:
                    corr = np.corrcoef(subject_patterns[subj1], subject_patterns[subj2])[0, 1]
                    correlation_matrix[i, j] = corr
        
        avg_correlation = np.mean(correlation_matrix[np.triu_indices(n_subjects, k=1)])
        print(f"被试间平均相关性: {avg_correlation:.3f}")
        
        if avg_correlation < 0.3:
            print("⚠️  被试间相关性极低 - 个体差异是主要挑战！")
        
        return correlation_matrix
    
    def propose_mathematical_solution(self, analysis_results):
        """基于数学分析提出解决方案"""
        print("\n💡 基于数学分析的解决方案")
        print("=" * 50)
        
        solutions = []
        
        # 1. 幅值标准化问题
        if analysis_results['amplitude_cv'] > 0.3:
            solutions.append({
                'problem': '被试间幅值差异过大',
                'solution': 'Z-score标准化 + 分位数标准化',
                'math_formula': 'x_norm = (x - μ) / σ, 然后 quantile_transform',
                'implementation': 'sklearn.preprocessing.QuantileTransformer'
            })
        
        # 2. 频域特征提取
        solutions.append({
            'problem': '时域信号个体差异大',
            'solution': '频域特征提取 + 小波变换',
            'math_formula': 'X(f) = FFT(x(t)), 提取功率谱密度特征',
            'implementation': 'scipy.signal.welch + pywt.cwt'
        })
        
        # 3. 主成分分析降维
        solutions.append({
            'problem': '高维数据中的噪声',
            'solution': 'PCA降维 + 保留主要成分',
            'math_formula': 'X_reduced = X @ V_k, 保留95%方差',
            'implementation': 'sklearn.decomposition.PCA'
        })
        
        # 4. 域适应方法
        if analysis_results.get('correlation_matrix') is not None:
            avg_corr = np.mean(analysis_results['correlation_matrix'])
            if avg_corr < 0.3:
                solutions.append({
                    'problem': '被试间模式差异大',
                    'solution': '域对抗神经网络(DANN) + 最大均值差异(MMD)',
                    'math_formula': 'L = L_cls + λ * L_domain, MMD = ||μ_s - μ_t||²',
                    'implementation': '自定义DANN层 + MMD损失'
                })
        
        for i, sol in enumerate(solutions, 1):
            print(f"\n{i}. {sol['problem']}")
            print(f"   解决方案: {sol['solution']}")
            print(f"   数学公式: {sol['math_formula']}")
            print(f"   实现方法: {sol['implementation']}")
        
        return solutions
    
    def visualize_analysis(self, analysis_results):
        """可视化分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 被试间幅值分布
        amplitudes = []
        subjects_list = []
        for subject in np.unique(self.subjects):
            subject_mask = self.subjects == subject
            subject_data = self.X[subject_mask]
            amp = np.mean(np.abs(subject_data), axis=(1, 2))
            amplitudes.extend(amp)
            subjects_list.extend([subject] * len(amp))
        
        df_amp = pd.DataFrame({'Subject': subjects_list, 'Amplitude': amplitudes})
        sns.boxplot(data=df_amp, x='Subject', y='Amplitude', ax=axes[0, 0])
        axes[0, 0].set_title('被试间幅值分布差异')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 相关性热力图
        if 'correlation_matrix' in analysis_results:
            sns.heatmap(analysis_results['correlation_matrix'], 
                       annot=True, cmap='coolwarm', center=0,
                       xticklabels=np.unique(self.subjects),
                       yticklabels=np.unique(self.subjects),
                       ax=axes[0, 1])
            axes[0, 1].set_title('被试间信号相关性')
        
        # 3. PCA可视化
        X_flat = self.X.reshape(len(self.X), -1)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_flat)
        
        for subject in np.unique(self.subjects):
            mask = self.subjects == subject
            axes[1, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                             label=f'Subject {subject}', alpha=0.6)
        axes[1, 0].set_title(f'PCA可视化 (方差贡献: {pca.explained_variance_ratio_.sum():.2%})')
        axes[1, 0].legend()
        
        # 4. 手势-被试混淆矩阵
        gesture_subject_matrix = np.zeros((len(np.unique(self.y)), len(np.unique(self.subjects))))
        for i, gesture in enumerate(np.unique(self.y)):
            for j, subject in enumerate(np.unique(self.subjects)):
                count = np.sum((self.y == gesture) & (self.subjects == subject))
                gesture_subject_matrix[i, j] = count
        
        sns.heatmap(gesture_subject_matrix, 
                   annot=True, fmt='.0f', cmap='Blues',
                   xticklabels=[f'S{s}' for s in np.unique(self.subjects)],
                   yticklabels=[f'G{g}' for g in np.unique(self.y)],
                   ax=axes[1, 1])
        axes[1, 1].set_title('手势-被试数据分布')
        
        plt.tight_layout()
        plt.savefig('emg_mathematical_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 分析图表已保存: emg_mathematical_analysis.png")
        
        return fig

def main():
    """主函数"""
    print("🧮 EMG数据集数学分析")
    print("=" * 60)
    
    # 创建分析器
    analyzer = EMGMathematicalAnalyzer("datasets/gesture_csv")
    
    # 执行数学分析
    analysis_results = analyzer.analyze_signal_characteristics()
    
    # 提出解决方案
    solutions = analyzer.propose_mathematical_solution(analysis_results)
    
    # 可视化分析
    analyzer.visualize_analysis(analysis_results)
    
    print("\n🎯 核心结论:")
    print("1. 如果幅值CV > 0.3，说明个体差异是主要问题")
    print("2. 如果相关性 < 0.3，说明需要域适应方法")
    print("3. 建议优先使用频域特征而非时域原始信号")
    print("4. PCA降维可能有助于去除个体特异性噪声")

if __name__ == "__main__":
    main()