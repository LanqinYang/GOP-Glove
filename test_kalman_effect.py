#!/usr/bin/env python3
"""
Kalman滤波效果测试
快速测试Kalman滤波对数据和模型性能的影响
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def load_raw_data(csv_dir):
    """加载原始数据（不应用任何滤波）"""
    print("📁 加载原始数据...")
    
    X_data = []
    y_data = []
    subjects = []
    
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        print(f"❌ 数据目录不存在: {csv_dir}")
        return None, None, None
    
    csv_files = list(csv_path.glob("*.csv"))
    if not csv_files:
        print(f"❌ 没有找到CSV文件: {csv_dir}")
        return None, None, None
    
    print(f"   找到 {len(csv_files)} 个CSV文件")
    
    for csv_file in csv_files:
        try:
            # 从文件名解析信息
            # 格式: user_006_gesture_1_One_sample_2_20250724_211119.csv
            filename = csv_file.stem
            parts = filename.split('_')
            if len(parts) >= 6:
                subject_id = int(parts[1])  # user_006 -> 006
                gesture_id = int(parts[3])  # gesture_1 -> 1
            else:
                continue
            
            # 读取数据
            df = pd.read_csv(csv_file, comment='#', header=0)
            
            # 提取传感器数据（跳过timestamp列，取后5列）
            sensor_data = df.iloc[:, 1:6].values.astype(float)  # thumb,index,middle,ring,pinky
            
            # 标准化长度到100
            if len(sensor_data) != 100:
                indices = np.linspace(0, len(sensor_data) - 1, 100).astype(int)
                sensor_data = sensor_data[indices]
            
            X_data.append(sensor_data)
            y_data.append(gesture_id)
            subjects.append(subject_id)
            
        except Exception as e:
            print(f"   ⚠️ 跳过文件 {csv_file.name}: {e}")
            continue
    
    if not X_data:
        print("❌ 没有成功加载任何数据")
        return None, None, None
    
    X = np.array(X_data)
    y = np.array(y_data)
    subjects = np.array(subjects)
    
    print(f"✅ 成功加载数据: {len(X)} 样本, {len(np.unique(subjects))} 被试, {len(np.unique(y))} 手势")
    return X, y, subjects

def apply_kalman_filter(data, Q=1e-5, R=0.1**2):
    """应用简单的Kalman滤波"""
    filtered_data = np.zeros_like(data)
    
    for sample_idx in range(data.shape[0]):
        for channel in range(data.shape[2]):
            # 对每个通道应用Kalman滤波
            signal = data[sample_idx, :, channel]
            
            # 初始化
            x = signal[0]  # 初始状态
            P = 1.0        # 初始协方差
            
            filtered_signal = np.zeros_like(signal)
            
            for i, measurement in enumerate(signal):
                # 预测步骤
                x_pred = x
                P_pred = P + Q
                
                # 更新步骤
                K = P_pred / (P_pred + R)  # Kalman增益
                x = x_pred + K * (measurement - x_pred)
                P = (1 - K) * P_pred
                
                filtered_signal[i] = x
            
            filtered_data[sample_idx, :, channel] = filtered_signal
    
    return filtered_data

def extract_simple_features(X):
    """提取简单特征用于快速测试"""
    features = []
    
    for sample in X:
        sample_features = []
        
        # 对每个通道提取基本统计特征
        for ch in range(sample.shape[1]):
            channel_data = sample[:, ch]
            
            sample_features.extend([
                np.mean(channel_data),      # 均值
                np.std(channel_data),       # 标准差
                np.min(channel_data),       # 最小值
                np.max(channel_data),       # 最大值
                np.median(channel_data),    # 中位数
                np.var(channel_data)        # 方差
            ])
        
        features.append(sample_features)
    
    return np.array(features)

def test_kalman_vs_raw():
    """测试Kalman滤波 vs 原始数据的效果"""
    print("🧪 Kalman滤波效果对比测试")
    print("=" * 50)
    
    # 加载数据
    X, y, subjects = load_raw_data("datasets/gesture_csv")
    if X is None:
        return
    
    print(f"数据形状: {X.shape}")
    print(f"手势分布: {np.bincount(y)}")
    print(f"被试分布: {np.bincount(subjects)}")
    
    # 应用Kalman滤波
    print("\n🔧 应用Kalman滤波...")
    X_kalman = apply_kalman_filter(X)
    
    # 提取特征
    print("🔍 提取特征...")
    X_raw_features = extract_simple_features(X)
    X_kalman_features = extract_simple_features(X_kalman)
    
    print(f"特征维度: {X_raw_features.shape[1]}")
    
    # 数据分割
    X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X_raw_features, y, test_size=0.3, random_state=42, stratify=y
    )
    X_kalman_train, X_kalman_test, _, _ = train_test_split(
        X_kalman_features, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 特征标准化
    scaler_raw = StandardScaler()
    scaler_kalman = StandardScaler()
    
    X_raw_train_scaled = scaler_raw.fit_transform(X_raw_train)
    X_raw_test_scaled = scaler_raw.transform(X_raw_test)
    
    X_kalman_train_scaled = scaler_kalman.fit_transform(X_kalman_train)
    X_kalman_test_scaled = scaler_kalman.transform(X_kalman_test)
    
    # 训练模型
    print("\n🚀 训练随机森林模型...")
    
    # 原始数据模型
    rf_raw = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_raw.fit(X_raw_train_scaled, y_train)
    y_pred_raw = rf_raw.predict(X_raw_test_scaled)
    acc_raw = accuracy_score(y_test, y_pred_raw)
    
    # Kalman滤波数据模型
    rf_kalman = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_kalman.fit(X_kalman_train_scaled, y_train)
    y_pred_kalman = rf_kalman.predict(X_kalman_test_scaled)
    acc_kalman = accuracy_score(y_test, y_pred_kalman)
    
    # 结果对比
    print("\n📊 结果对比:")
    print(f"   原始数据准确率:     {acc_raw:.4f} ({acc_raw*100:.2f}%)")
    print(f"   Kalman滤波准确率:   {acc_kalman:.4f} ({acc_kalman*100:.2f}%)")
    print(f"   性能差异:           {acc_kalman - acc_raw:.4f} ({(acc_kalman - acc_raw)*100:.2f}%)")
    
    if acc_kalman > acc_raw:
        print("✅ Kalman滤波提升了性能")
    elif acc_kalman < acc_raw:
        print("❌ Kalman滤波降低了性能")
        print("💡 建议：考虑移除Kalman滤波")
    else:
        print("➖ Kalman滤波对性能无明显影响")
    
    # 数据质量分析
    print("\n📈 数据质量分析:")
    
    # 计算信噪比变化
    raw_std = np.std(X_raw_features, axis=0).mean()
    kalman_std = np.std(X_kalman_features, axis=0).mean()
    
    print(f"   原始数据标准差:     {raw_std:.4f}")
    print(f"   Kalman滤波标准差:   {kalman_std:.4f}")
    print(f"   噪声减少比例:       {(raw_std - kalman_std) / raw_std * 100:.2f}%")
    
    # 特征相关性变化
    raw_corr = np.corrcoef(X_raw_features.T).mean()
    kalman_corr = np.corrcoef(X_kalman_features.T).mean()
    
    print(f"   原始数据特征相关性: {raw_corr:.4f}")
    print(f"   Kalman特征相关性:   {kalman_corr:.4f}")
    
    # 建议
    print("\n💡 建议:")
    performance_diff = acc_kalman - acc_raw
    
    if performance_diff < -0.02:  # 性能下降超过2%
        print("   🔴 强烈建议移除Kalman滤波 - 显著降低性能")
    elif performance_diff < -0.005:  # 性能下降超过0.5%
        print("   🟡 建议移除Kalman滤波 - 轻微降低性能")
    elif performance_diff > 0.02:  # 性能提升超过2%
        print("   🟢 建议保留Kalman滤波 - 显著提升性能")
    elif performance_diff > 0.005:  # 性能提升超过0.5%
        print("   🟢 建议保留Kalman滤波 - 轻微提升性能")
    else:
        print("   ➖ Kalman滤波影响很小，可保留或移除")
    
    return acc_raw, acc_kalman

def main():
    """主函数"""
    try:
        acc_raw, acc_kalman = test_kalman_vs_raw()
        
        print("\n" + "=" * 50)
        print("🎯 测试完成")
        
        if acc_raw is not None and acc_kalman is not None:
            if acc_kalman < acc_raw:
                print("📝 结论：Kalman滤波可能对当前数据集有害")
                print("🔧 建议：在pipeline中禁用Kalman滤波")
            else:
                print("📝 结论：Kalman滤波对当前数据集有益或无害")
                print("🔧 建议：可以保留Kalman滤波")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()