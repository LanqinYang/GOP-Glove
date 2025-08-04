#!/usr/bin/env python3
"""
简单的Kalman滤波效果测试
基于现有pipeline代码进行测试
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.training.pipeline import load_data, apply_kalman_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def extract_simple_features(X):
    """提取简单特征用于快速测试"""
    features = []
    
    for sample in X:
        sample_features = []
        
        # 对每个通道提取基本统计特征
        for ch in range(sample.shape[1]):  # 5个传感器通道
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

def test_kalman_effect():
    """测试Kalman滤波效果"""
    print("🧪 Kalman滤波效果测试")
    print("=" * 50)
    
    # 加载数据
    print("📁 加载数据...")
    X, y, subjects = load_data("datasets/gesture_csv")
    
    if X is None or len(X) == 0:
        print("❌ 数据加载失败")
        return
    
    print(f"✅ 成功加载数据: {len(X)} 样本, {len(np.unique(subjects))} 被试, {len(np.unique(y))} 手势")
    print(f"数据形状: {X.shape}")
    
    # 1. 测试原始数据
    print("\n🔍 测试原始数据...")
    X_raw_features = extract_simple_features(X)
    
    # 2. 测试Kalman滤波数据
    print("🔧 应用Kalman滤波...")
    
    # 对每个样本应用Kalman滤波
    X_kalman = np.array([apply_kalman_filter(sample) for sample in X])
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
        recommendation = "保留"
    elif acc_kalman < acc_raw:
        print("❌ Kalman滤波降低了性能")
        recommendation = "移除"
    else:
        print("➖ Kalman滤波对性能无明显影响")
        recommendation = "可选"
    
    # 数据质量分析
    print("\n📈 数据质量分析:")
    
    # 计算信噪比变化
    raw_std = np.std(X_raw_features, axis=0).mean()
    kalman_std = np.std(X_kalman_features, axis=0).mean()
    
    print(f"   原始数据标准差:     {raw_std:.4f}")
    print(f"   Kalman滤波标准差:   {kalman_std:.4f}")
    print(f"   噪声减少比例:       {(raw_std - kalman_std) / raw_std * 100:.2f}%")
    
    # 建议
    print("\n💡 建议:")
    performance_diff = acc_kalman - acc_raw
    
    if performance_diff < -0.02:  # 性能下降超过2%
        print("   🔴 强烈建议移除Kalman滤波 - 显著降低性能")
        final_recommendation = "删除Kalman滤波代码"
    elif performance_diff < -0.005:  # 性能下降超过0.5%
        print("   🟡 建议移除Kalman滤波 - 轻微降低性能")
        final_recommendation = "考虑删除Kalman滤波代码"
    elif performance_diff > 0.02:  # 性能提升超过2%
        print("   🟢 建议保留Kalman滤波 - 显著提升性能")
        final_recommendation = "保留Kalman滤波代码"
    elif performance_diff > 0.005:  # 性能提升超过0.5%
        print("   🟢 建议保留Kalman滤波 - 轻微提升性能")
        final_recommendation = "保留Kalman滤波代码"
    else:
        print("   ➖ Kalman滤波影响很小，可保留或移除")
        final_recommendation = "Kalman滤波可选"
    
    print(f"\n🎯 最终建议: {final_recommendation}")
    
    return acc_raw, acc_kalman, final_recommendation

def main():
    """主函数"""
    try:
        acc_raw, acc_kalman, recommendation = test_kalman_effect()
        
        print("\n" + "=" * 50)
        print("🎯 测试完成")
        print(f"📝 结论: {recommendation}")
        
        # 如果Kalman滤波有害，提供具体的代码修改建议
        if "删除" in recommendation:
            print("\n🔧 代码修改建议:")
            print("   1. 在 src/training/pipeline.py 中注释或删除 apply_kalman_filter 函数")
            print("   2. 在各模型文件中移除 kalman_ 相关的超参数定义")
            print("   3. 在 load_and_clean_data 函数中禁用 Kalman 滤波")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()