#!/usr/bin/env python3
"""
测试Arduino固件的特征提取是否与训练时一致
"""

import numpy as np
import pandas as pd
import glob
import os

def extract_arduino_features_python(sample):
    """Python版本的Arduino特征提取（与Arduino固件完全一致）"""
    features = []
    
    # 对每个通道提取特征
    for ch in range(sample.shape[1]):
        channel_data = sample[:, ch]
        
        # 计算基本统计特征
        mu = float(np.mean(channel_data)) if len(channel_data) > 0 else 0.0
        sd = float(np.std(channel_data)) if len(channel_data) > 0 else 0.0
        mn = float(np.min(channel_data)) if len(channel_data) > 0 else 0.0
        mx = float(np.max(channel_data)) if len(channel_data) > 0 else 0.0
        
        features.extend([mu, sd, mn, mx])
    
    return np.array(features, dtype=np.float32)

def load_sample_data(csv_file):
    """加载单个样本数据"""
    try:
        # 读取CSV，跳过注释行
        df = pd.read_csv(csv_file, comment='#', skiprows=2)
        
        # 提取传感器数据（跳过timestamp列）
        sensor_columns = ['thumb', 'index', 'middle', 'ring', 'pinky']
        data = df[sensor_columns].values
        
        # 重采样到100个时间点
        if len(data) != 100:
            indices = np.linspace(0, len(data) - 1, 100)
            resampled = np.zeros((100, 5))
            for i in range(5):
                resampled[:, i] = np.interp(indices, range(len(data)), data[:, i])
            data = resampled
        
        return data
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None

def test_gesture_specific_features():
    """测试特定手势的特征提取"""
    print("\n🎯 测试特定手势的特征提取...")
    
    # 测试手势1（One）和手势5（Five），因为你说这两个识别准确
    test_gestures = [1, 5]
    
    for gesture_id in test_gestures:
        print(f"\n📊 手势 {gesture_id} (One)" if gesture_id == 1 else f"\n📊 手势 {gesture_id} (Five)")
        
        # 找到对应的CSV文件
        csv_files = glob.glob(f"datasets/gesture_csv/user_001_gesture_{gesture_id}_*_sample_1_*.csv")
        if not csv_files:
            print(f"   找不到手势 {gesture_id} 的数据文件")
            continue
            
        csv_file = csv_files[0]
        print(f"   文件: {os.path.basename(csv_file)}")
        
        # 加载数据
        data = load_sample_data(csv_file)
        if data is None:
            continue
        
        # 提取特征
        features = extract_arduino_features_python(data)
        
        # 标准化
        scaler_mean = [217.92487890, 99.12553671, 39.38333333, 321.96363636, 134.00172744, 
                       54.51530460, 46.38484848, 207.51818182, 357.86704505, 81.96671142, 
                       187.82727273, 452.09242424, 274.56631785, 113.44793942, 48.06515152, 
                       379.94696970, 139.62681811, 76.01981062, 11.61818182, 215.74696970]
        
        scaler_scale = [158.40359425, 78.91607427, 59.64363890, 204.02523287, 108.81152311, 
                        75.26881004, 33.49912365, 177.95619474, 151.03873799, 61.48046568, 
                        129.55241310, 157.90832286, 145.04109254, 69.07941653, 42.09578181, 
                        180.82354174, 89.29714414, 45.53607427, 25.03548336, 115.04106476]
        
        normalized_features = []
        for i in range(len(features)):
            normalized = (features[i] - scaler_mean[i]) / scaler_scale[i]
            normalized_features.append(normalized)
        
        print(f"   原始特征: {features}")
        print(f"   标准化后: {normalized_features}")
        
        # 分析每个通道的特征
        channel_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, ch_name in enumerate(channel_names):
            start_idx = i * 4
            ch_features = features[start_idx:start_idx+4]
            ch_normalized = normalized_features[start_idx:start_idx+4]
            print(f"   {ch_name}: mean={ch_features[0]:.1f}, std={ch_features[1]:.1f}, min={ch_features[2]:.1f}, max={ch_features[3]:.1f}")
            print(f"   {ch_name} (标准化): {ch_normalized}")

def test_normalization():
    """测试标准化过程"""
    print("\n🔧 测试标准化过程...")
    
    # Arduino模型文件中的标准化参数
    scaler_mean = [217.92487890, 99.12553671, 39.38333333, 321.96363636, 134.00172744, 
                   54.51530460, 46.38484848, 207.51818182, 357.86704505, 81.96671142, 
                   187.82727273, 452.09242424, 274.56631785, 113.44793942, 48.06515152, 
                   379.94696970, 139.62681811, 76.01981062, 11.61818182, 215.74696970]
    
    scaler_scale = [158.40359425, 78.91607427, 59.64363890, 204.02523287, 108.81152311, 
                    75.26881004, 33.49912365, 177.95619474, 151.03873799, 61.48046568, 
                    129.55241310, 157.90832286, 145.04109254, 69.07941653, 42.09578181, 
                    180.82354174, 89.29714414, 45.53607427, 25.03548336, 115.04106476]
    
    # 测试几个样本
    csv_files = glob.glob("datasets/gesture_csv/user_001_gesture_*_sample_1_*.csv")
    csv_files = csv_files[:3]  # 只测试前3个文件
    
    for csv_file in csv_files:
        print(f"\n📁 测试文件: {os.path.basename(csv_file)}")
        
        # 加载数据
        data = load_sample_data(csv_file)
        if data is None:
            continue
        
        # 提取特征
        features = extract_arduino_features_python(data)
        
        # 标准化
        normalized_features = []
        for i in range(len(features)):
            normalized = (features[i] - scaler_mean[i]) / scaler_scale[i]
            normalized_features.append(normalized)
        
        print(f"   原始特征: {features[:8]}...")  # 只显示前8个
        print(f"   标准化后: {normalized_features[:8]}...")
        
        # 检查标准化后的值是否在合理范围内
        normalized_array = np.array(normalized_features)
        print(f"   标准化范围: {normalized_array.min():.3f} - {normalized_array.max():.3f}")
        print(f"   标准化均值: {normalized_array.mean():.3f}")
        print(f"   标准化标准差: {normalized_array.std():.3f}")

def test_feature_extraction():
    """测试特征提取的一致性"""
    print("🧪 测试Arduino特征提取一致性...")
    
    # 加载几个样本进行测试
    csv_files = glob.glob("datasets/gesture_csv/user_001_gesture_*_sample_1_*.csv")
    csv_files = csv_files[:5]  # 只测试前5个文件
    
    for csv_file in csv_files:
        print(f"\n📁 测试文件: {os.path.basename(csv_file)}")
        
        # 加载数据
        data = load_sample_data(csv_file)
        if data is None:
            continue
        
        print(f"   数据形状: {data.shape}")
        print(f"   数据范围: {data.min():.1f} - {data.max():.1f}")
        
        # 提取特征
        features = extract_arduino_features_python(data)
        print(f"   特征数量: {len(features)}")
        print(f"   特征值: {features}")
        
        # 检查特征是否合理
        for i, (ch_name, ch_idx) in enumerate([('thumb', 0), ('index', 1), ('middle', 2), ('ring', 3), ('pinky', 4)]):
            start_idx = i * 4
            ch_features = features[start_idx:start_idx+4]
            print(f"   {ch_name}: mean={ch_features[0]:.1f}, std={ch_features[1]:.1f}, min={ch_features[2]:.1f}, max={ch_features[3]:.1f}")

if __name__ == "__main__":
    test_feature_extraction()
    test_normalization()
    test_gesture_specific_features()
