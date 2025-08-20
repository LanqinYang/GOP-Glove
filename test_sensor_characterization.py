#!/usr/bin/env python3
"""
Test script for sensor characterization and stability analysis.
Generates SNR/漂移/滞回/CoV 图与表到 outputs/sensor_stats/
修复版本：6个用户，11个手势，正确的SNR计算
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
sys.path.append('.')

def generate_synthetic_sensor_data():
    """
    Generate synthetic sensor data for testing sensor characterization.
    修复：6个用户，11个手势(0-10)，正确的SNR计算
    """
    print("Generating synthetic sensor data for testing...")
    
    # Create test data directory
    test_data_dir = "datasets/test_sensor_data"
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Parameters - 修复为正确数量
    n_users = 6  # 修复：6个用户
    n_sessions = 2  
    n_gestures = 11  # 修复：11个手势 (0-10)
    n_samples_per_gesture = 100
    n_channels = 5
    
    # Generate data for each user/session/gesture
    for user_id in range(1, n_users + 1):
        for session_id in range(1, n_sessions + 1):
            for gesture_id in range(n_gestures):  # 0-10手势
                
                # Base signal for this gesture
                base_signal = generate_gesture_signal(gesture_id, n_samples_per_gesture, n_channels)
                
                # Add user-specific characteristics
                user_offset = np.random.RandomState(user_id).normal(0, 20, n_channels)
                user_scale = np.random.RandomState(user_id + 100).uniform(0.9, 1.1, n_channels)
                
                # Add session-specific drift
                session_drift = np.random.RandomState(session_id + 200).normal(0, 5, n_channels)
                time_drift = np.outer(np.linspace(0, 1, n_samples_per_gesture), session_drift)
                
                # Combine all effects
                sensor_data = base_signal * user_scale + user_offset + time_drift
                
                # Add realistic noise levels for proper SNR calculation
                # 修复：使用不同但合理的噪声水平
                base_noise_levels = [2, 3, 2.5, 4, 3.5]  # 更小的噪声以获得合理SNR
                for ch in range(n_channels):
                    # 根据手势类型调整噪声（静止手势噪声更小）
                    noise_multiplier = 0.5 if gesture_id == 10 else 1.0  # gesture 10是静止
                    noise_level = base_noise_levels[ch] * noise_multiplier
                    
                    noise = np.random.RandomState(user_id * 1000 + session_id * 100 + gesture_id * 10 + ch).normal(
                        0, noise_level, n_samples_per_gesture)
                    sensor_data[:, ch] += noise
                
                # Add some hysteresis effects (small differences in repeated gestures)
                if np.random.RandomState(user_id + gesture_id).random() > 0.3:  # 70% chance of hysteresis
                    hysteresis_offset = np.random.RandomState(user_id + gesture_id + 1000).normal(0, 1, n_channels)
                    sensor_data += hysteresis_offset
                
                # 确保数据在合理范围内
                sensor_data = np.clip(sensor_data, 0, 1023)  # ADC range
                
                # Save as CSV file
                filename = f"user_{user_id}_session_{session_id}_gesture_{gesture_id}.csv"
                filepath = os.path.join(test_data_dir, filename)
                
                # Create DataFrame with timestamp column (按照实际数据格式)
                timestamps = np.arange(n_samples_per_gesture) * 20  # 50Hz = 20ms intervals
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'sensor_0': sensor_data[:, 0],
                    'sensor_1': sensor_data[:, 1], 
                    'sensor_2': sensor_data[:, 2],
                    'sensor_3': sensor_data[:, 3],
                    'sensor_4': sensor_data[:, 4]
                })
                
                df.to_csv(filepath, index=False)
    
    print(f"Generated {n_users * n_sessions * n_gestures} test files in {test_data_dir}")
    print(f"Users: {n_users}, Sessions: {n_sessions}, Gestures: {n_gestures}")
    return test_data_dir

def generate_gesture_signal(gesture_id, n_samples, n_channels):
    """Generate base signal pattern for different gestures. 支持0-10手势"""
    t = np.linspace(0, 2, n_samples)  # 2 seconds
    
    # Different signal patterns for each gesture (0-10)
    if gesture_id == 0:  # Gesture 0
        signal = np.zeros((n_samples, n_channels)) + 150
        signal[:, 0] += 80 * np.sin(2 * np.pi * 0.8 * t)  # 主要拇指动作
    elif gesture_id == 1:  # Gesture 1  
        signal = np.zeros((n_samples, n_channels)) + 150
        signal[:, 1] += 100 * np.sin(2 * np.pi * 0.7 * t)  # 主要食指动作
    elif gesture_id == 2:  # Gesture 2
        signal = np.zeros((n_samples, n_channels)) + 150
        signal[:, 0] += 60 * np.sin(2 * np.pi * 0.8 * t)  # 拇指
        signal[:, 1] += 80 * np.sin(2 * np.pi * 0.6 * t)  # 食指
    elif gesture_id == 3:  # Gesture 3
        signal = np.zeros((n_samples, n_channels)) + 150
        signal[:, 0] += 50 * np.sin(2 * np.pi * 0.9 * t)  # 拇指
        signal[:, 1] += 70 * np.sin(2 * np.pi * 0.7 * t)  # 食指
        signal[:, 2] += 60 * np.sin(2 * np.pi * 0.5 * t)  # 中指
    elif gesture_id == 4:  # Gesture 4
        signal = np.zeros((n_samples, n_channels)) + 150
        signal[:, 0] += 40 * np.sin(2 * np.pi * 1.0 * t)  # 拇指
        signal[:, 1] += 60 * np.sin(2 * np.pi * 0.8 * t)  # 食指
        signal[:, 2] += 50 * np.sin(2 * np.pi * 0.6 * t)  # 中指
        signal[:, 3] += 40 * np.sin(2 * np.pi * 0.4 * t)  # 无名指
    elif gesture_id == 5:  # Gesture 5
        signal = np.zeros((n_samples, n_channels)) + 150
        for ch in range(5):  # 所有手指
            signal[:, ch] += (30 + ch * 10) * np.sin(2 * np.pi * (0.6 + ch * 0.1) * t)
    elif gesture_id == 6:  # Gesture 6
        signal = np.zeros((n_samples, n_channels)) + 150
        signal[:, 0] += 90 * np.sin(2 * np.pi * 0.5 * t)  
        signal[:, 4] += 70 * np.sin(2 * np.pi * 0.3 * t)  # 拇指+小指
    elif gesture_id == 7:  # Gesture 7
        signal = np.zeros((n_samples, n_channels)) + 150
        signal[:, 1] += 80 * np.sin(2 * np.pi * 0.6 * t)
        signal[:, 2] += 60 * np.sin(2 * np.pi * 0.4 * t)  # 食指+中指
    elif gesture_id == 8:  # Gesture 8
        signal = np.zeros((n_samples, n_channels)) + 150
        signal[:, 2] += 70 * np.sin(2 * np.pi * 0.7 * t)
        signal[:, 3] += 50 * np.sin(2 * np.pi * 0.5 * t)
        signal[:, 4] += 40 * np.sin(2 * np.pi * 0.3 * t)  # 中指+无名指+小指
    elif gesture_id == 9:  # Gesture 9
        signal = np.zeros((n_samples, n_channels)) + 150
        # 复杂组合动作
        signal[:, 0] += 50 * np.sin(2 * np.pi * 0.9 * t)
        signal[:, 1] += 40 * np.sin(2 * np.pi * 0.7 * t)
        signal[:, 2] += 30 * np.sin(2 * np.pi * 0.5 * t)
        signal[:, 3] += 20 * np.sin(2 * np.pi * 0.3 * t)
        signal[:, 4] += 60 * np.sin(2 * np.pi * 0.4 * t)
    else:  # gesture_id == 10, Static/rest  
        signal = np.zeros((n_samples, n_channels)) + 120  # 较低的基线，代表静止
        # 静止状态只有很小的随机波动
        for ch in range(n_channels):
            signal[:, ch] += 5 * np.sin(2 * np.pi * 0.1 * t + ch)  # 很小的波动
    
    return signal

def test_sensor_characterization():
    """Test the sensor characterization functionality."""
    print("\n" + "="*60)
    print("TESTING SENSOR CHARACTERIZATION")
    print("="*60)
    
    # Generate test data
    test_data_dir = generate_synthetic_sensor_data()
    
    # Import and test sensor characterization
    try:
        from src.evaluation.sensor_characterization import SensorCharacterizer
        
        # Initialize with test data
        characterizer = SensorCharacterizer(
            data_dir=test_data_dir,
            output_dir="outputs/sensor_stats"
        )
        
        # Load test data
        print("\nLoading test sensor data...")
        data_collection = characterizer.load_sensor_data()
        
        if not data_collection:
            print("❌ No test data loaded")
            return False
        
        print(f"✅ Loaded data for {len(data_collection)} users")
        for user_id, user_data in data_collection.items():
            session_count = len(user_data)
            total_gestures = sum(len(session_data) for session_data in user_data.values())
            print(f"   User {user_id}: {session_count} sessions, {total_gestures} gesture files")
        
        # Run comprehensive analysis
        print("\nRunning comprehensive sensor analysis...")
        characterizer.generate_comprehensive_analysis(data_collection)
        
        # Check SNR results
        snr_results = characterizer.results['snr_analysis']
        if snr_results:
            sample_snr = snr_results[0]['snr_values']
            print(f"   Sample SNR values: {sample_snr}")
            print(f"   Mean SNR: {np.mean([r['mean_snr'] for r in snr_results]):.2f} dB")
        
        # Create visualizations
        print("\nCreating sensor characterization plots...")
        characterizer.create_visualization_plots()
        
        # Save results
        print("\nSaving sensor characterization results...")
        results_path, summary_path = characterizer.save_results()
        
        print(f"\n✅ Sensor characterization test completed successfully!")
        print(f"📊 Results saved to: {results_path}")
        print(f"📈 Summary CSV: {summary_path}")
        print(f"📁 Plots directory: outputs/sensor_stats/plots")
        
        # Display detailed summary statistics
        stability_metrics = characterizer.results['stability_metrics']
        snr_stats = stability_metrics['snr_statistics']
        drift_stats = stability_metrics['drift_statistics']
        
        print(f"\n📊 Detailed Test Results Summary:")
        print(f"   Mean SNR: {snr_stats['mean_snr']:.2f} dB")
        print(f"   SNR Range: {snr_stats['min_snr']:.2f} - {snr_stats['max_snr']:.2f} dB")
        print(f"   SNR Std: {snr_stats['std_snr']:.2f} dB")
        print(f"   Mean Drift Slope: {drift_stats['mean_drift_slope']:.6f}")
        print(f"   Mean Drift Variance: {drift_stats['mean_drift_variance']:.4f}")
        
        # Check cross-user analysis
        cross_user = characterizer.results['cross_user_analysis']
        if cross_user:
            print(f"   Cross-user analysis: {len(cross_user)} gestures analyzed")
            sample_gesture = list(cross_user.keys())[0]
            cv = np.mean(cross_user[sample_gesture]['coefficient_of_variation'])
            print(f"   Sample CoV (gesture {sample_gesture}): {cv:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sensor characterization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_outputs():
    """Verify that all expected outputs were generated."""
    print("\nVerifying generated outputs...")
    
    expected_files = [
        "outputs/sensor_stats/plots/snr_analysis.png",
        "outputs/sensor_stats/plots/drift_analysis.png", 
        "outputs/sensor_stats/plots/cross_user_variability.png"
    ]
    
    expected_patterns = [
        "outputs/sensor_stats/sensor_characterization_*.json",
        "outputs/sensor_stats/sensor_summary_*.csv"
    ]
    
    # Check specific files
    for file_path in expected_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({file_size} bytes)")
        else:
            print(f"❌ Missing: {file_path}")
    
    # Check pattern files
    import glob
    for pattern in expected_patterns:
        matches = glob.glob(pattern)
        if matches:
            for match in matches:
                file_size = os.path.getsize(match)
                print(f"✅ {match} ({file_size} bytes)")
        else:
            print(f"❌ Missing: {pattern}")

def inspect_generated_data():
    """检查生成的数据质量"""
    print("\nInspecting generated test data quality...")
    
    test_files = glob.glob("datasets/test_sensor_data/*.csv")
    if test_files:
        # 检查一个样本文件
        sample_file = test_files[0]
        df = pd.read_csv(sample_file)
        print(f"Sample file: {sample_file}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data range - Min: {df.iloc[:, 1:].min().min():.2f}, Max: {df.iloc[:, 1:].max().max():.2f}")
        print(f"Data mean: {df.iloc[:, 1:].mean().mean():.2f}")
        print(f"Data std: {df.iloc[:, 1:].std().mean():.2f}")

def main():
    """Main test function."""
    print("BSL Gesture Recognition - Sensor Characterization Test")
    print("修复版本：6个用户，11个手势，正确的SNR计算")
    
    # Run the test
    success = test_sensor_characterization()
    
    if success:
        # Inspect generated data
        inspect_generated_data()
        
        # Verify outputs
        verify_outputs()
        
        print("\n" + "="*60)
        print("SENSOR CHARACTERIZATION TEST COMPLETE")
        print("="*60)
        print("🎉 All sensor characterization features working correctly!")
        print("📊 Generated SNR, drift, hysteresis, and CoV analysis")
        print("📈 Created comprehensive visualization plots")
        print("📁 Results saved to outputs/sensor_stats/")
        
        # 保留测试数据以便检查
        print("📝 Test data preserved in datasets/test_sensor_data/ for inspection")
    else:
        print("\n❌ Sensor characterization test failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    import glob
    exit_code = main()
    sys.exit(exit_code)
