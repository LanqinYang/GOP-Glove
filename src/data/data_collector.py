"""
BSL手势识别系统 - 数据采集模块

该模块负责：
1. 通过串口与Arduino通信接收传感器数据
2. 实时显示数据流
3. 按类别收集手势数据
4. 保存数据集为numpy格式

作者: Lambert Yang
版本: 1.0
"""

import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import csv
import json
from typing import List, Dict, Tuple, Optional
from collections import deque
import threading
from datetime import datetime
from loguru import logger


class BSLDataCollector:
    """BSL手势数据采集器"""
    
    def __init__(self, port: str = '/dev/ttyACM0', baudrate: int = 115200):
        """
        初始化数据采集器
        
        Args:
            port: Arduino串口端口
            baudrate: 波特率
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_collecting = False
        self.is_connected = False
        
        # 数据采集参数
        self.sample_rate = 50  # 50Hz
        self.sequence_length = 100  # 2秒 × 50Hz = 100个时间步
        self.num_sensors = 5
        
        # BSL手势类别定义
        self.gesture_classes = {
            0: "Rest/Neutral",
            1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
            6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Zero"
        }
        
        # 数据缓存
        self.data_buffer = deque(maxlen=self.sequence_length)
        self.collected_data = {class_id: [] for class_id in self.gesture_classes.keys()}
        
        # 实时可视化
        self.plot_enabled = False
        self.plot_data = deque(maxlen=200)  # 显示最近4秒的数据
        
        logger.info(f"Raw data collector initialized")
        logger.info(f"Target classes: {list(self.gesture_classes.values())}")
        logger.info("Mode: Raw ADC values (no filtering applied)")
    
    def connect(self) -> bool:
        """
        Connect to Arduino device
        
        Returns:
            bool: Connection success status
        """
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino initialization
            
            # Test connection
            if self.serial_conn.is_open:
                logger.success(f"Successfully connected to Arduino: {self.port}")
                print(f"✅ Connected to Arduino at {self.port}")
                self.is_connected = True
                return True
            else:
                logger.error(f"Cannot open serial port: {self.port}")
                print(f"❌ Cannot open serial port: {self.port}")
                return False
                
        except serial.SerialException as e:
            logger.error(f"Serial connection failed: {e}")
            print(f"❌ Serial connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.is_connected = False
            logger.info("Arduino disconnected")
            print("Arduino disconnected")
    
    def read_sensor_data(self) -> Optional[List[float]]:
        """
        读取一行传感器数据
        
        Returns:
            List[float]: [sensor1, sensor2, sensor3, sensor4, sensor5] 或 None
        """
        if not self.is_connected or not self.serial_conn:
            return None
        
        try:
            line = self.serial_conn.readline().decode('utf-8').strip()
            if line:
                # 添加调试信息
                if hasattr(self, '_debug_counter'):
                    self._debug_counter += 1
                else:
                    self._debug_counter = 1
                
                # 每100个数据包打印一次调试信息
                if self._debug_counter % 100 == 0:
                    print(f"Debug - Raw data: {line}")
                
                if ',' in line:
                    parts = line.split(',')
                    if len(parts) == 6:  # timestamp + 5个传感器值
                        try:
                            # Try to parse as integers first (raw ADC), then as floats (filtered)
                            sensor_values = []
                            for i in range(1, 6):
                                try:
                                    sensor_values.append(int(parts[i]))
                                except ValueError:
                                    sensor_values.append(float(parts[i]))
                            return sensor_values
                        except (ValueError, IndexError):
                            return None
                    elif len(parts) == 5:  # 只有5个传感器值
                        try:
                            sensor_values = []
                            for part in parts:
                                try:
                                    sensor_values.append(int(part))
                                except ValueError:
                                    sensor_values.append(float(part))
                            return sensor_values
                        except (ValueError, IndexError):
                            return None
        except (ValueError, UnicodeDecodeError) as e:
            logger.warning(f"Data parsing error: {e} - Raw line: {line}")
        
        return None
    
    def collect_gesture_sequence(self, gesture_class: int, user_id: str = "user1") -> bool:
        """
        采集单个手势序列
        
        Args:
            gesture_class: 手势类别 (0-10)
            user_id: 用户ID
            
        Returns:
            bool: 采集是否成功
        """
        if not self.is_connected:
            logger.error("Arduino未连接")
            return False
        
        gesture_name = self.gesture_classes[gesture_class]
        logger.info(f"准备采集手势: {gesture_name} (类别 {gesture_class})")
        
        # 清空缓存
        self.data_buffer.clear()
        
        # 倒计时
        for i in range(3, 0, -1):
            print(f"\r倒计时: {i} 秒...", end='', flush=True)
            time.sleep(1)
        print(f"\r开始采集手势: {gesture_name} - 请保持动作2秒!")
        
        start_time = time.time()
        target_samples = self.sequence_length
        collected_samples = 0
        
        while collected_samples < target_samples:
            sensor_data = self.read_sensor_data()
            if sensor_data:
                self.data_buffer.append(sensor_data)
                collected_samples += 1
                
                # 显示进度
                progress = int((collected_samples / target_samples) * 20)
                bar = '█' * progress + '░' * (20 - progress)
                print(f"\r进度: [{bar}] {collected_samples}/{target_samples}", end='', flush=True)
        
        # 保存采集到的序列
        if len(self.data_buffer) == self.sequence_length:
            sequence = np.array(list(self.data_buffer))
            
            # 添加元数据
            sample_data = {
                'sequence': sequence,
                'label': gesture_class,
                'gesture_name': gesture_name,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'duration': time.time() - start_time
            }
            
            self.collected_data[gesture_class].append(sample_data)
            
            print(f"\n✅ 成功采集手势 '{gesture_name}' - 共{len(sequence)}个时间步")
            logger.success(f"手势序列已保存: {gesture_name}")
            return True
        else:
            print(f"\n❌ 采集失败: 数据长度不足 ({len(self.data_buffer)}/{self.sequence_length})")
            return False
    
    def batch_collect_gestures(self, gestures: List[int], samples_per_gesture: int = 5, 
                              user_id: str = "user1"):
        """
        批量采集多个手势
        
        Args:
            gestures: 要采集的手势类别列表
            samples_per_gesture: 每个手势采集的样本数
            user_id: 用户ID
        """
        total_samples = len(gestures) * samples_per_gesture
        current_sample = 0
        
        logger.info(f"开始批量采集 - 总共需要采集 {total_samples} 个样本")
        
        for gesture_class in gestures:
            gesture_name = self.gesture_classes[gesture_class]
            logger.info(f"\n正在采集手势: {gesture_name}")
            
            for sample_idx in range(samples_per_gesture):
                current_sample += 1
                print(f"\n--- 样本 {sample_idx + 1}/{samples_per_gesture} ---")
                print(f"总体进度: {current_sample}/{total_samples}")
                
                success = self.collect_gesture_sequence(gesture_class, user_id)
                if not success:
                    logger.warning(f"手势 {gesture_name} 的第 {sample_idx + 1} 个样本采集失败")
                
                # 短暂休息
                time.sleep(1)
        
        logger.success("批量采集完成!")
    
    def collect_and_save_csv(self, duration: int = 30, output_dir: str = "datasets/csv", filename: str = None) -> str:
        """
        实时采集传感器数据并保存为CSV格式
        
        Args:
            duration: 采集时长（秒）
            output_dir: 输出目录
            filename: 文件名（不包含扩展名）
            
        Returns:
            str: 保存的CSV文件路径
        """
        if not self.is_connected:
            logger.error("Arduino未连接")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sensor_data_{timestamp}"
        
        csv_path = os.path.join(output_dir, f"{filename}.csv")
        
        # 准备CSV文件
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头（时间戳以毫秒为单位，从采集开始计算）
            writer.writerow(['timestamp_ms', 'thumb_sensor', 'index_sensor', 'middle_sensor', 'ring_sensor', 'pinky_sensor'])
            
            logger.info(f"开始采集传感器数据，时长: {duration}秒")
            print(f"🎯 开始采集5个传感器数据，时长: {duration}秒")
            print("📊 数据将保存到:", csv_path)
            
            start_time = time.time()
            sample_count = 0
            last_progress_time = start_time
            
            while time.time() - start_time < duration:
                sensor_data = self.read_sensor_data()
                if sensor_data and len(sensor_data) == 5:
                    # 计算相对时间戳（从开始采集的毫秒数）
                    relative_timestamp = (time.time() - start_time) * 1000  # 转换为毫秒
                    
                    # 写入CSV数据：相对时间戳 + 5个传感器值
                    row = [f"{relative_timestamp:.3f}"] + sensor_data
                    writer.writerow(row)
                    sample_count += 1
                    
                    # 每秒显示一次进度
                    current_time = time.time()
                    if current_time - last_progress_time >= 1.0:
                        elapsed = current_time - start_time
                        remaining = duration - elapsed
                        progress = (elapsed / duration) * 100
                        avg_rate = sample_count / elapsed if elapsed > 0 else 0
                        
                        print(f"\r⏱️  进度: {progress:.1f}% | 已采集: {sample_count} 样本 | 采样率: {avg_rate:.1f} Hz | 剩余: {remaining:.1f}秒", 
                              end='', flush=True)
                        last_progress_time = current_time
                
                # 小延迟避免过度占用CPU
                time.sleep(0.001)
            
            print(f"\n✅ 数据采集完成!")
            print(f"📁 总样本数: {sample_count}")
            print(f"📄 CSV文件: {csv_path}")
            
            logger.success(f"CSV数据已保存: {csv_path}")
            logger.info(f"总样本数: {sample_count}, 平均采样率: {sample_count/duration:.1f} Hz")
            
            return csv_path
    
    def auto_gesture_collection(self, user_id: str, gestures: List[int] = None, 
                               duration_per_gesture: int = 2, output_dir: str = "datasets/gesture_csv") -> List[str]:
        """
        自动手势数据采集程序
        
        Args:
            user_id: 用户ID（如 "001"）
            gestures: 要采集的手势列表，默认为[0,1,2,3,4,5,6,7,8,9,10]
            duration_per_gesture: 每个手势采集时长（秒）
            output_dir: 输出目录
            
        Returns:
            List[str]: 保存的CSV文件路径列表
        """
        if not self.is_connected:
            logger.error("Arduino未连接")
            return []
        
        if gestures is None:
            gestures = list(range(11))  # 0-10所有手势
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        print(f"\n🎯 开始为用户 {user_id} 自动采集手势数据")
        print(f"📋 将采集手势: {[self.gesture_classes[g] for g in gestures]}")
        print(f"⏱️  每个手势采集 {duration_per_gesture} 秒")
        print("=" * 60)
        
        for i, gesture_id in enumerate(gestures):
            gesture_name = self.gesture_classes[gesture_id]
            
            print(f"\n📊 [{i+1}/{len(gestures)}] 准备采集手势: {gesture_name} (ID: {gesture_id})")
            print(f"🤟 请准备做出 '{gesture_name}' 手势")
            
            # 等待用户按回车
            input("👆 准备好后按回车键开始采集...")
            
            # 生成文件名：用户ID_手势ID_手势名称_时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 清理手势名称，替换文件名中的非法字符
            clean_gesture_name = gesture_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
            filename = f"user_{user_id}_gesture_{gesture_id}_{clean_gesture_name}_{timestamp}"
            csv_path = os.path.join(output_dir, f"{filename}.csv")
            
            # 开始采集
            print(f"🎬 开始采集 '{gesture_name}' - 保持手势 {duration_per_gesture} 秒!")
            
            # 准备CSV文件
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # 写入表头和元数据
                writer.writerow(['# BSL Gesture Data Collection'])
                writer.writerow(['# User ID:', user_id])
                writer.writerow(['# Gesture ID:', gesture_id])
                writer.writerow(['# Gesture Name:', gesture_name])
                writer.writerow(['# Duration:', f'{duration_per_gesture}s'])
                writer.writerow(['# Collection Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow([])  # 空行分隔
                writer.writerow(['timestamp_ms', 'thumb_sensor', 'index_sensor', 'middle_sensor', 'ring_sensor', 'pinky_sensor'])
                
                start_time = time.time()
                sample_count = 0
                last_print_time = start_time
                
                while time.time() - start_time < duration_per_gesture:
                    sensor_data = self.read_sensor_data()
                    if sensor_data and len(sensor_data) == 5:
                        # 计算相对时间戳
                        relative_timestamp = (time.time() - start_time) * 1000
                        
                        # 写入数据
                        row = [f"{relative_timestamp:.3f}"] + sensor_data
                        writer.writerow(row)
                        sample_count += 1
                        
                        # 每0.5秒显示一次进度
                        current_time = time.time()
                        if current_time - last_print_time >= 0.5:
                            elapsed = current_time - start_time
                            remaining = duration_per_gesture - elapsed
                            progress = (elapsed / duration_per_gesture) * 100
                            
                            print(f"\r⏱️  进度: {progress:.1f}% | 样本: {sample_count} | 剩余: {remaining:.1f}秒   ", 
                                  end='', flush=True)
                            last_print_time = current_time
                    
                    time.sleep(0.001)  # 小延迟
            
            print(f"\n✅ 完成! 采集了 {sample_count} 个样本")
            print(f"📁 文件保存: {csv_path}")
            saved_files.append(csv_path)
            
            # 短暂休息，除非是最后一个手势
            if i < len(gestures) - 1:
                print("😌 休息一下...")
                time.sleep(1)
        
        print("\n🎉 所有手势采集完成!")
        print(f"📂 共生成 {len(saved_files)} 个CSV文件:")
        for file_path in saved_files:
            print(f"   📄 {os.path.basename(file_path)}")
        
        logger.success(f"用户 {user_id} 的手势数据采集完成，共 {len(saved_files)} 个文件")
        return saved_files
    
    def save_dataset(self, output_dir: str = "datasets/raw", filename: str = None):
        """
        保存采集的数据集
        
        Args:
            output_dir: 输出目录
            filename: 文件名前缀
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bsl_dataset_{timestamp}"
        
        # 统计数据
        total_samples = sum(len(samples) for samples in self.collected_data.values())
        if total_samples == 0:
            logger.warning("没有数据可保存")
            return
        
        # 准备数据
        X_list = []
        y_list = []
        metadata_list = []
        
        for class_id, samples in self.collected_data.items():
            for sample_data in samples:
                X_list.append(sample_data['sequence'])
                y_list.append(sample_data['label'])
                metadata_list.append({
                    'gesture_name': sample_data['gesture_name'],
                    'user_id': sample_data['user_id'],
                    'timestamp': sample_data['timestamp'],
                    'duration': sample_data['duration']
                })
        
        X = np.array(X_list)  # (N, 100, 5)
        y = np.array(y_list)  # (N,)
        
        # 保存numpy数组 (原始整数ADC值)
        np.save(os.path.join(output_dir, f"{filename}_X.npy"), X.astype(np.int16))  # Save as int16 to save space
        np.save(os.path.join(output_dir, f"{filename}_y.npy"), y)
        
        # 保存元数据
        metadata = {
            'dataset_info': {
                'total_samples': total_samples,
                'sequence_length': self.sequence_length,
                'num_sensors': self.num_sensors,
                'sample_rate': self.sample_rate,
                'gesture_classes': self.gesture_classes,
                'creation_time': datetime.now().isoformat()
            },
            'samples': metadata_list
        }
        
        with open(os.path.join(output_dir, f"{filename}_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 生成数据集报告
        self._generate_dataset_report(X, y, os.path.join(output_dir, f"{filename}_report.txt"))
        
        logger.success(f"数据集已保存到 {output_dir}")
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"数据维度: {X.shape}")
    
    def _generate_dataset_report(self, X: np.ndarray, y: np.ndarray, report_path: str):
        """生成数据集报告"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BSL手势识别数据集报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"数据集维度: {X.shape}\n")
            f.write(f"序列长度: {X.shape[1]}\n")
            f.write(f"传感器数量: {X.shape[2]}\n\n")
            
            f.write("各类别样本数量:\n")
            unique, counts = np.unique(y, return_counts=True)
            for class_id, count in zip(unique, counts):
                gesture_name = self.gesture_classes[class_id]
                f.write(f"  {class_id}: {gesture_name} - {count} 样本\n")
            
            f.write(f"\n总样本数: {len(y)}\n")
            
            # 数据统计
            f.write(f"\n传感器数据统计:\n")
            for sensor_idx in range(X.shape[2]):
                sensor_data = X[:, :, sensor_idx].flatten()
                f.write(f"  传感器 {sensor_idx + 1}: 均值={sensor_data.mean():.2f}, "
                       f"标准差={sensor_data.std():.2f}, "
                       f"范围=[{sensor_data.min():.2f}, {sensor_data.max():.2f}]\n")
    
    def visualize_realtime_data(self, duration: int = 30):
        """
        Realtime visualization of sensor data with enhanced feedback
        
        Args:
            duration: Visualization duration in seconds
        """
        if not self.is_connected:
            logger.error("Arduino not connected")
            return
        
        plt.ion()  # Interactive mode
        fig, axes = plt.subplots(self.num_sensors, 1, figsize=(14, 10))
        fig.suptitle('BSL Gesture Recognition - Raw Sensor Data (No Filtering)', fontsize=16, fontweight='bold')
        
        sensor_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Initialize plots with better styling
        lines = []
        value_texts = []  # For current value display
        
        for i in range(self.num_sensors):
            line, = axes[i].plot([], [], color=colors[i], linewidth=3, alpha=0.8)
            axes[i].set_ylabel(f'{sensor_names[i]}\nValue', fontsize=12, fontweight='bold')
            axes[i].set_xlim(0, 100)  # Shorter window for better visibility
            axes[i].set_ylim(0, 1024)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(labelsize=10)
            
            # Add current value text display
            value_text = axes[i].text(0.02, 0.85, '0', transform=axes[i].transAxes, 
                                    fontsize=12, fontweight='bold', 
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3))
            value_texts.append(value_text)
            lines.append(line)
        
        axes[-1].set_xlabel('Time Steps (Last 100 samples)', fontsize=12)
        
        # Add instructions
        fig.text(0.02, 0.02, 
                "Instructions: Touch sensors to see changes • Press Ctrl+C to stop", 
                fontsize=11, style='italic')
        
        start_time = time.time()
        time_data = deque(maxlen=100)  # Shorter window
        sensor_data_queues = [deque(maxlen=100) for _ in range(self.num_sensors)]
        baseline_values = [None] * self.num_sensors  # Track baseline for change detection
        
        logger.info(f"Starting enhanced data visualization - Duration: {duration} seconds")
        print(f"🔍 Enhanced Visualization Mode - {duration} seconds")
        print("📊 Touch sensors to see real-time changes!")
        print("💡 Look for value changes in the colored boxes")
        
        data_received_count = 0
        sample_rate_counter = 0
        rate_start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                sensor_values = self.read_sensor_data()
                if sensor_values:
                    data_received_count += 1
                    sample_rate_counter += 1
                    current_time = len(time_data)
                    time_data.append(current_time)
                    
                    # Establish baseline values
                    if data_received_count == 10:  # After 10 samples
                        baseline_values = sensor_values.copy()
                        print(f"📏 Baseline established: {[f'{v:.1f}' for v in baseline_values]}")
                    
                    for i, value in enumerate(sensor_values):
                        sensor_data_queues[i].append(value)
                        
                        # Update current value display with change indication
                        if baseline_values and baseline_values[i] is not None:
                            change = abs(value - baseline_values[i])
                            if change > 20:  # Significant change threshold
                                value_texts[i].set_text(f'{value:.1f} (Δ{change:.1f})')
                                value_texts[i].set_bbox(dict(boxstyle="round,pad=0.3", 
                                                          facecolor='yellow', alpha=0.8))
                            else:
                                value_texts[i].set_text(f'{value:.1f}')
                                value_texts[i].set_bbox(dict(boxstyle="round,pad=0.3", 
                                                          facecolor=colors[i], alpha=0.3))
                        else:
                            value_texts[i].set_text(f'{value:.1f}')
                    
                    # Update plots with auto-scaling
                    for i in range(self.num_sensors):
                        if len(time_data) > 1:
                            lines[i].set_data(list(time_data), list(sensor_data_queues[i]))
                            
                            # Auto-scale Y axis for better visibility
                            if len(sensor_data_queues[i]) > 10:
                                y_data = list(sensor_data_queues[i])
                                y_min, y_max = min(y_data), max(y_data)
                                y_range = y_max - y_min
                                if y_range > 5:  # Only scale if there's meaningful variation
                                    axes[i].set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
                                else:
                                    axes[i].set_ylim(y_min - 50, y_max + 50)
                            
                            axes[i].set_xlim(max(0, current_time - 100), current_time + 5)
                    
                    plt.pause(0.01)
                    
                    # Show data reception status and calculate sample rate
                    if data_received_count % 100 == 0:
                        elapsed = time.time() - rate_start_time
                        actual_rate = sample_rate_counter / elapsed if elapsed > 0 else 0
                        print(f"📡 Packets: {data_received_count} | Rate: {actual_rate:.1f} Hz | "
                              f"Time: {int(time.time() - start_time)}s")
                        sample_rate_counter = 0
                        rate_start_time = time.time()
                else:
                    time.sleep(0.001)  # Very small delay if no data
                
        except KeyboardInterrupt:
            logger.info("User interrupted visualization")
        finally:
            plt.ioff()
            plt.close()
            final_elapsed = time.time() - start_time
            actual_avg_rate = data_received_count / final_elapsed if final_elapsed > 0 else 0
            
            print(f"\n📊 Visualization Summary:")
            print(f"   Duration: {final_elapsed:.1f}s")
            print(f"   Total packets: {data_received_count}")
            print(f"   Average rate: {actual_avg_rate:.1f} Hz")
            print(f"   Target rate: 50 Hz")
            
            if data_received_count == 0:
                print("⚠️  No data received! Check Arduino connection and sensor wiring.")
            elif actual_avg_rate < 30:
                print("⚠️  Low sample rate detected. Upload the data collection Arduino code.")
            else:
                print("✅ Data collection working properly!")
    
    def print_data_summary(self):
        """打印数据采集摘要"""
        print("\n" + "="*60)
        print("数据采集摘要")
        print("="*60)
        
        total_samples = 0
        for class_id, samples in self.collected_data.items():
            gesture_name = self.gesture_classes[class_id]
            count = len(samples)
            total_samples += count
            print(f"{gesture_name:15} (类别 {class_id:2d}): {count:3d} 样本")
        
        print("-"*60)
        print(f"总计: {total_samples} 样本")
        print("="*60)


    def interactive_gesture_collection(self, samples_per_gesture: int = 10, user_id: str = "user1"):
        """
        Interactive gesture collection with user guidance
        
        Args:
            samples_per_gesture: Number of samples to collect per gesture
            user_id: User identifier
        """
        if not self.is_connected:
            logger.error("Arduino not connected")
            return False
        
        print("\n" + "="*60)
        print("🤟 BSL GESTURE DATA COLLECTION SYSTEM 🤟")
        print("="*60)
        print(f"👤 User: {user_id}")
        print(f"📊 Samples per gesture: {samples_per_gesture}")
        print(f"📝 Total gestures: {len(self.gesture_classes)}")
        print(f"⏱️  Total samples to collect: {len(self.gesture_classes) * samples_per_gesture}")
        print("="*60)
        
        # Show gesture list
        print("\n🔢 Gesture List:")
        for i, gesture_name in enumerate(self.gesture_classes):
            print(f"   {i:2d}: {gesture_name}")
        
        input("\n👆 Press Enter when ready to start collection...")
        
        total_collected = 0
        successful_gestures = 0
        
        for gesture_id in range(len(self.gesture_classes)):
            gesture_name = self.gesture_classes[gesture_id]
            
            print(f"\n{'='*50}")
            print(f"🎯 COLLECTING GESTURE: {gesture_name.upper()}")
            print(f"📋 Gesture {gesture_id + 1}/{len(self.gesture_classes)}")
            print(f"{'='*50}")
            
            if gesture_id == 0:
                print("💡 Instructions for 'Rest/Neutral':")
                print("   • Keep your hand relaxed and open")
                print("   • Don't touch any sensors")
                print("   • Natural resting position")
            else:
                print(f"💡 Instructions for '{gesture_name}':")
                print(f"   • Make the {gesture_name} gesture clearly")
                print("   • Hold the position steadily for 2 seconds")
                print("   • Touch the appropriate sensors")
            
            gesture_successful_samples = 0
            
            for sample_num in range(samples_per_gesture):
                print(f"\n📋 Sample {sample_num + 1}/{samples_per_gesture}")
                
                # Ask user if ready
                user_input = input("Press Enter to collect (or 's' to skip, 'q' to quit): ").strip().lower()
                
                if user_input == 'q':
                    print("❌ Collection cancelled by user")
                    return False
                elif user_input == 's':
                    print("⏭️  Sample skipped")
                    continue
                
                # Collect the gesture
                success = self.collect_gesture_sequence(gesture_id, user_id)
                
                if success:
                    total_collected += 1
                    gesture_successful_samples += 1
                    print(f"✅ Sample {sample_num + 1} collected successfully!")
                    print(f"📊 Progress: {total_collected}/{len(self.gesture_classes) * samples_per_gesture}")
                else:
                    print(f"❌ Sample {sample_num + 1} failed")
                    retry = input("   Retry this sample? (y/n): ").strip().lower()
                    if retry == 'y':
                        sample_num -= 1  # Retry current sample
                
                # Short break between samples
                if sample_num < samples_per_gesture - 1:
                    time.sleep(1)
            
            if gesture_successful_samples > 0:
                successful_gestures += 1
                print(f"✅ Gesture '{gesture_name}' completed: {gesture_successful_samples}/{samples_per_gesture} samples")
            else:
                print(f"❌ No samples collected for '{gesture_name}'")
            
            # Break between gestures
            if gesture_id < len(self.gesture_classes) - 1:
                print(f"\n⏸️  Take a short break...")
                input("   Press Enter to continue to next gesture...")
        
        # Collection summary
        print(f"\n{'='*60}")
        print("📊 COLLECTION SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Total samples collected: {total_collected}")
        print(f"🎯 Successful gestures: {successful_gestures}/{len(self.gesture_classes)}")
        print(f"📈 Success rate: {(total_collected / (len(self.gesture_classes) * samples_per_gesture) * 100):.1f}%")
        
        if total_collected > 0:
            self.print_data_summary()
            
            # Auto-save option
            save_data = input("\n💾 Save collected data? (y/n): ").strip().lower()
            if save_data == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"bsl_gestures_{user_id}_{timestamp}"
                self.save_dataset("datasets/raw", filename)
                print(f"✅ Data saved as: {filename}")
        
        return total_collected > 0


def main():
    """Main function - Interactive data collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BSL Gesture Data Collector')
    parser.add_argument('--port', default='/dev/cu.usbmodem101', help='Arduino serial port')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baud rate')
    parser.add_argument('--output', default='datasets/raw', help='Output directory')
    parser.add_argument('--samples', type=int, default=10, help='Samples per gesture')
    parser.add_argument('--user', default='user1', help='User ID')
    args = parser.parse_args()
    
    # Create data collector
    collector = BSLDataCollector(port=args.port, baudrate=args.baudrate)
    
    # Connect to Arduino
    if not collector.connect():
        print("❌ Failed to connect to Arduino. Please check:")
        print("   1. Arduino is connected to the correct port")
        print("   2. Upload the data collection firmware")
        print("   3. Check sensor connections")
        return 1
    
    try:
        print("\n🔧 BSL Data Collection Options:")
        print("1. 👀 Visualize sensor data (test sensors)")
        print("2. 📊 Interactive gesture collection")
        print("3. 🧪 Quick sensor test")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            duration = input("Visualization duration in seconds (default 30): ").strip()
            duration = int(duration) if duration.isdigit() else 30
            collector.visualize_realtime_data(duration)
            
        elif choice == '2':
            print("Starting interactive gesture collection...")
            collector.interactive_gesture_collection(args.samples, args.user)
            
        elif choice == '3':
            print("Quick sensor test - touch each sensor to see changes:")
            collector.visualize_realtime_data(15)
            
        else:
            print("Invalid option. Starting visualization...")
            collector.visualize_realtime_data(30)
            
    except KeyboardInterrupt:
        print("\n👋 Collection interrupted by user")
    finally:
        collector.disconnect()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main() 