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
import time
import os
import csv
from typing import List, Optional
from datetime import datetime
from loguru import logger


class BSLDataCollector:
    """极简BSL手势数据采集器"""
    
    def __init__(self, port: str = '/dev/cu.usbmodem1101', baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self.num_sensors = 5
        self.gesture_classes = {
            0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
            5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
        }
        logger.info("极简数据采集器已初始化")
    
    def connect(self) -> bool:
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            if self.serial_conn.is_open:
                # 关键修复：主动、循环地清空缓冲区，确保丢弃所有初始的陈旧数据
                time.sleep(0.1) # 等待数据到达
                while self.serial_conn.in_waiting > 0:
                    self.serial_conn.read(self.serial_conn.in_waiting)
                logger.success(f"成功连接到Arduino: {self.port}")
                print(f"✅ 已连接到Arduino: {self.port}")
                self.is_connected = True
                return True
            else:
                logger.error(f"无法打开串口: {self.port}")
                return False
        except serial.SerialException as e:
            logger.error(f"串口连接失败: {e}")
            return False
    
    def disconnect(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.is_connected = False
            logger.info("Arduino已断开连接")
    
    def read_sensor_data(self) -> Optional[List[int]]:
        if not self.is_connected or not self.serial_conn:
            return None
        try:
            line = self.serial_conn.readline().decode('utf-8').strip()
            if line and '\t' in line:
                parts = line.split('\t')
                if len(parts) == 5:
                    return [int(p) for p in parts]
        except (ValueError, UnicodeDecodeError, IndexError) as e:
            logger.warning(f"数据解析错误: {e}")
        return None

    def test_sensor_data(self, duration: int = 15):
        """测试传感器，实时打印并保存原始数据到CSV"""
        if not self.is_connected:
            print("❌ Arduino未连接")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "datasets/csv"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"test_data_{timestamp}.csv"
        csv_path = os.path.join(output_dir, filename)
        
        print(f"\n🧪 测试模式: 正在运行 {duration} 秒...")
        print(f"💾 数据将保存至: {csv_path}")
        
        # 清空串口缓冲区，确保采集的是实时数据
        self.serial_conn.reset_input_buffer()
        start_time = time.time()
        sample_count = 0

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp_ms', 'thumb', 'index', 'middle', 'ring', 'pinky'])

                print("\n--- 实时传感器数据 ---")
                while time.time() - start_time < duration:
                    sensor_data = self.read_sensor_data()
                    if sensor_data:
                        sample_count += 1
                        relative_timestamp = (time.time() - start_time) * 1000
                        writer.writerow([f"{relative_timestamp:.3f}"] + sensor_data)
                        print(f"\r{sensor_data}   ", end="")
        except IOError as e:
            logger.error(f"文件写入失败: {e}")
            print(f"❌ 错误: 无法写入文件 {csv_path}")
            return
        
        print(f"\n\n✅ 测试完成, 已采集 {sample_count} 个样本。")
        print(f"📁 数据已保存: {csv_path}")

    def manual_gesture_collection(self, user_id: str):
        """手动逐一采集手势0-9, 每个手势采集10次"""
        if not self.is_connected:
            print("❌ Arduino未连接")
            return
        
        output_dir = "datasets/gesture_csv"
        os.makedirs(output_dir, exist_ok=True)
        num_samples_per_gesture = 10
        
        print(f"\n🎯 用户 {user_id} 的手势采集开始")
        print(f"每个手势将连续采集 {num_samples_per_gesture} 次，每次2秒")
        print("=" * 50)
        
        for gesture_id in range(10): # 0-9
            gesture_name = self.gesture_classes[gesture_id]
            print(f"\n下一个手势: ({gesture_id}) {gesture_name}")
            
            for sample_num in range(num_samples_per_gesture):
                input(f"  -> 请按 [Enter] 键，开始采集第 {sample_num + 1}/{num_samples_per_gesture} 个样本...")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"user_{user_id}_gesture_{gesture_id}_{gesture_name}_sample_{sample_num+1}_{timestamp}.csv"
                csv_path = os.path.join(output_dir, filename)
                
                print(f"    🎬 正在采集...")
                
                # 关键修复：在每次采集前，主动、循环地清空缓冲区
                time.sleep(0.1) # 等待数据到达
                while self.serial_conn.in_waiting > 0:
                    self.serial_conn.read(self.serial_conn.in_waiting)

                start_time = time.time()
                sample_count = 0
                
                try:
                    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['# BSL Gesture Data'])
                        writer.writerow([f'# User: {user_id}, Gesture: {gesture_id} ({gesture_name}), Sample: {sample_num+1}'])
                        writer.writerow(['timestamp_ms', 'thumb', 'index', 'middle', 'ring', 'pinky'])
                        
                        while time.time() - start_time < 2: # 采集2秒
                            sensor_data = self.read_sensor_data()
                            if sensor_data:
                                sample_count += 1
                                relative_timestamp = (time.time() - start_time) * 1000
                                writer.writerow([f"{relative_timestamp:.3f}"] + sensor_data)
                except IOError as e:
                    logger.error(f"文件写入失败: {e}")
                    print(f"❌ 错误: 无法写入文件 {csv_path}")
                    continue

                print(f"    ✅ 完成! (样本 {sample_num + 1}/{num_samples_per_gesture}) 已采集 {sample_count} 个数据点。")
        
        print("\n🎉 所有手势采集完成!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='极简BSL手势数据采集器')
    parser.add_argument('mode', choices=['test', 'auto'], help='运行模式: test(测试传感器) 或 auto(手动采集手势)')
    parser.add_argument('--port', default='/dev/cu.usbmodem1101', help='Arduino串口')
    parser.add_argument('--duration', type=int, default=15, help='test模式持续时间(秒)')
    
    args = parser.parse_args()
    
    collector = BSLDataCollector(port=args.port)
    
    if not collector.connect():
        return
    
    try:
        if args.mode == 'test':
            collector.test_sensor_data(args.duration)
        elif args.mode == 'auto':
            while True:
                user_id = input("👤 请输入用户ID: ").strip()
                if user_id:
                    break
                print("❌ 用户ID不能为空。")
            collector.manual_gesture_collection(user_id)
    except KeyboardInterrupt:
        print("\n\n🛑 操作被用户中断。")
    except Exception as e:
        logger.error(f"发生未知错误: {e}")
        print(f"\n❌ 发生未知错误: {e}")
    finally:
        collector.disconnect()
        print("\n👋 程序结束。")

if __name__ == "__main__":
    main() 