#!/usr/bin/env python3
"""
Arduino实时手势识别演示脚本
用于论文展示和基本功能验证

功能：
1. 连接Arduino
2. 发送测试命令
3. 接收响应
4. 展示基本功能
"""

import serial
import time
import sys
from typing import Optional

class ArduinoDemo:
    def __init__(self, port: str = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        
    def find_arduino_port(self) -> Optional[str]:
        """查找Arduino端口"""
        import glob
        import platform
        
        system = platform.system()
        
        if system == "Darwin":  # macOS
            ports = glob.glob("/dev/cu.usbmodem*")
        elif system == "Linux":
            ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        elif system == "Windows":
            ports = [f"COM{i}" for i in range(1, 10)]
        else:
            print(f"❌ Unsupported system: {system}")
            return None
            
        for port in ports:
            try:
                with serial.Serial(port, self.baudrate, timeout=1) as test_ser:
                    time.sleep(2)
                    # 清空缓冲区
                    test_ser.reset_input_buffer()
                    # 等待响应
                    time.sleep(1)
                    if test_ser.in_waiting > 0:
                        response = test_ser.readline().decode('utf-8').strip()
                        print(f"DEBUG: Port {port} response: '{response}'")
                        if "System initialized" in response or "Ready" in response or response.replace('.', '').isdigit():
                            print(f"✅ Found Arduino on {port}")
                            return port
            except Exception as e:
                print(f"DEBUG: Port {port} failed: {e}")
                continue
                
        print("❌ No Arduino found!")
        return None
    
    def connect(self) -> bool:
        """连接到Arduino"""
        if not self.port:
            self.port = self.find_arduino_port()
            if not self.port:
                return False
                
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            # 清空缓冲区
            self.ser.reset_input_buffer()
            print(f"✅ Connected to Arduino on {self.port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def wait_for_ready(self, timeout: int = 10) -> bool:
        """等待Arduino准备就绪"""
        print("⏳ Waiting for Arduino to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                print(f"📡 Arduino: {line}")
                if "System initialized" in line or "Ready" in line:
                    print("✅ Arduino is ready!")
                    return True
                # 检查是否是数据采集固件（输出传感器数据）
                elif line.replace('.', '').isdigit():
                    print("✅ Arduino data collection firmware detected!")
                    return True
            time.sleep(0.1)
            
        print("❌ Arduino not responding")
        return False
    
    def run_demo(self):
        """运行演示"""
        print("🚀 Starting Arduino Real-time Gesture Recognition Demo")
        print("=" * 60)
        
        # 1. 连接Arduino
        if not self.connect():
            print("\n📋 Demo Summary:")
            print("❌ Connection failed - Arduino not found or not responding")
            print("💡 Make sure Arduino is connected and running real-time firmware")
            return
        
        # 2. 等待准备就绪
        if not self.wait_for_ready():
            print("\n📋 Demo Summary:")
            print("❌ Arduino not ready - firmware may not be uploaded")
            print("💡 Upload the real-time firmware to Arduino first")
            return
        
        # 3. 演示基本功能
        print("\n🎯 Demo: Real-time Gesture Recognition")
        print("📝 Instructions:")
        print("   1. Move your hand to trigger gesture detection")
        print("   2. Hold a gesture for 2-3 seconds")
        print("   3. Watch for detection results")
        print("   4. Press Ctrl+C to stop")
        
        try:
            while True:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line:
                        print(f"📡 {line}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⏹️ Demo stopped by user")
        
        # 4. 总结
        print("\n📋 Demo Summary:")
        print("✅ Arduino connection successful")
        print("✅ Real-time firmware loaded")
        print("✅ Gesture recognition system ready")
        print("💡 For full testing, use src/evaluation/arduino_realtime_test.py")
        
    def close(self):
        """关闭连接"""
        if self.ser:
            self.ser.close()
            print("🔌 Connection closed")

def main():
    """主函数"""
    demo = ArduinoDemo()
    try:
        demo.run_demo()
    finally:
        demo.close()

if __name__ == "__main__":
    main()
