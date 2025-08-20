#!/usr/bin/env python3
"""
Quick Arduino Real-time Test.
Simple test to verify Arduino real-time gesture recognition is working.
"""

import os
import sys
import time
import serial
import glob
from datetime import datetime

def find_arduino_port():
    """Find Arduino port automatically."""
    if sys.platform.startswith('win'):
        ports = glob.glob('COM*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/cu.usbmodem*')
    else:
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    
    return ports[0] if ports else None

def test_arduino_connection():
    """Test basic Arduino connection."""
    print("🔍 Looking for Arduino...")
    
    port = find_arduino_port()
    if not port:
        print("❌ No Arduino found!")
        return False
    
    print(f"✅ Found Arduino on {port}")
    
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)
        
        # Clear buffer
        ser.reset_input_buffer()
        
        # Wait for initialization message
        print("⏳ Waiting for Arduino initialization...")
        start_time = time.time()
        
        while time.time() - start_time < 10:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                print(f"   Arduino: {line}")
                
                if "System initialized" in line or "waiting for gesture" in line:
                    print("✅ Arduino is ready for real-time testing!")
                    ser.close()
                    return True
        
        print("❌ Arduino not responding with expected messages")
        ser.close()
        return False
        
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False

def quick_realtime_test():
    """Run a quick real-time test."""
    print("🚀 Starting Quick Arduino Real-time Test")
    print("=" * 50)
    
    # Test connection
    if not test_arduino_connection():
        return False
    
    print("\n📋 Test Instructions:")
    print("1. Upload Real_time_test.ino to your Arduino")
    print("2. Connect sensors to A0-A4")
    print("3. Run this script again to start the test")
    print("4. Perform gestures and watch the results")
    
    # Ask user if they want to proceed
    response = input("\n🤔 Ready to start real-time test? (y/n): ")
    if response.lower() != 'y':
        print("👋 Test cancelled")
        return True
    
    # Connect for real-time test
    port = find_arduino_port()
    if not port:
        print("❌ Arduino not found!")
        return False
    
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        
        print("\n🎯 Real-time Test Started!")
        print("📱 Perform gestures and watch the results...")
        print("⏹️  Press Ctrl+C to stop")
        print("-" * 50)
        
        # Monitor Arduino output
        start_time = time.time()
        detections = 0
        errors = 0
        
        while True:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        print(f"   {line}")
                        
                        if "✅ Detected:" in line:
                            detections += 1
                        elif "⚠️ No gesture recognized" in line:
                            errors += 1
                        
                        # Show stats every 10 detections
                        if detections % 10 == 0 and detections > 0:
                            elapsed = time.time() - start_time
                            rate = detections / (elapsed / 60)
                            print(f"   📊 Stats: {detections} detections, {errors} errors, {rate:.1f}/min")
                
                except Exception as e:
                    print(f"   ⚠️ Error: {e}")
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Test stopped by user")
        elapsed = time.time() - start_time
        rate = detections / (elapsed / 60) if elapsed > 0 else 0
        
        print(f"\n📊 Final Stats:")
        print(f"   Total Detections: {detections}")
        print(f"   Total Errors: {errors}")
        print(f"   Test Duration: {elapsed:.1f} seconds")
        print(f"   Detection Rate: {rate:.1f} per minute")
        
        ser.close()
        return True
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function."""
    print("=" * 50)
    print("QUICK ARDUINO REAL-TIME TEST")
    print("=" * 50)
    
    success = quick_realtime_test()
    
    if success:
        print("\n✅ Quick test completed successfully!")
    else:
        print("\n❌ Quick test failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
