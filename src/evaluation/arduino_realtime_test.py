#!/usr/bin/env python3
"""
Arduino Real-time Gesture Recognition Test.
Tests the real-time performance of gesture recognition on Arduino.
"""

import os
import sys
import time
import json
import serial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

class ArduinoRealtimeTester:
    """Test real-time gesture recognition on Arduino."""
    
    def __init__(self, port=None, baudrate=115200, output_dir="outputs/arduino_realtime_test"):
        self.port = port
        self.baudrate = baudrate
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Test configuration
        self.test_duration = 120  # 2 minutes per test
        self.num_rounds = 3       # 3 rounds per gesture
        self.gestures_to_test = [0, 1, 2, 3, 4, 5]  # Test gestures 0-5
        
        # Results storage
        self.results = {
            'test_config': {},
            'round_results': [],
            'performance_metrics': {},
            'error_analysis': {}
        }
    
    def find_arduino_port(self):
        """Find Arduino port automatically."""
        import glob
        
        # Common port patterns
        if sys.platform.startswith('win'):
            ports = glob.glob('COM*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/cu.usbmodem*')
        else:
            ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        
        if not ports:
            print("❌ No Arduino ports found!")
            return None
        
        # Try to connect to the first available port
        for port in ports:
            try:
                with serial.Serial(port, self.baudrate, timeout=1) as ser:
                    time.sleep(2)
                    if ser.in_waiting > 0:
                        response = ser.readline().decode('utf-8').strip()
                        if "System initialized" in response or "waiting for gesture" in response:
                            print(f"✅ Found Arduino on {port}")
                            return port
            except Exception as e:
                print(f"⚠️ Failed to connect to {port}: {e}")
                continue
        
        print("❌ No Arduino with real-time test firmware found!")
        return None
    
    def connect_to_arduino(self):
        """Connect to Arduino."""
        if not self.port:
            self.port = self.find_arduino_port()
        
        if not self.port:
            return None
        
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            
            # Clear any existing data
            self.ser.reset_input_buffer()
            
            print(f"✅ Connected to Arduino on {self.port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Arduino: {e}")
            return False
    
    def wait_for_arduino_ready(self):
        """Wait for Arduino to be ready."""
        print("⏳ Waiting for Arduino to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < 10:  # Wait up to 10 seconds
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                if "System initialized" in line or "waiting for gesture" in line:
                    print("✅ Arduino is ready!")
                    return True
            time.sleep(0.1)
        
        print("❌ Arduino not responding")
        return False
    
    def run_single_test_round(self, gesture_id, round_num):
        """Run a single test round for a specific gesture."""
        print(f"\n🎯 Testing Gesture {gesture_id} - Round {round_num + 1}")
        print("=" * 50)
        
        # Test configuration
        test_start_time = time.time()
        detections = []
        errors = []
        latencies = []
        
        # Clear any existing data
        self.ser.reset_input_buffer()
        
        # Wait for Arduino to be ready
        if not self.wait_for_arduino_ready():
            return None
        
        print(f"📱 Please perform Gesture {gesture_id} for {self.test_duration} seconds...")
        print("   (The system will automatically detect and classify your gestures)")
        
        # Monitor Arduino output during test
        while time.time() - test_start_time < self.test_duration:
            if self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line:
                        print(f"   Arduino: {line}")
                        
                        # Parse detection results
                        if "✅ Detected:" in line:
                            # Extract gesture and confidence
                            parts = line.split("✅ Detected: ")[1].split(" (")
                            detected_gesture = parts[0].strip()
                            confidence = float(parts[1].split("%")[0])
                            
                            detection_time = time.time() - test_start_time
                            detections.append({
                                'time': detection_time,
                                'detected_gesture': detected_gesture,
                                'confidence': confidence,
                                'expected_gesture': gesture_id
                            })
                            
                            # Calculate accuracy
                            is_correct = self._is_gesture_correct(detected_gesture, gesture_id)
                            
                            print(f"   📊 Detection: {detected_gesture} ({confidence:.1f}%) - {'✅ Correct' if is_correct else '❌ Wrong'}")
                        
                        elif "⚠️ No gesture recognized" in line:
                            detection_time = time.time() - test_start_time
                            errors.append({
                                'time': detection_time,
                                'error_type': 'no_recognition',
                                'expected_gesture': gesture_id
                            })
                            print("   ⚠️ No gesture recognized")
                        
                        elif "Change detected" in line or "Movement finished" in line:
                            # These are debug messages, not errors
                            pass
                        
                except Exception as e:
                    print(f"   ⚠️ Error reading Arduino: {e}")
                    errors.append({
                        'time': time.time() - test_start_time,
                        'error_type': 'communication_error',
                        'expected_gesture': gesture_id
                    })
            
            time.sleep(0.01)  # Small delay to prevent busy waiting
        
        # Calculate round metrics
        round_metrics = self._calculate_round_metrics(detections, errors, gesture_id)
        
        print(f"\n📊 Round {round_num + 1} Results:")
        print(f"   Detections: {len(detections)}")
        print(f"   Correct: {round_metrics['correct_detections']}")
        print(f"   Accuracy: {round_metrics['accuracy']:.1f}%")
        print(f"   Average Confidence: {round_metrics['avg_confidence']:.1f}%")
        print(f"   Errors: {len(errors)}")
        
        return {
            'gesture_id': gesture_id,
            'round_num': round_num,
            'detections': detections,
            'errors': errors,
            'metrics': round_metrics,
            'test_duration': self.test_duration
        }
    
    def _is_gesture_correct(self, detected_gesture, expected_gesture):
        """Check if detected gesture matches expected gesture."""
        gesture_names = {
            'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 
            'Four': 4, 'Five': 5, 'Six': 6, 'Seven': 7, 
            'Eight': 8, 'Nine': 9, 'Static': 10
        }
        
        detected_id = gesture_names.get(detected_gesture, -1)
        return detected_id == expected_gesture
    
    def _calculate_round_metrics(self, detections, errors, expected_gesture):
        """Calculate metrics for a single test round."""
        if not detections:
            return {
                'correct_detections': 0,
                'total_detections': 0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'detection_rate': 0.0
            }
        
        correct_detections = sum(1 for d in detections if self._is_gesture_correct(d['detected_gesture'], expected_gesture))
        total_detections = len(detections)
        accuracy = (correct_detections / total_detections * 100) if total_detections > 0 else 0.0
        avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0.0
        
        # Detection rate (detections per minute)
        detection_rate = len(detections) / (self.test_duration / 60)
        
        return {
            'correct_detections': correct_detections,
            'total_detections': total_detections,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'detection_rate': detection_rate
        }
    
    def run_comprehensive_test(self):
        """Run comprehensive real-time test for all gestures."""
        print("🚀 Starting Arduino Real-time Gesture Recognition Test")
        print("=" * 60)
        
        # Connect to Arduino
        if not self.connect_to_arduino():
            return False
        
        # Test configuration
        self.results['test_config'] = {
            'port': self.port,
            'baudrate': self.baudrate,
            'test_duration': self.test_duration,
            'num_rounds': self.num_rounds,
            'gestures_to_test': self.gestures_to_test,
            'timestamp': datetime.now().isoformat()
        }
        
        # Run tests for each gesture
        for gesture_id in self.gestures_to_test:
            print(f"\n🎯 Testing Gesture {gesture_id}")
            print("-" * 40)
            
            for round_num in range(self.num_rounds):
                round_result = self.run_single_test_round(gesture_id, round_num)
                if round_result:
                    self.results['round_results'].append(round_result)
                
                # Brief pause between rounds
                if round_num < self.num_rounds - 1:
                    print("⏸️  Pausing 5 seconds before next round...")
                    time.sleep(5)
            
            # Brief pause between gestures
            if gesture_id != self.gestures_to_test[-1]:
                print("⏸️  Pausing 10 seconds before next gesture...")
                time.sleep(10)
        
        # Calculate overall performance metrics
        self._calculate_overall_metrics()
        
        # Save results
        self._save_results()
        
        # Close Arduino connection
        if hasattr(self, 'ser'):
            self.ser.close()
        
        return True
    
    def _calculate_overall_metrics(self):
        """Calculate overall performance metrics."""
        print("\n📊 Calculating overall performance metrics...")
        
        # Aggregate all detections
        all_detections = []
        all_errors = []
        
        for round_result in self.results['round_results']:
            all_detections.extend(round_result['detections'])
            all_errors.extend(round_result['errors'])
        
        # Overall accuracy
        correct_detections = sum(1 for d in all_detections 
                               if self._is_gesture_correct(d['detected_gesture'], d['expected_gesture']))
        total_detections = len(all_detections)
        overall_accuracy = (correct_detections / total_detections * 100) if total_detections > 0 else 0.0
        
        # Per-gesture accuracy
        gesture_accuracies = {}
        for gesture_id in self.gestures_to_test:
            gesture_detections = [d for d in all_detections if d['expected_gesture'] == gesture_id]
            if gesture_detections:
                correct = sum(1 for d in gesture_detections 
                            if self._is_gesture_correct(d['detected_gesture'], gesture_id))
                gesture_accuracies[gesture_id] = (correct / len(gesture_detections) * 100)
            else:
                gesture_accuracies[gesture_id] = 0.0
        
        # Error analysis
        error_types = {}
        for error in all_errors:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Performance metrics
        self.results['performance_metrics'] = {
            'overall_accuracy': overall_accuracy,
            'total_detections': total_detections,
            'correct_detections': correct_detections,
            'total_errors': len(all_errors),
            'avg_confidence': np.mean([d['confidence'] for d in all_detections]) if all_detections else 0.0,
            'detection_rate_per_minute': len(all_detections) / (len(self.results['round_results']) * self.test_duration / 60),
            'gesture_accuracies': gesture_accuracies,
            'error_types': error_types
        }
        
        print(f"✅ Overall Accuracy: {overall_accuracy:.1f}%")
        print(f"📊 Total Detections: {total_detections}")
        print(f"❌ Total Errors: {len(all_errors)}")
        print(f"🎯 Average Confidence: {self.results['performance_metrics']['avg_confidence']:.1f}%")
    
    def _save_results(self):
        """Save test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_path = os.path.join(self.output_dir, f"realtime_test_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create summary CSV
        summary_data = []
        for round_result in self.results['round_results']:
            summary_data.append({
                'Gesture_ID': round_result['gesture_id'],
                'Round': round_result['round_num'] + 1,
                'Total_Detections': round_result['metrics']['total_detections'],
                'Correct_Detections': round_result['metrics']['correct_detections'],
                'Accuracy_Percent': round_result['metrics']['accuracy'],
                'Avg_Confidence': round_result['metrics']['avg_confidence'],
                'Detection_Rate_Per_Min': round_result['metrics']['detection_rate'],
                'Errors': len(round_result['errors'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, f"realtime_test_summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Create performance visualization
        self._create_performance_plots(timestamp)
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"📄 Detailed results: {results_path}")
        print(f"📊 Summary CSV: {summary_path}")
    
    def _create_performance_plots(self, timestamp):
        """Create performance visualization plots."""
        print("📈 Creating performance plots...")
        
        # Prepare data for plotting
        gesture_accuracies = self.results['performance_metrics']['gesture_accuracies']
        gestures = list(gesture_accuracies.keys())
        accuracies = list(gesture_accuracies.values())
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Arduino Real-time Gesture Recognition Performance', fontsize=16, fontweight='bold')
        
        # Plot 1: Per-gesture accuracy
        bars = ax1.bar(gestures, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Accuracy by Gesture')
        ax1.set_xlabel('Gesture ID')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Overall performance metrics
        metrics = self.results['performance_metrics']
        metric_names = ['Overall Accuracy', 'Detection Rate', 'Avg Confidence']
        metric_values = [
            metrics['overall_accuracy'],
            metrics['detection_rate_per_minute'],
            metrics['avg_confidence']
        ]
        
        bars2 = ax2.bar(metric_names, metric_values, color='lightcoral', alpha=0.7)
        ax2.set_title('Overall Performance Metrics')
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars2, metric_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}', ha='center', va='bottom')
        
        # Plot 3: Error analysis
        error_types = metrics['error_types']
        if error_types:
            error_names = list(error_types.keys())
            error_counts = list(error_types.values())
            
            ax3.pie(error_counts, labels=error_names, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Error Types Distribution')
        else:
            ax3.text(0.5, 0.5, 'No Errors Recorded', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Error Types Distribution')
        
        # Plot 4: Detection timeline (sample from first round)
        if self.results['round_results']:
            first_round = self.results['round_results'][0]
            if first_round['detections']:
                times = [d['time'] for d in first_round['detections']]
                confidences = [d['confidence'] for d in first_round['detections']]
                
                ax4.scatter(times, confidences, alpha=0.6, color='green')
                ax4.set_title(f'Detection Timeline - Gesture {first_round["gesture_id"]} (Round 1)')
                ax4.set_xlabel('Time (seconds)')
                ax4.set_ylabel('Confidence (%)')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No Detections in First Round', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Detection Timeline')
        else:
            ax4.text(0.5, 0.5, 'No Test Data Available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Detection Timeline')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f"realtime_performance_plots_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Performance plots saved: {plot_path}")

def main():
    """Main function to run Arduino real-time test."""
    print("=" * 60)
    print("ARDUINO REAL-TIME GESTURE RECOGNITION TEST")
    print("=" * 60)
    
    # Initialize tester
    tester = ArduinoRealtimeTester()
    
    # Run comprehensive test
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n" + "=" * 60)
        print("REAL-TIME TEST COMPLETE")
        print("=" * 60)
        
        # Print summary
        metrics = tester.results['performance_metrics']
        print(f"🎯 Overall Accuracy: {metrics['overall_accuracy']:.1f}%")
        print(f"📊 Total Detections: {metrics['total_detections']}")
        print(f"❌ Total Errors: {metrics['total_errors']}")
        print(f"🎯 Average Confidence: {metrics['avg_confidence']:.1f}%")
        print(f"📈 Detection Rate: {metrics['detection_rate_per_minute']:.1f} per minute")
        
        print(f"\n📁 Results saved to: {tester.output_dir}")
    else:
        print("\n❌ Real-time test failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
