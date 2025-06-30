#!/usr/bin/env python3
"""
BSL手势识别系统 - 主启动脚本

该脚本提供统一的命令行界面来执行项目的各种功能：
- 数据采集
- 数据预处理
- 模型训练
- 模型评估
- 实时推理

作者: Lambert Yang
版本: 1.0
"""

import argparse
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger


def setup_logging():
    """设置日志"""
    logger.remove()  # 移除默认处理器
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )


def data_collection(args):
    """Data collection mode"""
    logger.info("Starting data collection mode...")
    
    try:
        from src.data.data_collector import BSLDataCollector
        
        collector = BSLDataCollector(port=args.port, baudrate=args.baudrate)
        
        if not collector.connect():
            logger.error("Arduino connection failed")
            print("❌ Failed to connect to Arduino. Please check:")
            print("   1. Arduino is connected to the correct port")
            print("   2. Upload the data collection firmware to Arduino")
            print("   3. Check sensor connections")
            return 1
        
        try:
            if args.visualize:
                # Real-time visualization mode
                print(f"🔍 Starting sensor visualization for {args.duration} seconds...")
                collector.visualize_realtime_data(args.duration)
            elif args.csv:
                # CSV data collection mode
                print(f"📊 Starting CSV data collection for {args.duration} seconds...")
                csv_path = collector.collect_and_save_csv(
                    duration=args.duration,
                    output_dir=args.output_dir,
                    filename=args.filename
                )
                if csv_path:
                    print(f"✅ CSV file saved: {csv_path}")
                else:
                    print("❌ CSV collection failed")
            elif args.auto:
                # Auto gesture collection mode
                print(f"🤖 Starting auto gesture collection mode...")
                gestures = args.gestures if args.gestures else [0, 1, 2]  # 默认采集0,1,2
                output_dir = args.output_dir.replace('csv', 'gesture_csv') if args.output_dir == 'datasets/csv' else args.output_dir
                
                saved_files = collector.auto_gesture_collection(
                    user_id=args.user_id,
                    gestures=gestures,
                    duration_per_gesture=args.gesture_duration,
                    output_dir=output_dir
                )
                if saved_files:
                    print(f"✅ Auto collection completed! {len(saved_files)} files saved")
                else:
                    print("❌ Auto collection failed")
            elif hasattr(args, 'interactive') and args.interactive:
                # Interactive gesture collection mode
                print("🤟 Starting interactive gesture collection...")
                samples = getattr(args, 'samples', 10)
                user_id = getattr(args, 'user_id', 'user1')
                collector.interactive_gesture_collection(samples, user_id)
            else:
                # Default: show options
                print("\n🔧 BSL Data Collection Options:")
                print("1. 👀 Visualize sensor data (test sensors)")
                print("2. 📊 Collect data and save as CSV")
                print("3. 🤖 Auto gesture collection (按手势分类采集)")
                print("4. 🤟 Interactive gesture collection")
                print("5. 🧪 Quick sensor test")
                
                choice = input("\nSelect option (1-5): ").strip()
                
                if choice == '1':
                    duration = args.duration
                    collector.visualize_realtime_data(duration)
                elif choice == '2':
                    print(f"Starting CSV data collection for {args.duration} seconds...")
                    csv_path = collector.collect_and_save_csv(args.duration)
                    if csv_path:
                        print(f"✅ CSV file saved: {csv_path}")
                elif choice == '3':
                    print("Starting auto gesture collection...")
                    user_id = input("请输入用户ID (如 001): ").strip() or "001"
                    gestures_input = input("请输入要采集的手势ID，用空格分隔 (默认 0 1 2): ").strip()
                    if gestures_input:
                        gestures = [int(x) for x in gestures_input.split()]
                    else:
                        gestures = [0, 1, 2]
                    
                    saved_files = collector.auto_gesture_collection(
                        user_id=user_id,
                        gestures=gestures,
                        duration_per_gesture=2,
                        output_dir="datasets/gesture_csv"
                    )
                    if saved_files:
                        print(f"✅ Auto collection completed! {len(saved_files)} files saved")
                elif choice == '4':
                    print("Starting interactive gesture collection...")
                    collector.interactive_gesture_collection(10, 'user1')
                elif choice == '5':
                    print("Quick sensor test - touch each sensor to see changes:")
                    collector.visualize_realtime_data(15)
                else:
                    print("Invalid option. Starting visualization...")
                    collector.visualize_realtime_data(args.duration)
        
        finally:
            collector.disconnect()
            
        logger.success("Data collection completed!")
        return 0
        
    except ImportError as e:
        logger.error(f"Module import failed: {e}")
        print("❌ Make sure you have installed all required Python packages")
        return 1
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return 1


def data_preprocessing(args):
    """数据预处理模式"""
    logger.info("启动数据预处理模式...")
    
    try:
        from data.data_preprocessor import BSLDataPreprocessor
        
        preprocessor = BSLDataPreprocessor()
        
        output_prefix = preprocessor.create_processed_dataset(
            data_dir=args.input_dir,
            filename_prefix=args.input_prefix,
            output_dir=args.output_dir,
            augment_factor=args.augment_factor,
            normalize_method=args.normalize_method
        )
        
        if args.visualize:
            X, y, _ = preprocessor.load_dataset(args.input_dir, args.input_prefix)
            save_path = os.path.join(args.output_dir, f"{output_prefix}_distribution.png")
            preprocessor.visualize_data_distribution(X, y, save_path)
        
        logger.success("数据预处理完成!")
        return 0
        
    except Exception as e:
        logger.error(f"数据预处理失败: {e}")
        return 1


def model_training(args):
    """模型训练模式"""
    logger.info("启动模型训练模式...")
    
    try:
        # 这里应该调用训练脚本
        import subprocess
        
        cmd = [
            sys.executable, "src/training/train_models.py",
            "--model_type", args.model_type,
            "--data_dir", args.data_dir,
            "--data_prefix", args.data_prefix,
            "--output_dir", args.output_dir
        ]
        
        if args.epochs:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.batch_size:
            cmd.extend(["--batch_size", str(args.batch_size)])
        if args.learning_rate:
            cmd.extend(["--learning_rate", str(args.learning_rate)])
        if args.visualize:
            cmd.append("--visualize")
        
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
        
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        return 1


def model_evaluation(args):
    """模型评估模式"""
    logger.info("启动模型评估模式...")
    
    try:
        # 这里可以添加模型评估逻辑
        logger.info("模型评估功能开发中...")
        return 0
        
    except Exception as e:
        logger.error(f"模型评估失败: {e}")
        return 1


def real_inference(args):
    """实时推理模式"""
    logger.info("启动实时推理模式...")
    
    try:
        # 这里可以添加实时推理逻辑
        logger.info("实时推理功能开发中...")
        return 0
        
    except Exception as e:
        logger.error(f"实时推理失败: {e}")
        return 1


def create_project_structure():
    """创建项目目录结构"""
    logger.info("创建项目目录结构...")
    
    directories = [
        "datasets/raw",
        "datasets/processed",
        "datasets/cache",
        "models/trained",
        "models/tflite",
        "models/arduino",
        "logs",
        "outputs",
        "docs"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {directory}")
    
    logger.success("项目目录结构创建完成!")


def main():
    """主函数"""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="BSL手势识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  数据采集:
    可视化:     python run.py collect --visualize --duration 30
    CSV采集:    python run.py collect --csv --duration 60 --filename test_data
    自动采集:   python run.py collect --auto --user_id 001 --gestures 0 1 2
    交互式:     python run.py collect --interactive --samples 10
  
  数据预处理:   python run.py preprocess --input_prefix raw_data
  模型训练:     python run.py train --data_prefix processed_data --model_type both
  模型评估:     python run.py evaluate --model_path models/trained/model.pth
  实时推理:     python run.py infer --model_path models/trained/model.pth --port /dev/ttyACM0
  创建目录:     python run.py init
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 数据采集命令
    collect_parser = subparsers.add_parser('collect', help='Data collection and visualization')
    collect_parser.add_argument('--port', default='/dev/cu.usbmodem101', help='Arduino serial port')
    collect_parser.add_argument('--baudrate', type=int, default=115200, help='Baud rate')
    collect_parser.add_argument('--visualize', action='store_true', help='Realtime visualization only')
    collect_parser.add_argument('--csv', action='store_true', help='Collect data and save as CSV')
    collect_parser.add_argument('--auto', action='store_true', help='Auto gesture collection mode')
    collect_parser.add_argument('--duration', type=int, default=30, help='Collection/visualization duration (seconds)')
    collect_parser.add_argument('--gesture_duration', type=int, default=2, help='Duration per gesture in auto mode (seconds)')
    collect_parser.add_argument('--output_dir', default='datasets/csv', help='CSV output directory')
    collect_parser.add_argument('--filename', help='CSV filename (without extension)')
    collect_parser.add_argument('--user_id', default='001', help='User identifier for auto collection')
    collect_parser.add_argument('--gestures', nargs='+', type=int, help='Gesture IDs to collect (e.g., 0 1 2)')
    collect_parser.add_argument('--interactive', action='store_true', help='Interactive gesture collection')
    collect_parser.add_argument('--samples', type=int, default=10, help='Samples per gesture')
    
    # 数据预处理命令
    preprocess_parser = subparsers.add_parser('preprocess', help='数据预处理')
    preprocess_parser.add_argument('--input_dir', default='datasets/raw', help='输入目录')
    preprocess_parser.add_argument('--output_dir', default='datasets/processed', help='输出目录')
    preprocess_parser.add_argument('--input_prefix', required=True, help='输入文件前缀')
    preprocess_parser.add_argument('--augment_factor', type=int, default=2, help='数据增强倍数')
    preprocess_parser.add_argument('--normalize_method', default='standard', 
                                 choices=['standard', 'minmax'], help='归一化方法')
    preprocess_parser.add_argument('--visualize', action='store_true', help='生成数据可视化')
    
    # 模型训练命令
    train_parser = subparsers.add_parser('train', help='模型训练')
    train_parser.add_argument('--model_type', choices=['transformer', 'cnn1d', 'both'], 
                            default='both', help='模型类型')
    train_parser.add_argument('--data_dir', default='datasets/processed', help='数据目录')
    train_parser.add_argument('--data_prefix', required=True, help='数据文件前缀')
    train_parser.add_argument('--output_dir', default='models/trained', help='输出目录')
    train_parser.add_argument('--epochs', type=int, help='训练轮数')
    train_parser.add_argument('--batch_size', type=int, help='批次大小')
    train_parser.add_argument('--learning_rate', type=float, help='学习率')
    train_parser.add_argument('--visualize', action='store_true', help='显示训练曲线')
    
    # 模型评估命令
    eval_parser = subparsers.add_parser('evaluate', help='模型评估')
    eval_parser.add_argument('--model_path', required=True, help='模型路径')
    eval_parser.add_argument('--data_dir', default='datasets/processed', help='测试数据目录')
    eval_parser.add_argument('--data_prefix', required=True, help='数据文件前缀')
    eval_parser.add_argument('--output_dir', default='outputs/evaluation', help='输出目录')
    
    # 实时推理命令
    infer_parser = subparsers.add_parser('infer', help='实时推理')
    infer_parser.add_argument('--model_path', required=True, help='模型路径')
    infer_parser.add_argument('--port', default='/dev/cu.usbmodem101', help='Arduino串口端口')
    infer_parser.add_argument('--baudrate', type=int, default=115200, help='波特率')
    infer_parser.add_argument('--confidence', type=float, default=0.8, help='置信度阈值')
    
    # 初始化命令
    init_parser = subparsers.add_parser('init', help='初始化项目目录结构')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    logger.info(f"BSL手势识别系统 v1.0")
    logger.info(f"执行命令: {args.command}")
    
    # 路由到对应的处理函数
    if args.command == 'collect':
        return data_collection(args)
    elif args.command == 'preprocess':
        return data_preprocessing(args)
    elif args.command == 'train':
        return model_training(args)
    elif args.command == 'evaluate':
        return model_evaluation(args)
    elif args.command == 'infer':
        return real_inference(args)
    elif args.command == 'init':
        create_project_structure()
        return 0
    else:
        logger.error(f"未知命令: {args.command}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 