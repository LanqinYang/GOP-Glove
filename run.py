#!/usr/bin/env python3
"""
BSL Gesture Recognition System
Author: Lambert Yang
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Project root
project_root = Path(__file__).parent

def data_collection(args):
    """Data collection mode"""
    cmd = [sys.executable, "src/data/data_collector.py", args.mode]
    if args.port:
        cmd.extend(["--port", args.port])
    
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode

def model_training(args):
    """Model training mode"""
    # Dynamically import the required training functions
    if args.model_type == '1D_CNN':
        from src.training.train_cnn1d import train_model, train_loso_model
    elif args.model_type == 'XGBoost':
        from src.training.train_xgboost import train_model, train_loso_model
    elif args.model_type == 'CNN_LSTM':
        from src.training.train_cnn_lstm import train_model, train_loso_model
    elif args.model_type == 'Transformer_Encoder':
        from src.training.train_transformer import train_model, train_loso_model
    else:
        print(f"错误：未知的模型类型 '{args.model_type}'")
        return 1

    if args.loso:
        print(f"🚀 开始对 {args.model_type} 模型进行LOSO训练...")
        if args.model_type == 'XGBoost':
            train_loso_model(args.csv_dir, args.output_dir, args.model_type, args.n_trials, args.arduino)
        else:
            train_loso_model(args.csv_dir, args.output_dir, args.model_type, args.n_trials, args.epochs, args.arduino)
    else:
        print(f"🚀 开始对 {args.model_type} 模型进行标准训练...")
        if args.model_type == 'XGBoost':
            train_model(args.csv_dir, args.output_dir, args.n_trials, args.model_type, args.arduino)
        else:
            train_model(args.csv_dir, args.output_dir, args.model_type, args.n_trials, args.epochs, args.arduino)
    
    return 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='BSL Gesture Recognition System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data collection command
    collect_parser = subparsers.add_parser('collect', help='Data collection')
    collect_parser.add_argument('mode', choices=['test', 'auto'], help='Collection mode')
    collect_parser.add_argument('--port', help='Serial port (e.g., /dev/cu.usbmodem2101)')
    collect_parser.set_defaults(func=data_collection)
    
    # Model training command
    train_parser = subparsers.add_parser('train', help='Model training')
    train_parser.add_argument('--csv_dir', default='datasets/gesture_csv')
    train_parser.add_argument('--output_dir', default='models/trained')
    train_parser.add_argument('--model_type', required=True,
                             choices=['1D_CNN', 'XGBoost', 'CNN_LSTM', 'Transformer_Encoder'],
                             help='Type of model to train (required)')
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--n_trials', type=int, default=100)
    train_parser.add_argument('--arduino', action='store_true', 
                             help='Use Arduino optimization mode for XGBoost (smaller file size)')
    train_parser.add_argument('--loso', action='store_true', help='使用“留一被试法”（LOSO）交叉验证')
    train_parser.set_defaults(func=model_training)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 