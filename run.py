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
    cmd = [
        sys.executable, "src/training/train.py",
        "--csv_dir", args.csv_dir,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--n_trials", str(args.n_trials)
    ]
    
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode

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
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--n_trials', type=int, default=100)
    train_parser.set_defaults(func=model_training)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 