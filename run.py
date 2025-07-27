#!/usr/bin/env python3
"""
BSL Gesture Recognition System - Main Entry Point
"""

import sys
import argparse
from pathlib import Path

# Project root
project_root = Path(__file__).parent

def model_training(args):
    """Dynamically imports and runs the training pipeline for a given model."""
    
    # Dynamically import the required ModelCreator
    if args.model_type == '1D_CNN':
        from src.training.train_cnn1d import Cnn1dModelCreator
        model_creator = Cnn1dModelCreator()
    elif args.model_type == 'XGBoost':
        from src.training.train_xgboost import XgboostModelCreator
        model_creator = XgboostModelCreator()
    elif args.model_type == 'CNN_LSTM':
        from src.training.train_cnn_lstm import CnnLstmModelCreator
        model_creator = CnnLstmModelCreator()
    elif args.model_type == 'Transformer_Encoder':
        from src.training.train_transformer import TransformerModelCreator
        model_creator = TransformerModelCreator()
    else:
        print(f"ERROR: Unknown model type '{args.model_type}'")
        return 1

    # Import and run the unified pipeline
    from src.training.pipeline import run_training_pipeline
    run_training_pipeline(args, model_creator)
    
    return 0

def main():
    """Main function to parse arguments and dispatch commands."""
    parser = argparse.ArgumentParser(description='BSL Gesture Recognition System')
    
    # Model training command
    # No subparsers needed if training is the main function
    parser.add_argument('--csv_dir', default='datasets/gesture_csv', help="Directory with gesture CSV files")
    parser.add_argument('--output_dir', default='models/trained', help="Directory to save trained model artifacts")
    parser.add_argument('--model_type', required=True,
                             choices=['1D_CNN', 'XGBoost', 'CNN_LSTM', 'Transformer_Encoder'],
                             help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs (for TF models)")
    parser.add_argument('--n_trials', type=int, default=100, help="Number of Optuna trials for hyperparameter optimization")
    parser.add_argument('--arduino', action='store_true', 
                             help='Use Arduino optimization mode (smaller model, potentially lower accuracy)')
    parser.add_argument('--loso', action='store_true', help='Use Leave-One-Subject-Out (LOSO) cross-validation')
    
    args = parser.parse_args()
    
    # Directly call the training function
    return model_training(args)

if __name__ == "__main__":
    sys.exit(main()) 