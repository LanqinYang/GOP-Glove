#!/usr/bin/env python3
"""
Generate final hyperparameters table for all models.
This script extracts the best hyperparameters from Optuna studies and creates a comprehensive table.
"""

import os
import sys
import json
import pandas as pd
import yaml
from datetime import datetime
import glob
from pathlib import Path

# Add project root to path
sys.path.append('.')

class FinalHyperparamsGenerator:
    """Generate final hyperparameters table for all models."""
    
    def __init__(self, output_dir="outputs/final_hyperparams"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Model configurations
        self.models = [
            '1D_CNN',
            'Transformer_Encoder', 
            'XGBoost',
            'LightGBM',
            'ADANN',
            'ADANN_LightGBM'
        ]
        
        self.results = {}
    
    def extract_hyperparams_from_optuna(self):
        """Extract best hyperparameters from Optuna study files."""
        print("🔍 Extracting hyperparameters from Optuna studies...")
        
        for model in self.models:
            print(f"  Processing {model}...")
            
            # Look for Optuna study files
            study_pattern = f"outputs/{model}/**/optuna_study_*.json"
            study_files = glob.glob(study_pattern, recursive=True)
            
            if not study_files:
                print(f"    ⚠️ No Optuna study found for {model}")
                continue
            
            # Use the most recent study file
            latest_study = max(study_files, key=os.path.getctime)
            
            try:
                with open(latest_study, 'r') as f:
                    study_data = json.load(f)
                
                # Extract best parameters
                if 'best_params' in study_data:
                    self.results[model] = {
                        'best_params': study_data['best_params'],
                        'best_value': study_data.get('best_value', 'N/A'),
                        'study_file': latest_study
                    }
                    print(f"    ✅ Extracted {len(study_data['best_params'])} parameters")
                else:
                    print(f"    ⚠️ No best_params found in {latest_study}")
                    
            except Exception as e:
                print(f"    ❌ Error reading {latest_study}: {e}")
    
    def get_default_hyperparams(self):
        """Get default hyperparameters from model creators."""
        print("📋 Getting default hyperparameters from model creators...")
        
        # Import model creators
        try:
            from src.training.train_cnn1d import Cnn1dModelCreator
            from src.training.train_transformer import TransformerModelCreator
            from src.training.train_xgboost import XgboostModelCreator
            from src.training.train_lightgbm import LightgbmModelCreator
            from src.training.train_adann import AdannModelCreator
            from src.training.train_adann_lightgbm import AdannLightgbmModelCreator
            
            # Create model creators
            creators = {
                '1D_CNN': Cnn1dModelCreator(),
                'Transformer_Encoder': TransformerModelCreator(),
                'XGBoost': XgboostModelCreator(),
                'LightGBM': LightgbmModelCreator(),
                'ADANN': AdannModelCreator(),
                'ADANN_LightGBM': AdannLightgbmModelCreator()
            }
            
            # Get default parameters using a mock trial
            import optuna
            trial = optuna.trial.FixedTrial({})
            
            for model, creator in creators.items():
                try:
                    default_params = creator.define_hyperparams(trial, arduino_mode=False)
                    if model not in self.results:
                        self.results[model] = {}
                    self.results[model]['default_params'] = default_params
                    print(f"    ✅ Got default params for {model}")
                except Exception as e:
                    print(f"    ❌ Error getting default params for {model}: {e}")
                    
        except ImportError as e:
            print(f"    ⚠️ Could not import model creators: {e}")
    
    def create_hyperparams_table(self):
        """Create comprehensive hyperparameters table."""
        print("📊 Creating hyperparameters table...")
        
        table_data = []
        
        for model in self.models:
            row = {'Model': model}
            
            if model in self.results:
                # Add best parameters
                if 'best_params' in self.results[model]:
                    best_params = self.results[model]['best_params']
                    for param, value in best_params.items():
                        row[f'Best_{param}'] = value
                    
                    # Add best value
                    if 'best_value' in self.results[model]:
                        row['Best_Validation_Accuracy'] = self.results[model]['best_value']
                
                # Add default parameters for comparison
                if 'default_params' in self.results[model]:
                    default_params = self.results[model]['default_params']
                    for param, value in default_params.items():
                        if f'Best_{param}' not in row:  # Only add if not already in best params
                            row[f'Default_{param}'] = value
            
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Reorder columns to group by model, best params, default params
        model_cols = ['Model']
        best_cols = [col for col in df.columns if col.startswith('Best_')]
        default_cols = [col for col in df.columns if col.startswith('Default_')]
        
        # Sort best and default columns
        best_cols.sort()
        default_cols.sort()
        
        # Reorder DataFrame
        df = df[model_cols + best_cols + default_cols]
        
        return df
    
    def create_arduino_hyperparams_table(self):
        """Create Arduino-specific hyperparameters table."""
        print("📊 Creating Arduino hyperparameters table...")
        
        table_data = []
        
        for model in self.models:
            row = {'Model': model}
            
            if model in self.results:
                # Get Arduino parameters
                try:
                    from src.training.train_cnn1d import Cnn1dModelCreator
                    from src.training.train_transformer import TransformerModelCreator
                    from src.training.train_xgboost import XgboostModelCreator
                    from src.training.train_lightgbm import LightgbmModelCreator
                    from src.training.train_adann import AdannModelCreator
                    from src.training.train_adann_lightgbm import AdannLightgbmModelCreator
                    
                    creators = {
                        '1D_CNN': Cnn1dModelCreator(),
                        'Transformer_Encoder': TransformerModelCreator(),
                        'XGBoost': XgboostModelCreator(),
                        'LightGBM': LightgbmModelCreator(),
                        'ADANN': AdannModelCreator(),
                        'ADANN_LightGBM': AdannLightgbmModelCreator()
                    }
                    
                    if model in creators:
                        import optuna
                        trial = optuna.trial.FixedTrial({})
                        arduino_params = creators[model].define_hyperparams(trial, arduino_mode=True)
                        
                        for param, value in arduino_params.items():
                            row[f'Arduino_{param}'] = value
                        
                        print(f"    ✅ Got Arduino params for {model}")
                        
                except Exception as e:
                    print(f"    ❌ Error getting Arduino params for {model}: {e}")
            
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Reorder columns
        model_cols = ['Model']
        arduino_cols = [col for col in df.columns if col.startswith('Arduino_')]
        arduino_cols.sort()
        
        df = df[model_cols + arduino_cols]
        
        return df
    
    def create_summary_table(self):
        """Create a summary table with key hyperparameters."""
        print("📊 Creating summary table...")
        
        summary_data = []
        
        for model in self.models:
            row = {'Model': model}
            
            if model in self.results and 'best_params' in self.results[model]:
                best_params = self.results[model]['best_params']
                
                # Extract key parameters based on model type
                if '1D_CNN' in model:
                    row['Learning_Rate'] = best_params.get('learning_rate', 'N/A')
                    row['Batch_Size'] = best_params.get('batch_size', 'N/A')
                    row['Dense_Units'] = best_params.get('dense_units', 'N/A')
                    row['Augment_Factor'] = best_params.get('augment_factor', 'N/A')
                
                elif 'Transformer' in model:
                    row['Learning_Rate'] = best_params.get('learning_rate', 'N/A')
                    row['Batch_Size'] = best_params.get('batch_size', 'N/A')
                    row['D_Model'] = best_params.get('d_model', 'N/A')
                    row['Num_Heads'] = best_params.get('num_heads', 'N/A')
                    row['Augment_Factor'] = best_params.get('augment_factor', 'N/A')
                
                elif 'XGBoost' in model:
                    row['Learning_Rate'] = best_params.get('learning_rate', 'N/A')
                    row['Max_Depth'] = best_params.get('max_depth', 'N/A')
                    row['N_Estimators'] = best_params.get('n_estimators', 'N/A')
                    row['Augment_Factor'] = best_params.get('augment_factor', 'N/A')
                
                elif 'LightGBM' in model:
                    row['Learning_Rate'] = best_params.get('learning_rate', 'N/A')
                    row['Num_Leaves'] = best_params.get('num_leaves', 'N/A')
                    row['N_Estimators'] = best_params.get('n_estimators', 'N/A')
                    row['Augment_Factor'] = best_params.get('augment_factor', 'N/A')
                
                elif 'ADANN' in model:
                    row['Learning_Rate'] = best_params.get('learning_rate', 'N/A')
                    row['Batch_Size'] = best_params.get('batch_size', 'N/A')
                    row['Feature_Size'] = best_params.get('feature_size', 'N/A')
                    row['Augment_Factor'] = best_params.get('augment_factor', 'N/A')
                
                # Add best validation accuracy
                if 'best_value' in self.results[model]:
                    row['Best_Validation_Accuracy'] = self.results[model]['best_value']
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def save_tables(self):
        """Save all hyperparameter tables."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive table
        comprehensive_df = self.create_hyperparams_table()
        comprehensive_path = f"{self.output_dir}/comprehensive_hyperparams_{timestamp}.csv"
        comprehensive_df.to_csv(comprehensive_path, index=False)
        
        # Create Arduino table
        arduino_df = self.create_arduino_hyperparams_table()
        arduino_path = f"{self.output_dir}/arduino_hyperparams_{timestamp}.csv"
        arduino_df.to_csv(arduino_path, index=False)
        
        # Create summary table
        summary_df = self.create_summary_table()
        summary_path = f"{self.output_dir}/summary_hyperparams_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Create markdown table for documentation
        markdown_path = f"{self.output_dir}/hyperparams_table_{timestamp}.md"
        self._create_markdown_table(summary_df, markdown_path)
        
        print(f"📊 Comprehensive table: {comprehensive_path}")
        print(f"📊 Arduino table: {arduino_path}")
        print(f"📊 Summary table: {summary_path}")
        print(f"📊 Markdown table: {markdown_path}")
        
        return comprehensive_path, arduino_path, summary_path, markdown_path
    
    def _create_markdown_table(self, df, output_path):
        """Create a markdown table for documentation."""
        with open(output_path, 'w') as f:
            f.write("# Final Hyperparameters Table\n\n")
            f.write("This table shows the best hyperparameters found by Optuna for each model.\n\n")
            f.write("**Note**: These are the fixed hyperparameters used for final training and evaluation. HPO was only used during development phase.\n\n")
            
            # Write the table
            f.write(df.to_markdown(index=False))
            
            f.write("\n\n## Usage Notes\n\n")
            f.write("- **Reproducibility**: Use these exact hyperparameters with fixed random seeds for reproducible results\n")
            f.write("- **HPO Purpose**: Hyperparameter optimization was used only during development to find optimal configurations\n")
            f.write("- **Final Training**: All reported results use these fixed hyperparameters without further optimization\n")
            f.write("- **Arduino Mode**: Some models have different hyperparameters for Arduino deployment (see arduino_hyperparams table)\n")

def main():
    """Main function to generate final hyperparameters table."""
    print("="*60)
    print("FINAL HYPERPARAMETERS TABLE GENERATOR")
    print("="*60)
    
    # Initialize generator
    generator = FinalHyperparamsGenerator()
    
    # Extract hyperparameters
    generator.extract_hyperparams_from_optuna()
    generator.get_default_hyperparams()
    
    # Save tables
    comprehensive_path, arduino_path, summary_path, markdown_path = generator.save_tables()
    
    print("\n" + "="*60)
    print("HYPERPARAMETERS TABLE GENERATION COMPLETE")
    print("="*60)
    print(f"📊 Comprehensive table: {comprehensive_path}")
    print(f"📊 Arduino table: {arduino_path}")
    print(f"📊 Summary table: {summary_path}")
    print(f"📊 Markdown table: {markdown_path}")
    print("\nThese tables contain the fixed hyperparameters for reproducible training.")

if __name__ == "__main__":
    main()
