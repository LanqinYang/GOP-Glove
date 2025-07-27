"""
BSL Gesture Recognition - CNN-LSTM Model Definition
"""
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Constants
SEQUENCE_LENGTH = 100
N_FEATURES = 5
N_CLASSES = 11

class CnnLstmModelCreator:
    def __init__(self):
        pass

    def extract_and_scale_features(self, X_data, fit=False, scaler=None, arduino_mode=False):
        # For TF models, "features" are the raw sequence data.
        # We only need to scale them.
        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_data.reshape(-1, X_data.shape[-1])).reshape(X_data.shape)
            return X_scaled, scaler
        else:
            X_scaled = scaler.transform(X_data.reshape(-1, X_data.shape[-1])).reshape(X_data.shape)
            return X_scaled, None

    def define_hyperparams(self, trial, arduino_mode=False):
        if arduino_mode:
            return {
                'use_dropout': trial.suggest_categorical('use_dropout', [True, False]),
                'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32])
            }
        else:
            params = {
                'n_conv_layers': trial.suggest_int('n_conv_layers', 1, 3),
                'n_lstm_layers': trial.suggest_int('n_lstm_layers', 1, 2),
                'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
                'use_conv_dropout': trial.suggest_categorical('use_conv_dropout', [True, False]),
                'use_lstm_dropout': trial.suggest_categorical('use_lstm_dropout', [True, False]),
                'use_dense_dropout': trial.suggest_categorical('use_dense_dropout', [True, False]),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'swish']),
                'dense_units': trial.suggest_int('dense_units', 32, 128),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
            }
            # Add conditional params
            for i in range(params['n_conv_layers']):
                params[f'conv{i+1}_filters'] = trial.suggest_int(f'conv{i+1}_filters', 32, 128, log=True)
                # Reduce kernel size range to prevent negative dimension errors
                params[f'conv{i+1}_kernel'] = trial.suggest_int(f'conv{i+1}_kernel', 3, 5)
            for i in range(params['n_lstm_layers']):
                params[f'lstm{i+1}_units'] = trial.suggest_int(f'lstm{i+1}_units', 32, 128)
            if params['use_conv_dropout']:
                params['conv_dropout'] = trial.suggest_float('conv_dropout', 0.1, 0.5)
            if params['use_lstm_dropout']:
                params['lstm_dropout'] = trial.suggest_float('lstm_dropout', 0.1, 0.5)
            if params['use_dense_dropout']:
                params['dense_dropout'] = trial.suggest_float('dense_dropout', 0.2, 0.6)
            return params

    def create_model(self, params, arduino_mode=False, callbacks=None):
        model = Sequential()
        model.add(layers.Input(shape=(SEQUENCE_LENGTH, N_FEATURES)))
    
        if arduino_mode:
            model.add(layers.Conv1D(16, 3, activation='relu'))
            model.add(layers.MaxPooling1D(2))
            model.add(layers.Conv1D(32, 3, activation='relu'))
            model.add(layers.MaxPooling1D(2))
            model.add(layers.LSTM(32, return_sequences=False))
            if params.get('use_dropout', False):
                model.add(layers.Dropout(0.3))
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(N_CLASSES, activation='softmax'))
            lr = params.get('learning_rate', 1e-3)
        else:
            for i in range(params['n_conv_layers']):
                model.add(layers.Conv1D(params[f'conv{i+1}_filters'], params[f'conv{i+1}_kernel'], activation=params['activation']))
                if params['use_batch_norm']:
                    model.add(layers.BatchNormalization())
                model.add(layers.MaxPooling1D(2))
                if params['use_conv_dropout']:
                    model.add(layers.Dropout(params['conv_dropout']))
            
            for i in range(params['n_lstm_layers']):
                return_sequences = (i < params['n_lstm_layers'] - 1)
                model.add(layers.LSTM(params[f'lstm{i+1}_units'], return_sequences=return_sequences))
                if params['use_lstm_dropout']:
                    model.add(layers.Dropout(params['lstm_dropout']))
            
            model.add(layers.Dense(params['dense_units'], activation=params['activation']))
            if params['use_dense_dropout']:
                model.add(layers.Dropout(params['dense_dropout']))
            model.add(layers.Dense(N_CLASSES, activation='softmax'))
            lr = params['learning_rate']
        
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model 