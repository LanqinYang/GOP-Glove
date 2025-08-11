"""
BSL Gesture Recognition - Transformer Encoder Model Definition
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Constants
SEQUENCE_LENGTH = 100
N_FEATURES = 5
N_CLASSES = 11

class TransformerModelCreator:
    def __init__(self):
        pass

    def extract_and_scale_features(self, X_data, fit=False, scaler=None, arduino_mode=False):
        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_data.reshape(-1, X_data.shape[-1])).reshape(X_data.shape)
            return X_scaled, scaler
        else:
            X_scaled = scaler.transform(X_data.reshape(-1, X_data.shape[-1])).reshape(X_data.shape)
            return X_scaled, scaler

    def define_hyperparams(self, trial, arduino_mode=False):
        if arduino_mode:
            return {
                'use_dropout': trial.suggest_categorical('use_dropout', [True, False]),
                'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32])
            }
        else:
            params = {
                'n_transformer_layers': trial.suggest_int('n_transformer_layers', 1, 4),
                'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
                'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
                'ff_dim': trial.suggest_int('ff_dim', 64, 256),
                'use_attention_dropout': trial.suggest_categorical('use_attention_dropout', [True, False]),
                'use_transformer_dropout': trial.suggest_categorical('use_transformer_dropout', [True, False]),
                'use_dense_dropout': trial.suggest_categorical('use_dense_dropout', [True, False]),
                'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'swish']),
                'dense_units': trial.suggest_int('dense_units', 32, 128),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                
                # 数据增强参数 - 基于论文研究的保守设置
                'augment_factor': trial.suggest_int('augment_factor', 1, 2),
                'jitter_noise_level': trial.suggest_float('jitter_noise_level', 0.005, 0.015),
                'time_warp_max_speed': trial.suggest_int('time_warp_max_speed', 2, 3),
                'scale_min': trial.suggest_float('scale_min', 0.95, 0.98),
                'scale_max': trial.suggest_float('scale_max', 1.02, 1.05),
                'augment_prob': trial.suggest_float('augment_prob', 0.3, 0.6)
            }
            while params['d_model'] % params['num_heads'] != 0:
                params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8])
            
            if params['use_attention_dropout']:
                params['attention_dropout'] = trial.suggest_float('attention_dropout', 0.1, 0.3)
            if params['use_transformer_dropout']:
                params['transformer_dropout'] = trial.suggest_float('transformer_dropout', 0.1, 0.3)
            if params['use_dense_dropout']:
                params['dense_dropout'] = trial.suggest_float('dense_dropout', 0.2, 0.6)
            
            # 构建scale_range
            params['scale_range'] = [params['scale_min'], params['scale_max']]
            
            return params

    def create_model(self, params, arduino_mode=False, callbacks=None):
        inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, N_FEATURES))
        
        if arduino_mode:
            d_model, num_heads, ff_dim = 32, 2, 64
            x = layers.Dense(d_model)(inputs)
            attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=0.1)(x, x)
            attention_output = layers.Dropout(0.1)(attention_output)
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization()(x)
            ff_output = layers.Dense(ff_dim, activation='relu')(x)
            ff_output = layers.Dense(d_model)(ff_output)
            ff_output = layers.Dropout(0.1)(ff_output)
            x = layers.Add()([x, ff_output])
            x = layers.LayerNormalization()(x)
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dense(32, activation='relu')(x)
            if params.get('use_dropout', False):
                x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(N_CLASSES, activation='softmax')(x)
            lr = 1e-3
        else:
            x = layers.Dense(params['d_model'])(inputs)
            for _ in range(params['n_transformer_layers']):
                attention_output = layers.MultiHeadAttention(
                    num_heads=params['num_heads'],
                    key_dim=params['d_model'] // params['num_heads'],
                    dropout=params.get('attention_dropout', 0.0)
                )(x, x)
                attention_output = layers.Dropout(params.get('transformer_dropout', 0.0))(attention_output)
                x = layers.Add()([x, attention_output])
                x = layers.LayerNormalization()(x)
                ff_output = layers.Dense(params['ff_dim'], activation=params['activation'])(x)
                ff_output = layers.Dense(params['d_model'])(ff_output)
                ff_output = layers.Dropout(params.get('transformer_dropout', 0.0))(ff_output)
                x = layers.Add()([x, ff_output])
                x = layers.LayerNormalization()(x)
            
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dense(params['dense_units'], activation=params['activation'])(x)
            if params.get('use_dense_dropout', False):
                x = layers.Dropout(params['dense_dropout'])(x)
            outputs = layers.Dense(N_CLASSES, activation='softmax')(x)
            lr = params['learning_rate']
            
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model 