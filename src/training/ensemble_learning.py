"""
集成学习模块 - 基于最新研究的多模型融合策略
用于提升跨被试手势识别的鲁棒性和准确率
"""

import numpy as np
import tensorflow as tf
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
from typing import List, Dict, Any
import os

class EMGEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    EMG手势识别集成分类器
    结合多个模型的预测结果以提升鲁棒性
    """
    
    def __init__(self, models: List[Any], weights: List[float] = None, voting: str = 'soft'):
        """
        初始化集成分类器
        
        参数:
        - models: 基础模型列表
        - weights: 模型权重
        - voting: 投票方式 ('soft' 或 'hard')
        """
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        self.voting = voting
        self.n_classes_ = None
        
    def fit(self, X, y):
        """训练所有基础模型"""
        # 假设所有模型都已经训练好了
        self.n_classes_ = len(np.unique(y))
        return self
    
    def predict_proba(self, X):
        """预测类别概率"""
        if self.voting == 'soft':
            # 软投票：平均概率
            all_probas = []
            for i, model in enumerate(self.models):
                try:
                    if hasattr(model, 'predict_proba'):
                        probas = model.predict_proba(X)
                    elif hasattr(model, 'predict'):
                        # 对于神经网络模型
                        predictions = model.predict(X)
                        if len(predictions.shape) == 2 and predictions.shape[1] > 1:
                            probas = predictions
                        else:
                            # 转换为one-hot编码
                            probas = tf.keras.utils.to_categorical(predictions, num_classes=self.n_classes_)
                    else:
                        continue
                    
                    all_probas.append(probas * self.weights[i])
                except Exception as e:
                    print(f"模型 {i} 预测失败: {e}")
                    continue
            
            if all_probas:
                return np.mean(all_probas, axis=0)
            else:
                raise ValueError("没有模型能够成功预测")
        else:
            # 硬投票：多数表决
            all_predictions = []
            for i, model in enumerate(self.models):
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict(X)
                        if len(pred.shape) == 2:
                            pred = np.argmax(pred, axis=1)
                        all_predictions.append(pred)
                except Exception as e:
                    print(f"模型 {i} 预测失败: {e}")
                    continue
            
            if all_predictions:
                # 投票决定最终预测
                predictions_array = np.array(all_predictions).T
                final_predictions = []
                for row in predictions_array:
                    values, counts = np.unique(row, return_counts=True)
                    final_predictions.append(values[np.argmax(counts)])
                
                # 转换为概率形式
                probas = np.zeros((len(final_predictions), self.n_classes_))
                for i, pred in enumerate(final_predictions):
                    probas[i, pred] = 1.0
                return probas
            else:
                raise ValueError("没有模型能够成功预测")
    
    def predict(self, X):
        """预测类别"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

class AdaptiveWeightEnsemble:
    """
    自适应权重集成学习
    根据每个模型在验证集上的表现动态调整权重
    """
    
    def __init__(self, models: List[Any]):
        self.models = models
        self.weights = np.ones(len(models)) / len(models)
        self.validation_scores = np.zeros(len(models))
        
    def update_weights(self, X_val, y_val):
        """根据验证集表现更新模型权重"""
        scores = []
        for model in self.models:
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_val)
                    if len(pred.shape) == 2:
                        pred = np.argmax(pred, axis=1)
                    score = np.mean(pred == y_val)
                    scores.append(score)
                else:
                    scores.append(0.0)
            except:
                scores.append(0.0)
        
        self.validation_scores = np.array(scores)
        
        # 使用softmax计算权重，表现好的模型权重更高
        exp_scores = np.exp(self.validation_scores * 5)  # 温度参数=5
        self.weights = exp_scores / np.sum(exp_scores)
        
        print(f"更新后的模型权重: {self.weights}")
        return self.weights
    
    def predict(self, X):
        """加权预测"""
        all_predictions = []
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(X)
                if len(pred.shape) == 2:
                    # 概率输出
                    all_predictions.append(pred * self.weights[i])
                else:
                    # 类别输出，转换为one-hot
                    pred_onehot = tf.keras.utils.to_categorical(pred, num_classes=11)
                    all_predictions.append(pred_onehot * self.weights[i])
            except Exception as e:
                print(f"模型 {i} 预测失败: {e}")
                continue
        
        if all_predictions:
            ensemble_pred = np.sum(all_predictions, axis=0)
            return np.argmax(ensemble_pred, axis=1)
        else:
            raise ValueError("所有模型预测都失败了")

def create_cross_subject_ensemble(model_creators, X_train, y_train, X_val, y_val, subjects_train, subjects_val):
    """
    创建跨被试集成模型
    
    参数:
    - model_creators: 模型创建器列表
    - X_train, y_train: 训练数据
    - X_val, y_val: 验证数据
    - subjects_train, subjects_val: 被试标签
    
    返回:
    - 训练好的集成模型
    """
    trained_models = []
    
    # 为每个被试训练专门的模型
    unique_subjects = np.unique(subjects_train)
    subject_models = {}
    
    for subject in unique_subjects:
        subject_mask = subjects_train == subject
        X_subj = X_train[subject_mask]
        y_subj = y_train[subject_mask]
        
        if len(X_subj) > 0:
            # 为每个模型类型训练一个被试特定的模型
            subject_models[subject] = []
            for creator in model_creators:
                try:
                    # 为不同模型准备合适的参数
                    if hasattr(creator, 'define_hyperparams'):
                        # 使用Optuna trial来获取默认参数
                        import optuna
                        study = optuna.create_study(direction='maximize')
                        trial = study.ask()
                        params = creator.define_hyperparams(trial, arduino_mode=False)
                    else:
                        # 简化的通用参数
                        params = {
                            'conv1_filters': 64,
                            'conv1_kernel': 5,
                            'conv2_filters': 128,
                            'conv2_kernel': 3,
                            'conv3_filters': 256,
                            'conv3_kernel': 3,
                            'dropout_rate': 0.3,
                            'l2_reg': 1e-4,
                            'dense_units': 128,
                            'use_batch_norm': True,
                            'learning_rate': 0.001,
                            'batch_size': 32
                        }
                    
                    model = creator.create_model(params, arduino_mode=False)
                    
                    # 训练模型
                    model.fit(X_subj, y_subj, epochs=50, verbose=0, 
                             validation_split=0.2,
                             callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
                    
                    subject_models[subject].append(model)
                except Exception as e:
                    print(f"被试 {subject} 的模型训练失败: {e}")
    
    # 创建通用模型（使用所有数据）
    general_models = []
    for creator in model_creators:
        try:
            # 为不同模型准备合适的参数
            if hasattr(creator, 'define_hyperparams'):
                import optuna
                study = optuna.create_study(direction='maximize')
                trial = study.ask()
                params = creator.define_hyperparams(trial, arduino_mode=False)
            else:
                params = {
                    'conv1_filters': 64,
                    'conv1_kernel': 5,
                    'conv2_filters': 128,
                    'conv2_kernel': 3,
                    'conv3_filters': 256,
                    'conv3_kernel': 3,
                    'dropout_rate': 0.3,
                    'l2_reg': 1e-4,
                    'dense_units': 128,
                    'use_batch_norm': True,
                    'learning_rate': 0.001,
                    'batch_size': 32
                }
            
            model = creator.create_model(params, arduino_mode=False)
            
            # 检查是否是多输出模型（如RAC的新版本）
            if hasattr(model, 'output_names') and len(model.output_names) > 1:
                # 多输出模型，只训练手势识别部分
                print(f"检测到多输出模型，跳过训练以避免复杂性")
                continue
            else:
                # 单输出模型，正常训练
                model.fit(X_train, y_train, epochs=50, verbose=0,
                         validation_data=(X_val, y_val),
                         callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
                
                general_models.append(model)
        except Exception as e:
            print(f"通用模型训练失败: {e}")
    
    # 组合所有模型
    all_models = general_models.copy()
    for subject_model_list in subject_models.values():
        all_models.extend(subject_model_list)
    
    # 创建自适应权重集成
    ensemble = AdaptiveWeightEnsemble(all_models)
    ensemble.update_weights(X_val, y_val)
    
    return ensemble, subject_models, general_models

def save_ensemble(ensemble, filepath):
    """保存集成模型"""
    joblib.dump(ensemble, filepath)

def load_ensemble(filepath):
    """加载集成模型"""
    return joblib.load(filepath)