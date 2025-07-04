"""
BSL手势识别系统 - 模型训练脚本

该脚本支持训练两种模型：
1. Transformer模型（PC端高精度）
2. 1D-CNN模型（Arduino端轻量级）

功能：
- 数据加载和预处理
- 模型训练和验证
- 训练过程可视化
- 模型保存和评估
- 早停和学习率调度

作者: Lambert Yang
版本: 1.0
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
from loguru import logger
import yaml

# PyTorch相关
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# TensorFlow相关
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau as TF_ReduceLROnPlateau, ModelCheckpoint

# 项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_model import create_transformer_model, TransformerConfig
from models.cnn1d_model import create_cnn1d_model, compile_model, CNN1DConfig
from data.data_preprocessor import BSLDataPreprocessor


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str):
        """
        初始化训练日志记录器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_acc: float, val_acc: float, lr: float):
        """记录单个epoch的训练指标"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rate'].append(lr)
        
        logger.info(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                   f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, LR={lr:.6f}")
    
    def save_history(self, filename: str = "training_history.json"):
        """保存训练历史"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"训练历史已保存到: {filepath}")
    
    def plot_training_curves(self, save_path: str = None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 损失曲线
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 准确率曲线
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # 学习率曲线
        axes[2].plot(epochs, self.history['learning_rate'], 'g-', label='Learning Rate')
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练曲线已保存到: {save_path}")
        
        plt.show()


class TransformerTrainer:
    """Transformer模型训练器"""
    
    def __init__(self, config: TransformerConfig, log_dir: str):
        """
        初始化训练器
        
        Args:
            config: 模型配置
            log_dir: 日志目录
        """
        self.config = config
        self.log_dir = log_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = create_transformer_model(config).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 创建损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        # 训练记录器
        self.train_logger = TrainingLogger(log_dir)
        
        logger.info(f"Transformer训练器初始化完成，设备: {self.device}")
    
    def prepare_data(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """
        准备训练数据
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            Tuple[DataLoader, DataLoader]: 训练和验证数据加载器
        """
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"数据准备完成 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}")
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output['logits'], target)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output['logits'].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output['logits'], target)
                
                total_loss += loss.item()
                pred = output['logits'].argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            Dict: 训练结果
        """
        logger.info("开始训练Transformer模型...")
        
        # 准备数据
        train_loader, val_loader = self.prepare_data(X_train, y_train, X_val, y_val)
        
        # 学习率调度器
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs)
        
        # 早停
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(1, self.config.num_epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # 学习率调度
            scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录日志
            self.train_logger.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, current_lr)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                logger.info(f"早停触发，在第 {epoch} 轮停止训练")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # 保存训练历史和模型
        self.train_logger.save_history()
        self.save_model()
        
        logger.success("Transformer模型训练完成!")
        
        return {
            'best_val_loss': best_val_loss,
            'final_train_acc': self.train_logger.history['train_acc'][-1],
            'final_val_acc': self.train_logger.history['val_acc'][-1]
        }
    
    def save_model(self, filename: str = None):
        """保存模型"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transformer_model_{timestamp}.pth"
        
        filepath = os.path.join(self.log_dir, filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'model_info': self.model.get_model_info()
        }, filepath)
        
        logger.info(f"Transformer模型已保存到: {filepath}")


class CNN1DTrainer:
    """1D-CNN模型训练器"""
    
    def __init__(self, config: CNN1DConfig, log_dir: str):
        """
        初始化训练器
        
        Args:
            config: 模型配置
            log_dir: 日志目录
        """
        self.config = config
        self.log_dir = log_dir
        
        # 创建模型
        self.model = create_cnn1d_model(config)
        compile_model(self.model, config)
        
        # 训练记录器
        self.train_logger = TrainingLogger(log_dir)
        
        logger.info("1D-CNN训练器初始化完成")
    
    def create_callbacks(self) -> List:
        """创建训练回调"""
        callbacks = []
        
        # 早停
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # 学习率衰减
        lr_scheduler = TF_ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # 模型检查点
        checkpoint_path = os.path.join(self.log_dir, 'best_model.h5')
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        return callbacks
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            Dict: 训练结果
        """
        logger.info("开始训练1D-CNN模型...")
        
        # 创建回调
        callbacks = self.create_callbacks()
        
        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.num_epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # 更新训练记录器
        self.train_logger.history = {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'train_acc': history.history['accuracy'],
            'val_acc': history.history['val_accuracy'],
            'learning_rate': history.history.get('lr', [self.config.learning_rate] * len(history.history['loss']))
        }
        
        # 保存训练历史
        self.train_logger.save_history()
        self.save_model()
        
        logger.success("1D-CNN模型训练完成!")
        
        return {
            'best_val_loss': min(history.history['val_loss']),
            'final_train_acc': history.history['accuracy'][-1],
            'final_val_acc': history.history['val_accuracy'][-1]
        }
    
    def save_model(self, filename: str = None):
        """保存模型"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cnn1d_model_{timestamp}.h5"
        
        filepath = os.path.join(self.log_dir, filename)
        self.model.save(filepath)
        
        # 保存配置
        config_path = os.path.join(self.log_dir, f"cnn1d_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"1D-CNN模型已保存到: {filepath}")


def load_processed_data(data_dir: str, filename_prefix: str) -> Tuple[np.ndarray, ...]:
    """
    加载预处理后的数据
    
    Args:
        data_dir: 数据目录
        filename_prefix: 文件名前缀
        
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info(f"加载预处理数据: {data_dir}/{filename_prefix}")
    
    X_train = np.load(os.path.join(data_dir, f"{filename_prefix}_X_train.npy"))
    X_val = np.load(os.path.join(data_dir, f"{filename_prefix}_X_val.npy"))
    X_test = np.load(os.path.join(data_dir, f"{filename_prefix}_X_test.npy"))
    y_train = np.load(os.path.join(data_dir, f"{filename_prefix}_y_train.npy"))
    y_val = np.load(os.path.join(data_dir, f"{filename_prefix}_y_val.npy"))
    y_test = np.load(os.path.join(data_dir, f"{filename_prefix}_y_test.npy"))
    
    logger.info(f"数据加载完成:")
    logger.info(f"  训练集: {X_train.shape}, {y_train.shape}")
    logger.info(f"  验证集: {X_val.shape}, {y_val.shape}")
    logger.info(f"  测试集: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BSL手势识别模型训练')
    parser.add_argument('--model_type', choices=['transformer', 'cnn1d', 'both'], 
                       default='both', help='要训练的模型类型')
    parser.add_argument('--config', default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--data_dir', default='datasets/processed', help='数据目录')
    parser.add_argument('--data_prefix', required=True, help='数据文件前缀')
    parser.add_argument('--output_dir', default='models/trained', help='输出目录')
    parser.add_argument('--epochs', type=int, help='训练轮数（覆盖默认配置）')
    parser.add_argument('--batch_size', type=int, help='批次大小（覆盖默认配置）')
    parser.add_argument('--learning_rate', type=float, help='学习率（覆盖默认配置）')
    parser.add_argument('--visualize', action='store_true', help='显示训练曲线')
    
    args = parser.parse_args()
    
    # 加载配置文件
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {args.config}")
    except FileNotFoundError:
        logger.error(f"配置文件未找到: {args.config}")
        return
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(
            args.data_dir, args.data_prefix
        )
    except FileNotFoundError as e:
        logger.error(f"数据文件未找到: {e}")
        return
    
    results = {}
    
    # 训练Transformer模型
    if args.model_type in ['transformer', 'both']:
        logger.info("="*50)
        logger.info("开始训练Transformer模型")
        logger.info("="*50)
        
        # 创建配置
        transformer_config = TransformerConfig()
        transformer_config.num_classes = config_data['data']['num_classes']
        if args.epochs:
            transformer_config.num_epochs = args.epochs
        if args.batch_size:
            transformer_config.batch_size = args.batch_size
        if args.learning_rate:
            transformer_config.learning_rate = args.learning_rate
        
        # 创建训练器
        transformer_log_dir = os.path.join(args.output_dir, 'transformer')
        transformer_trainer = TransformerTrainer(transformer_config, transformer_log_dir)
        
        # 训练
        transformer_results = transformer_trainer.train(X_train, y_train, X_val, y_val)
        results['transformer'] = transformer_results
        
        # 可视化
        if args.visualize:
            plot_path = os.path.join(transformer_log_dir, 'training_curves.png')
            transformer_trainer.train_logger.plot_training_curves(plot_path)
    
    # 训练1D-CNN模型
    if args.model_type in ['cnn1d', 'both']:
        logger.info("="*50)
        logger.info("开始训练1D-CNN模型")
        logger.info("="*50)
        
        # 创建配置
        cnn1d_config = CNN1DConfig()
        cnn1d_config.num_classes = config_data['data']['num_classes']
        if args.epochs:
            cnn1d_config.num_epochs = args.epochs
        if args.batch_size:
            cnn1d_config.batch_size = args.batch_size
        if args.learning_rate:
            cnn1d_config.learning_rate = args.learning_rate
        
        # 创建训练器
        cnn1d_log_dir = os.path.join(args.output_dir, 'cnn1d')
        cnn1d_trainer = CNN1DTrainer(cnn1d_config, cnn1d_log_dir)
        
        # 训练
        cnn1d_results = cnn1d_trainer.train(X_train, y_train, X_val, y_val)
        results['cnn1d'] = cnn1d_results
        
        # 可视化
        if args.visualize:
            plot_path = os.path.join(cnn1d_log_dir, 'training_curves.png')
            cnn1d_trainer.train_logger.plot_training_curves(plot_path)
    
    # 保存训练结果摘要
    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.success("="*50)
    logger.success("所有模型训练完成!")
    logger.success("="*50)
    
    # 打印结果摘要
    for model_name, result in results.items():
        logger.info(f"{model_name.upper()} 结果:")
        logger.info(f"  最佳验证损失: {result['best_val_loss']:.4f}")
        logger.info(f"  最终训练准确率: {result['final_train_acc']:.4f}")
        logger.info(f"  最终验证准确率: {result['final_val_acc']:.4f}")


if __name__ == "__main__":
    main() 