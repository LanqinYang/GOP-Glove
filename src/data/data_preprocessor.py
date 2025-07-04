"""
BSL手势识别系统 - 数据预处理模块

该模块负责：
1. 数据加载和验证
2. 数据增强（抖动、缩放）
3. 数据归一化和标准化
4. 数据集分割（训练/验证/测试）
5. 数据格式转换

作者: Lambert Yang
版本: 1.0
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, List, Optional
import json
import os
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns


class BSLDataPreprocessor:
    """BSL手势数据预处理器"""
    
    def __init__(self):
        """初始化预处理器"""
        self.scaler = None
        self.gesture_classes = {
            0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
            5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
        }
        self.num_classes = len(self.gesture_classes)
        
        logger.info("数据预处理器初始化完成")
    
    def load_dataset(self, data_dir: str, filename_prefix: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        加载数据集
        
        Args:
            data_dir: 数据目录
            filename_prefix: 文件名前缀
            
        Returns:
            Tuple[X, y, metadata]: 特征数组、标签数组、元数据
        """
        X_path = os.path.join(data_dir, f"{filename_prefix}_X.npy")
        y_path = os.path.join(data_dir, f"{filename_prefix}_y.npy")
        metadata_path = os.path.join(data_dir, f"{filename_prefix}_metadata.json")
        
        # 检查文件存在性
        for path in [X_path, y_path, metadata_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"找不到文件: {path}")
        
        # 加载数据
        X = np.load(X_path)
        y = np.load(y_path)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"成功加载数据集: {X.shape}, 标签: {y.shape}")
        logger.info(f"数据集信息: {metadata['dataset_info']}")
        
        return X, y, metadata
    
    def validate_data(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        验证数据集的完整性和格式
        
        Args:
            X: 特征数组 (N, 100, 5)
            y: 标签数组 (N,)
            
        Returns:
            bool: 数据是否有效
        """
        logger.info("开始数据验证...")
        
        # 检查维度
        if len(X.shape) != 3:
            logger.error(f"X维度错误: 期望3维，实际{len(X.shape)}维")
            return False
        
        if X.shape[1] != 100:
            logger.warning(f"序列长度异常: 期望100，实际{X.shape[1]}")
        
        if X.shape[2] != 5:
            logger.error(f"传感器数量错误: 期望5个，实际{X.shape[2]}个")
            return False
        
        # 检查数据类型
        if not np.isfinite(X).all():
            logger.error("X包含无穷大或NaN值")
            return False
        
        # 检查标签范围
        unique_labels = np.unique(y)
        expected_labels = set(range(self.num_classes))
        actual_labels = set(unique_labels)
        
        if not actual_labels.issubset(expected_labels):
            logger.error(f"标签超出范围: {actual_labels - expected_labels}")
            return False
        
        # 检查样本数量一致性
        if X.shape[0] != y.shape[0]:
            logger.error(f"样本数量不一致: X={X.shape[0]}, y={y.shape[0]}")
            return False
        
        logger.success("数据验证通过")
        return True
    
    def analyze_data_distribution(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        分析数据分布
        
        Args:
            X: 特征数组
            y: 标签数组
            
        Returns:
            Dict: 分析结果
        """
        analysis = {}
        
        # 类别分布
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        analysis['class_distribution'] = class_distribution
        
        # 传感器数据统计
        sensor_stats = {}
        for sensor_idx in range(X.shape[2]):
            sensor_data = X[:, :, sensor_idx].flatten()
            sensor_stats[f'sensor_{sensor_idx + 1}'] = {
                'mean': float(sensor_data.mean()),
                'std': float(sensor_data.std()),
                'min': float(sensor_data.min()),
                'max': float(sensor_data.max()),
                'median': float(np.median(sensor_data))
            }
        analysis['sensor_stats'] = sensor_stats
        
        # 数据质量指标
        analysis['data_quality'] = {
            'total_samples': len(y),
            'sequence_length': X.shape[1],
            'num_sensors': X.shape[2],
            'missing_values': int(np.isnan(X).sum()),
            'infinite_values': int(np.isinf(X).sum())
        }
        
        logger.info("数据分布分析完成")
        return analysis
    
    def apply_jittering(self, X: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        应用抖动数据增强
        
        Args:
            X: 输入数据 (N, 100, 5)
            noise_level: 噪声水平
            
        Returns:
            np.ndarray: 增强后的数据
        """
        X_jittered = X.copy()
        
        # 为每个样本添加不同的随机噪声
        for i in range(X.shape[0]):
            noise = np.random.normal(0, noise_level * X[i].std(), X[i].shape)
            X_jittered[i] += noise
        
        logger.info(f"抖动增强完成: 噪声水平={noise_level}")
        return X_jittered
    
    def apply_scaling(self, X: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        应用缩放数据增强
        
        Args:
            X: 输入数据 (N, 100, 5)
            scale_range: 缩放范围
            
        Returns:
            np.ndarray: 增强后的数据
        """
        X_scaled = X.copy()
        
        # 为每个样本应用随机缩放
        for i in range(X.shape[0]):
            scale_factor = np.random.uniform(scale_range[0], scale_range[1])
            X_scaled[i] *= scale_factor
        
        logger.info(f"缩放增强完成: 缩放范围={scale_range}")
        return X_scaled
    
    def augment_data(self, X: np.ndarray, y: np.ndarray, 
                    augment_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        综合数据增强
        
        Args:
            X: 输入数据
            y: 标签
            augment_factor: 增强倍数
            
        Returns:
            Tuple[X_aug, y_aug]: 增强后的数据和标签
        """
        logger.info(f"开始数据增强: 增强倍数={augment_factor}")
        
        X_augmented = [X]
        y_augmented = [y]
        
        for i in range(augment_factor):
            # 抖动增强
            X_jittered = self.apply_jittering(X, noise_level=0.01 * (i + 1))
            X_augmented.append(X_jittered)
            y_augmented.append(y)
            
            # 缩放增强
            X_scaled = self.apply_scaling(X, scale_range=(0.9 - i*0.05, 1.1 + i*0.05))
            X_augmented.append(X_scaled)
            y_augmented.append(y)
        
        X_final = np.concatenate(X_augmented, axis=0)
        y_final = np.concatenate(y_augmented, axis=0)
        
        logger.success(f"数据增强完成: {X.shape} -> {X_final.shape}")
        return X_final, y_final
    
    def normalize_data(self, X_train: np.ndarray, X_val: np.ndarray = None, 
                      X_test: np.ndarray = None, method: str = 'standard') -> Tuple:
        """
        数据归一化
        
        Args:
            X_train: 训练数据
            X_val: 验证数据
            X_test: 测试数据
            method: 归一化方法 ('standard', 'minmax')
            
        Returns:
            Tuple: 归一化后的数据
        """
        logger.info(f"开始数据归一化: 方法={method}")
        
        # 重塑数据用于归一化 (N*100, 5)
        original_shape = X_train.shape
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        
        # 选择归一化器
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
        
        # 拟合和转换训练数据
        X_train_normalized = self.scaler.fit_transform(X_train_reshaped)
        X_train_normalized = X_train_normalized.reshape(original_shape)
        
        results = [X_train_normalized]
        
        # 处理验证和测试数据
        for X_data in [X_val, X_test]:
            if X_data is not None:
                X_reshaped = X_data.reshape(-1, X_data.shape[-1])
                X_normalized = self.scaler.transform(X_reshaped)
                X_normalized = X_normalized.reshape(X_data.shape)
                results.append(X_normalized)
            else:
                results.append(None)
        
        logger.success("数据归一化完成")
        return tuple(results)
    
    def split_dataset(self, X: np.ndarray, y: np.ndarray, 
                     train_size: float = 0.7, val_size: float = 0.15, 
                     test_size: float = 0.15, random_state: int = 42) -> Tuple:
        """
        分割数据集
        
        Args:
            X: 特征数组
            y: 标签数组
            train_size: 训练集比例
            val_size: 验证集比例
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # 验证比例
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size 必须等于 1.0")
        
        logger.info(f"分割数据集: 训练={train_size}, 验证={val_size}, 测试={test_size}")
        
        # 第一次分割：分离出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 第二次分割：从剩余数据中分离训练集和验证集
        val_size_adjusted = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"数据集分割完成:")
        logger.info(f"  训练集: {X_train.shape}")
        logger.info(f"  验证集: {X_val.shape}")
        logger.info(f"  测试集: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_processed_dataset(self, data_dir: str, filename_prefix: str,
                               output_dir: str, augment_factor: int = 2,
                               normalize_method: str = 'standard') -> str:
        """
        创建完整的预处理数据集
        
        Args:
            data_dir: 原始数据目录
            filename_prefix: 文件名前缀
            output_dir: 输出目录
            augment_factor: 数据增强倍数
            normalize_method: 归一化方法
            
        Returns:
            str: 输出文件前缀
        """
        logger.info("开始创建预处理数据集")
        
        # 加载原始数据
        X, y, metadata = self.load_dataset(data_dir, filename_prefix)
        
        # 验证数据
        if not self.validate_data(X, y):
            raise ValueError("数据验证失败")
        
        # 分析数据分布
        analysis = self.analyze_data_distribution(X, y)
        
        # 数据增强
        if augment_factor > 0:
            X_augmented, y_augmented = self.augment_data(X, y, augment_factor)
        else:
            X_augmented, y_augmented = X, y
        
        # 分割数据集
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(
            X_augmented, y_augmented
        )
        
        # 数据归一化
        X_train_norm, X_val_norm, X_test_norm = self.normalize_data(
            X_train, X_val, X_test, method=normalize_method
        )
        
        # 保存处理后的数据
        os.makedirs(output_dir, exist_ok=True)
        output_prefix = f"{filename_prefix}_processed"
        
        # 保存numpy数组
        np.save(os.path.join(output_dir, f"{output_prefix}_X_train.npy"), X_train_norm)
        np.save(os.path.join(output_dir, f"{output_prefix}_X_val.npy"), X_val_norm)
        np.save(os.path.join(output_dir, f"{output_prefix}_X_test.npy"), X_test_norm)
        np.save(os.path.join(output_dir, f"{output_prefix}_y_train.npy"), y_train)
        np.save(os.path.join(output_dir, f"{output_prefix}_y_val.npy"), y_val)
        np.save(os.path.join(output_dir, f"{output_prefix}_y_test.npy"), y_test)
        
        # 保存预处理器
        import pickle
        with open(os.path.join(output_dir, f"{output_prefix}_scaler.pkl"), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 更新元数据
        processed_metadata = metadata.copy()
        processed_metadata['preprocessing'] = {
            'augment_factor': augment_factor,
            'normalize_method': normalize_method,
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'original_samples': len(y),
            'augmented_samples': len(y_augmented)
        }
        processed_metadata['data_analysis'] = analysis
        
        with open(os.path.join(output_dir, f"{output_prefix}_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(processed_metadata, f, ensure_ascii=False, indent=2)
        
        logger.success(f"预处理数据集已保存到: {output_dir}")
        return output_prefix
    
    def visualize_data_distribution(self, X: np.ndarray, y: np.ndarray, 
                                  save_path: str = None):
        """
        可视化数据分布
        
        Args:
            X: 特征数据
            y: 标签
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('BSL数据集分布分析', fontsize=16)
        
        # 1. 类别分布
        unique, counts = np.unique(y, return_counts=True)
        gesture_names = [self.gesture_classes[i] for i in unique]
        
        axes[0, 0].bar(gesture_names, counts)
        axes[0, 0].set_title('类别分布')
        axes[0, 0].set_xlabel('手势类别')
        axes[0, 0].set_ylabel('样本数量')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 传感器数据分布
        for sensor_idx in range(min(5, X.shape[2])):
            sensor_data = X[:, :, sensor_idx].flatten()
            axes[0, 1].hist(sensor_data, bins=50, alpha=0.7, 
                          label=f'传感器{sensor_idx+1}')
        axes[0, 1].set_title('传感器数据分布')
        axes[0, 1].set_xlabel('传感器值')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].legend()
        
        # 3. 传感器相关性矩阵
        sensor_means = X.mean(axis=1)  # (N, 5)
        correlation_matrix = np.corrcoef(sensor_means.T)
        im = axes[0, 2].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[0, 2].set_title('传感器相关性')
        axes[0, 2].set_xticks(range(5))
        axes[0, 2].set_yticks(range(5))
        axes[0, 2].set_xticklabels([f'S{i+1}' for i in range(5)])
        axes[0, 2].set_yticklabels([f'S{i+1}' for i in range(5)])
        plt.colorbar(im, ax=axes[0, 2])
        
        # 4. 时间序列示例
        sample_indices = np.random.choice(len(X), 3, replace=False)
        for i, idx in enumerate(sample_indices):
            for sensor_idx in range(X.shape[2]):
                axes[1, 0].plot(X[idx, :, sensor_idx], 
                              label=f'样本{i+1}-S{sensor_idx+1}', alpha=0.7)
        axes[1, 0].set_title('时间序列示例')
        axes[1, 0].set_xlabel('时间步')
        axes[1, 0].set_ylabel('传感器值')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 5. 数据质量指标
        quality_metrics = {
            '缺失值': np.isnan(X).sum(),
            '无穷值': np.isinf(X).sum(),
            '零值比例': (X == 0).sum() / X.size * 100
        }
        
        metrics_names = list(quality_metrics.keys())
        metrics_values = list(quality_metrics.values())
        
        axes[1, 1].bar(metrics_names, metrics_values)
        axes[1, 1].set_title('数据质量指标')
        axes[1, 1].set_ylabel('数量/百分比')
        
        # 6. 各类别传感器统计
        class_sensor_means = []
        for class_id in unique:
            class_mask = (y == class_id)
            class_data = X[class_mask].mean(axis=(0, 1))  # 对样本和时间维度求平均
            class_sensor_means.append(class_data)
        
        class_sensor_means = np.array(class_sensor_means)
        im2 = axes[1, 2].imshow(class_sensor_means.T, cmap='viridis', aspect='auto')
        axes[1, 2].set_title('各类别传感器平均值')
        axes[1, 2].set_xlabel('手势类别')
        axes[1, 2].set_ylabel('传感器')
        axes[1, 2].set_xticks(range(len(unique)))
        axes[1, 2].set_xticklabels([self.gesture_classes[i] for i in unique], rotation=45)
        axes[1, 2].set_yticks(range(5))
        axes[1, 2].set_yticklabels([f'S{i+1}' for i in range(5)])
        plt.colorbar(im2, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"数据分布图已保存到: {save_path}")
        
        plt.show()


def main():
    """主函数 - 数据预处理流水线"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BSL数据预处理器')
    parser.add_argument('--input_dir', default='datasets/raw', help='输入数据目录')
    parser.add_argument('--output_dir', default='datasets/processed', help='输出数据目录')
    parser.add_argument('--filename_prefix', required=True, help='输入文件名前缀')
    parser.add_argument('--augment_factor', type=int, default=2, help='数据增强倍数')
    parser.add_argument('--normalize_method', default='standard', choices=['standard', 'minmax'], help='归一化方法')
    parser.add_argument('--visualize', action='store_true', help='生成数据可视化')
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = BSLDataPreprocessor()
    
    try:
        # 执行预处理
        output_prefix = preprocessor.create_processed_dataset(
            data_dir=args.input_dir,
            filename_prefix=args.filename_prefix,
            output_dir=args.output_dir,
            augment_factor=args.augment_factor,
            normalize_method=args.normalize_method
        )
        
        # 可视化（如果需要）
        if args.visualize:
            X, y, _ = preprocessor.load_dataset(args.input_dir, args.filename_prefix)
            save_path = os.path.join(args.output_dir, f"{output_prefix}_distribution.png")
            preprocessor.visualize_data_distribution(X, y, save_path)
        
        logger.success(f"数据预处理完成! 输出前缀: {output_prefix}")
        
    except Exception as e:
        logger.error(f"预处理失败: {e}")
        raise


if __name__ == "__main__":
    main() 