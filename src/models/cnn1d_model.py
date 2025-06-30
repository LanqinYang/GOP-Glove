"""
BSL手势识别系统 - 1D-CNN模型（Arduino端）

该模块实现了轻量级的1D-CNN模型，用于Arduino端的实时手势识别。
模型特点：
- 小参数量，适合微控制器部署
- 高效的1D卷积操作
- 针对时间序列数据优化
- 支持TensorFlow Lite转换

作者: Lambert Yang
版本: 1.0
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Optional, Tuple, List
import json
import os
from loguru import logger


class Conv1DBlock(layers.Layer):
    """1D卷积块"""
    
    def __init__(self, filters: int, kernel_size: int = 3, 
                 activation: str = 'relu', dropout_rate: float = 0.2, 
                 use_batch_norm: bool = True, **kwargs):
        """
        初始化1D卷积块
        
        Args:
            filters: 卷积核数量
            kernel_size: 卷积核大小
            activation: 激活函数
            dropout_rate: Dropout比例
            use_batch_norm: 是否使用BatchNorm
        """
        super().__init__(**kwargs)
        
        self.conv1d = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation=None
        )
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = layers.BatchNormalization()
        
        self.activation = layers.Activation(activation)
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=None):
        """前向传播"""
        x = self.conv1d(inputs)
        
        if self.use_batch_norm:
            x = self.batch_norm(x, training=training)
        
        x = self.activation(x)
        x = self.dropout(x, training=training)
        
        return x


class BSL_CNN1D(keras.Model):
    """BSL手势识别1D-CNN模型"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int] = (100, 5),
                 num_classes: int = 11,
                 filters_list: List[int] = [32, 64, 128, 64],
                 kernel_sizes: List[int] = [7, 5, 3, 3],
                 pool_sizes: List[int] = [2, 2, 2, 2],
                 dropout_rate: float = 0.3,
                 dense_units: List[int] = [128, 64],
                 use_batch_norm: bool = True,
                 **kwargs):
        """
        初始化1D-CNN模型
        
        Args:
            input_shape: 输入形状 (sequence_length, num_sensors)
            num_classes: 分类类别数
            filters_list: 各层卷积核数量
            kernel_sizes: 各层卷积核大小
            pool_sizes: 各层池化大小
            dropout_rate: Dropout比例
            dense_units: 全连接层单元数
            use_batch_norm: 是否使用BatchNorm
        """
        super().__init__(**kwargs)
        
        self.input_shape_param = input_shape
        self.num_classes = num_classes
        self.filters_list = filters_list
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        
        # 构建网络层
        self.conv_blocks = []
        self.pool_layers = []
        
        # 卷积层
        for i, (filters, kernel_size, pool_size) in enumerate(
            zip(filters_list, kernel_sizes, pool_sizes)):
            
            # 卷积块
            conv_block = Conv1DBlock(
                filters=filters,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                name=f'conv_block_{i+1}'
            )
            self.conv_blocks.append(conv_block)
            
            # 池化层
            if pool_size > 1:
                pool_layer = layers.MaxPooling1D(
                    pool_size=pool_size,
                    name=f'pool_{i+1}'
                )
                self.pool_layers.append(pool_layer)
            else:
                self.pool_layers.append(None)
        
        # 全局平均池化
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        
        # 全连接层
        self.dense_layers = []
        for i, units in enumerate(dense_units):
            dense_layer = layers.Dense(
                units=units,
                activation='relu',
                name=f'dense_{i+1}'
            )
            self.dense_layers.append(dense_layer)
            
            # Dropout层
            dropout_layer = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')
            self.dense_layers.append(dropout_layer)
        
        # 输出层
        self.output_layer = layers.Dense(
            units=num_classes,
            activation='softmax',
            name='predictions'
        )
        
        logger.info(f"1D-CNN模型初始化完成:")
        logger.info(f"  输入形状: {input_shape}")
        logger.info(f"  卷积层配置: {list(zip(filters_list, kernel_sizes))}")
        logger.info(f"  全连接层: {dense_units}")
        logger.info(f"  类别数: {num_classes}")
    
    def call(self, inputs, training=None):
        """前向传播"""
        x = inputs
        
        # 卷积层
        for conv_block, pool_layer in zip(self.conv_blocks, self.pool_layers):
            x = conv_block(x, training=training)
            if pool_layer is not None:
                x = pool_layer(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        
        # 全连接层
        for dense_layer in self.dense_layers:
            x = dense_layer(x, training=training)
        
        # 输出层
        outputs = self.output_layer(x)
        
        return outputs
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_name': 'BSL_CNN1D',
            'input_shape': self.input_shape_param,
            'num_classes': self.num_classes,
            'filters_list': self.filters_list,
            'kernel_sizes': self.kernel_sizes,
            'pool_sizes': self.pool_sizes
        }


class CNN1DConfig:
    """1D-CNN模型配置类"""
    
    def __init__(self):
        # 模型架构参数
        self.input_shape = (100, 5)
        self.num_classes = 11
        self.filters_list = [32, 64, 128, 64]
        self.kernel_sizes = [7, 5, 3, 3]
        self.pool_sizes = [2, 2, 2, 2]
        self.dropout_rate = 0.3
        self.dense_units = [128, 64]
        self.use_batch_norm = True
        
        # 训练参数
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.num_epochs = 150
        self.patience = 20
        
        # 优化器参数
        self.optimizer_type = 'Adam'
        self.loss_function = 'sparse_categorical_crossentropy'
        
        # 数据参数
        self.validation_split = 0.2
        
        # TensorFlow Lite参数
        self.quantization = True
        self.representative_dataset_size = 100
        
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """从字典创建配置"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


def create_cnn1d_model(config: CNN1DConfig) -> BSL_CNN1D:
    """
    创建1D-CNN模型
    
    Args:
        config: 模型配置
        
    Returns:
        BSL_CNN1D: 创建的模型
    """
    model = BSL_CNN1D(
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        filters_list=config.filters_list,
        kernel_sizes=config.kernel_sizes,
        pool_sizes=config.pool_sizes,
        dropout_rate=config.dropout_rate,
        dense_units=config.dense_units,
        use_batch_norm=config.use_batch_norm
    )
    
    # 构建模型（需要调用一次以初始化权重）
    dummy_input = tf.random.normal((1,) + config.input_shape)
    _ = model(dummy_input)
    
    logger.info(f"创建1D-CNN模型完成，参数量: {model.count_params():,}")
    return model


def compile_model(model: BSL_CNN1D, config: CNN1DConfig) -> BSL_CNN1D:
    """
    编译模型
    
    Args:
        model: 模型实例
        config: 配置
        
    Returns:
        BSL_CNN1D: 编译后的模型
    """
    # 选择优化器
    if config.optimizer_type == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    elif config.optimizer_type == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=0.9)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss=config.loss_function,
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
    )
    
    logger.info("模型编译完成")
    return model


def convert_to_tflite(model: keras.Model, 
                     representative_dataset: Optional[np.ndarray] = None,
                     quantize: bool = True,
                     output_path: str = "model.tflite") -> str:
    """
    将Keras模型转换为TensorFlow Lite格式
    
    Args:
        model: Keras模型
        representative_dataset: 代表性数据集用于量化
        quantize: 是否进行量化
        output_path: 输出路径
        
    Returns:
        str: 输出文件路径
    """
    logger.info("开始转换模型为TensorFlow Lite格式...")
    
    # 创建转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize and representative_dataset is not None:
        # 设置量化选项
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: _representative_data_gen(representative_dataset)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        logger.info("启用INT8量化")
    
    # 转换模型
    tflite_model = converter.convert()
    
    # 保存模型
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # 获取模型大小
    model_size = len(tflite_model)
    logger.success(f"TensorFlow Lite模型已保存: {output_path}")
    logger.info(f"模型大小: {model_size / 1024:.2f} KB")
    
    return output_path


def _representative_data_gen(representative_dataset: np.ndarray):
    """代表性数据生成器（用于量化）"""
    for i in range(min(100, len(representative_dataset))):
        yield [representative_dataset[i:i+1].astype(np.float32)]


def generate_c_array(tflite_model_path: str, output_path: str = "model_data.h") -> str:
    """
    将TensorFlow Lite模型转换为C数组
    
    Args:
        tflite_model_path: TFLite模型路径
        output_path: 输出头文件路径
        
    Returns:
        str: 输出文件路径
    """
    logger.info("生成C数组头文件...")
    
    # 读取模型文件
    with open(tflite_model_path, 'rb') as f:
        model_data = f.read()
    
    # 生成C数组
    array_name = "bsl_model_data"
    array_size = len(model_data)
    
    c_code = f"""// BSL手势识别模型数据
// 自动生成文件，请勿手动修改

#ifndef BSL_MODEL_DATA_H
#define BSL_MODEL_DATA_H

// 模型大小: {array_size} bytes
const unsigned int {array_name}_len = {array_size};

// 模型数据
const unsigned char {array_name}[] = {{
"""
    
    # 添加数据
    for i, byte in enumerate(model_data):
        if i % 16 == 0:
            c_code += "\n  "
        c_code += f"0x{byte:02x},"
        if i < len(model_data) - 1 and (i + 1) % 16 != 0:
            c_code += " "
    
    c_code += """
};

#endif // BSL_MODEL_DATA_H
"""
    
    # 保存文件
    with open(output_path, 'w') as f:
        f.write(c_code)
    
    logger.success(f"C数组头文件已生成: {output_path}")
    logger.info(f"数组大小: {array_size} bytes")
    
    return output_path


def create_lightweight_model() -> BSL_CNN1D:
    """创建针对Arduino优化的超轻量级模型"""
    config = CNN1DConfig()
    
    # 超轻量级配置
    config.filters_list = [16, 32, 16]
    config.kernel_sizes = [5, 3, 3]
    config.pool_sizes = [2, 2, 2]
    config.dense_units = [32]
    config.dropout_rate = 0.2
    
    model = create_cnn1d_model(config)
    compile_model(model, config)
    
    logger.info("创建超轻量级模型完成")
    return model


def test_cnn1d_model():
    """测试1D-CNN模型"""
    logger.info("开始测试1D-CNN模型...")
    
    # 创建配置
    config = CNN1DConfig()
    
    # 创建模型
    model = create_cnn1d_model(config)
    compile_model(model, config)
    
    # 创建测试数据
    batch_size = 8
    X_test = np.random.randn(batch_size, 100, 5).astype(np.float32)
    y_test = np.random.randint(0, 11, batch_size)
    
    # 测试推理
    predictions = model.predict(X_test)
    logger.info(f"输入形状: {X_test.shape}")
    logger.info(f"输出形状: {predictions.shape}")
    logger.info(f"预测类别: {np.argmax(predictions, axis=1)}")
    
    # 测试训练（一个batch）
    loss = model.train_on_batch(X_test, y_test)
    logger.info(f"训练损失: {loss}")
    
    # 测试TensorFlow Lite转换
    tflite_path = "test_model.tflite"
    convert_to_tflite(model, X_test, quantize=False, output_path=tflite_path)
    
    # 测试量化转换
    tflite_quantized_path = "test_model_quantized.tflite"
    convert_to_tflite(model, X_test, quantize=True, output_path=tflite_quantized_path)
    
    # 生成C数组
    c_array_path = "test_model_data.h"
    generate_c_array(tflite_quantized_path, c_array_path)
    
    # 清理测试文件
    for file_path in [tflite_path, tflite_quantized_path, c_array_path]:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    logger.success("1D-CNN模型测试通过!")


if __name__ == "__main__":
    test_cnn1d_model() 