"""
BSL手势识别系统 - Transformer模型（PC端）

该模块实现了基于Encoder-Only架构的Transformer模型，用于高精度手势识别。
模型架构包含：
- 输入嵌入层
- 位置编码
- 多层Transformer编码器
- 分类头

作者: Lambert Yang
版本: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
import numpy as np
from loguru import logger


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比例
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算位置编码
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 注册为buffer，不需要梯度更新
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (seq_len, batch_size, d_model)
            
        Returns:
            torch.Tensor: 添加位置编码后的张量
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """基于Transformer的手势分类器"""
    
    def __init__(self, 
                 input_dim: int = 5,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 num_classes: int = 10,
                 max_seq_len: int = 100):
        """
        初始化Transformer分类器
        
        Args:
            input_dim: 输入特征维度（传感器数量）
            d_model: Transformer模型维度
            nhead: 多头注意力头数
            num_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比例
            num_classes: 分类类别数
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 输入投影层：将传感器数据映射到d_model维度
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False  # (seq, batch, feature)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"Transformer模型初始化完成:")
        logger.info(f"  输入维度: {input_dim}")
        logger.info(f"  模型维度: {d_model}")
        logger.info(f"  注意力头数: {nhead}")
        logger.info(f"  编码器层数: {num_layers}")
        logger.info(f"  类别数: {num_classes}")
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, 
                src_key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
            src_key_padding_mask: 填充掩码 (batch_size, seq_len)
            
        Returns:
            Dict: 包含logits和attention_weights的字典
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # (batch, seq, d_model)
        x = x * math.sqrt(self.d_model)  # 缩放
        
        # 转换为Transformer期望的格式 (seq, batch, d_model)
        x = x.transpose(0, 1)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        encoded = self.transformer_encoder(
            x, 
            src_key_padding_mask=src_key_padding_mask
        )  # (seq, batch, d_model)
        
        # 全局平均池化
        if src_key_padding_mask is not None:
            # 处理填充的情况
            mask = ~src_key_padding_mask.transpose(0, 1).unsqueeze(-1)  # (seq, batch, 1)
            encoded_masked = encoded * mask
            pooled = encoded_masked.sum(dim=0) / mask.sum(dim=0)  # (batch, d_model)
        else:
            pooled = encoded.mean(dim=0)  # (batch, d_model)
        
        # 分类
        logits = self.classifier(pooled)  # (batch, num_classes)
        
        return {
            'logits': logits,
            'features': pooled,
            'encoded': encoded.transpose(0, 1)  # 返回batch_first格式
        }
    
    def get_attention_weights(self, x: torch.Tensor, 
                            layer_idx: int = -1) -> torch.Tensor:
        """
        获取注意力权重（用于可视化）
        
        Args:
            x: 输入张量
            layer_idx: 层索引，-1表示最后一层
            
        Returns:
            torch.Tensor: 注意力权重
        """
        # 注意：这需要修改Transformer实现以返回attention weights
        # 这里提供一个简化的实现
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            # 实际实现中需要从transformer_encoder中提取attention weights
            return None
    
    def count_parameters(self) -> int:
        """统计模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_name': 'TransformerClassifier',
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'num_classes': self.num_classes,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class TransformerConfig:
    """Transformer模型配置类"""
    
    def __init__(self):
        # 模型架构参数
        self.input_dim = 5
        self.d_model = 128
        self.nhead = 8
        self.num_layers = 6
        self.dim_feedforward = 512
        self.dropout = 0.1
        self.num_classes = 10
        self.max_seq_len = 100
        
        # 训练参数
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.batch_size = 32
        self.num_epochs = 100
        self.patience = 15
        
        # 优化器参数
        self.optimizer_type = 'AdamW'
        self.scheduler_type = 'CosineAnnealing'
        self.warmup_steps = 1000
        
        # 数据参数
        self.label_smoothing = 0.1
        
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


def create_transformer_model(config: TransformerConfig) -> TransformerClassifier:
    """
    创建Transformer模型
    
    Args:
        config: 模型配置
        
    Returns:
        TransformerClassifier: 创建的模型
    """
    model = TransformerClassifier(
        input_dim=config.input_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        num_classes=config.num_classes,
        max_seq_len=config.max_seq_len
    )
    
    logger.info(f"创建Transformer模型完成，参数量: {model.count_parameters():,}")
    return model


def test_transformer_model():
    """测试Transformer模型"""
    logger.info("开始测试Transformer模型...")
    
    # 创建配置
    config = TransformerConfig()
    
    # 创建模型
    model = create_transformer_model(config)
    
    # 创建测试数据
    batch_size = 4
    seq_len = 100
    input_dim = 5
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    logger.info(f"输入形状: {x.shape}")
    logger.info(f"输出logits形状: {output['logits'].shape}")
    logger.info(f"特征形状: {output['features'].shape}")
    logger.info(f"模型信息: {model.get_model_info()}")
    
    # 测试梯度
    model.train()
    output = model(x)
    loss = F.cross_entropy(output['logits'], torch.randint(0, 11, (batch_size,)))
    loss.backward()
    
    logger.success("Transformer模型测试通过!")


if __name__ == "__main__":
    test_transformer_model() 