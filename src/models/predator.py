"""
Predator 模型实现
基于 Transformer 的点云配准网络
论文：https://arxiv.org/abs/2108.03279
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class ResUNetBlock(nn.Module):
    """ResUNet 残差块"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = None
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.skip is not None:
            identity = self.skip(identity)
        
        out += identity
        out = self.relu(out)
        
        return out


class Predator(nn.Module):
    """
    Predator: Point Cloud Registration Network
    
    架构:
    1. ResUNet 编码器 - 提取多尺度特征
    2. Transformer 编码器 - 捕获全局上下文
    3. 匹配头 - 预测重叠区域和变换参数
    """
    
    def __init__(self, config: Dict):
        super(Predator, self).__init__()
        self.config = config
        
        # 编码器参数
        planes = config.get('UNET', {}).get('PLANES', [64, 128, 256, 512])
        input_dim = config.get('INPUT_DIM', 3)
        
        # ResUNet 编码器
        self.encoders = nn.ModuleList()
        for i, plane in enumerate(planes):
            in_ch = input_dim if i == 0 else planes[i-1]
            self.encoders.append(ResUNetBlock(in_ch, plane))
        
        # 投影到 transformer 维度
        hidden_dim = config.get('TRANSFORMER', {}).get('HIDDEN_DIM', 512)
        self.project_to_transformer = nn.Linear(planes[-1], hidden_dim)
        
        # Transformer 编码器
        transformer_config = config.get('TRANSFORMER', {})
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_config.get('HIDDEN_DIM', 512),
            nhead=transformer_config.get('NUM_HEADS', 8),
            dim_feedforward=transformer_config.get('HIDDEN_DIM', 512) * 4,
            dropout=transformer_config.get('DROPOUT', 0.1),
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_config.get('NUM_LAYERS', 6)
        )
        
        # 匹配头
        matching_config = config.get('MATCHING', {})
        feature_dim = matching_config.get('FEATURE_DIM', 256)
        hidden_dim = transformer_config.get('HIDDEN_DIM', 512)
        
        self.feature_head = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 重叠预测头
        self.overlap_head = nn.Linear(feature_dim, 1)
        
        # 变换回归头
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 3 旋转 + 3 平移
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, points1: torch.Tensor, points2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            points1: 源点云 [B, N, 3]
            points2: 目标点云 [B, M, 3]
            
        Returns:
            features1: 源点云特征 [B, N, D]
            features2: 目标点云特征 [B, M, D]
            transform_pred: 预测的变换参数 [B, 6]
        """
        # 转置为 [B, 3, N] 以适应 Conv1d
        x1 = points1.transpose(1, 2)
        x2 = points2.transpose(1, 2)
        
        # 编码
        for encoder in self.encoders:
            x1 = encoder(x1)
            x2 = encoder(x2)
        
        # 转换为 Transformer 输入格式 [B, N, D]
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)
        
        # 投影到 transformer 维度
        x1 = self.project_to_transformer(x1)
        x2 = self.project_to_transformer(x2)
        
        # Transformer 编码
        x1 = self.transformer(x1)
        x2 = self.transformer(x2)
        
        # 特征投影
        features1 = self.feature_head(x1)
        features2 = self.feature_head(x2)
        
        # 全局池化
        global_feat1 = features1.max(dim=1)[0]
        global_feat2 = features2.max(dim=1)[0]
        
        # 预测变换参数
        combined = torch.cat([global_feat1, global_feat2], dim=-1)
        transform_pred = self.regressor(combined)
        
        return features1, features2, transform_pred
    
    def predict_transform(self, points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        """
        预测变换矩阵
        
        Args:
            points1: 源点云 [B, N, 3]
            points2: 目标点云 [B, M, 3]
            
        Returns:
            transform_matrix: 4x4 变换矩阵 [B, 4, 4]
        """
        _, _, transform_params = self.forward(points1, points2)
        
        # 将 6 参数转换为 4x4 变换矩阵
        batch_size = transform_params.shape[0]
        transform_matrices = []
        
        for i in range(batch_size):
            params = transform_params[i]
            rotation = params[:3]
            translation = params[3:]
            
            # 简化的旋转表示（实际应该使用更复杂的表示）
            # 这里使用轴角表示
            angle = torch.norm(rotation)
            axis = rotation / (angle + 1e-8)
            
            # 构建旋转矩阵（简化版本）
            R = self._axis_angle_to_matrix(axis, angle)
            t = translation
            
            # 构建 4x4 变换矩阵
            T = torch.eye(4, device=transform_params.device)
            T[:3, :3] = R
            T[:3, 3] = t
            
            transform_matrices.append(T)
        
        return torch.stack(transform_matrices)
    
    def _axis_angle_to_matrix(self, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """轴角表示转换为旋转矩阵"""
        # 归一化轴
        axis = axis / torch.norm(axis)
        
        # Rodrigues 公式
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=axis.device)
        
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        R = cos_a * torch.eye(3, device=axis.device) + \
            sin_a * K + \
            (1 - cos_a) * torch.ger(axis, axis)
        
        return R


def test_predator():
    """测试 Predator 模型"""
    print("测试 Predator 模型...")
    
    config = {
        'MODEL': {
            'UNET': {'PLANES': [64, 128, 256, 512]},
            'INPUT_DIM': 3,
            'TRANSFORMER': {
                'NUM_LAYERS': 2,
                'NUM_HEADS': 4,
                'HIDDEN_DIM': 256
            },
            'MATCHING': {'FEATURE_DIM': 128}
        }
    }
    
    model = Predator(config['MODEL'])
    
    # 创建测试数据
    batch_size = 2
    points1 = torch.randn(batch_size, 100, 3)
    points2 = torch.randn(batch_size, 100, 3)
    
    # 前向传播
    with torch.no_grad():
        features1, features2, transform_pred = model(points1, points2)
        transform_matrix = model.predict_transform(points1, points2)
    
    print(f"✓ 输入形状：{points1.shape}")
    print(f"✓ 输出特征形状：{features1.shape}, {features2.shape}")
    print(f"✓ 预测变换形状：{transform_pred.shape}")
    print(f"✓ 变换矩阵形状：{transform_matrix.shape}")
    
    print("\nPredator 模型测试通过！")
    return model


if __name__ == '__main__':
    test_predator()
