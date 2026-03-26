"""
DCP (Deep Closest Point) 模型实现
基于 Transformer 的点云配准网络
论文：https://arxiv.org/abs/1903.07600
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict


class PointNetEncoder(nn.Module):
    """PointNet 特征编码器"""
    
    def __init__(self, embedding_dim=1024):
        super(PointNetEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv5 = nn.Conv1d(256, embedding_dim, kernel_size=1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(embedding_dim)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 点云 [B, N, 3]
        Returns:
            全局特征 [B, embedding_dim]
        """
        # 转置为 [B, 3, N]
        x = x.transpose(1, 2)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        
        # 全局最大池化
        x = torch.max(x, dim=2, keepdim=True)[0]
        
        return x.squeeze(-1)


class DCP(nn.Module):
    """
    Deep Closest Point (DCP)
    
    架构:
    1. PointNet 编码器 - 提取点云特征
    2. Transformer 编码器 - 捕获点对关系
    3. 回归头 - 预测变换参数 Δ(R,t)
    """
    
    def __init__(self, config: Dict):
        super(DCP, self).__init__()
        self.config = config
        
        # PointNet 编码器
        pointnet_config = config.get('POINTNET', {})
        embedding_dim = pointnet_config.get('EMBEDDING_DIM', 1024)
        
        self.encoder = PointNetEncoder(embedding_dim)
        
        # Transformer 编码器
        transformer_config = config.get('TRANSFORMER', {})
        hidden_dim = transformer_config.get('HIDDEN_DIM', 512)
        
        # 投影层：从 encoder 维度到 transformer 维度
        self.project_to_transformer = nn.Linear(embedding_dim * 2, hidden_dim)
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
        
        # 回归头
        regressor_config = config.get('REGRESSOR', {})
        layers = regressor_config.get('LAYERS', [512, 256, 128, 6])
        hidden_dim = transformer_config.get('HIDDEN_DIM', 512)
        
        # 确保第一层的输入维度与 transformer 输出一致
        regression_layers = []
        prev_dim = hidden_dim  # 使用 transformer 的输出维度
        for i in range(len(layers) - 1):
            regression_layers.extend([
                nn.Linear(prev_dim, layers[i+1] if i > 0 else layers[0]),
                nn.ReLU(inplace=True)
            ])
            prev_dim = layers[i+1] if i > 0 else layers[0]
        
        # 移除最后一个 ReLU，因为输出是连续的变换参数
        self.regressor = nn.Sequential(*regression_layers[:-1])
        
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
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            src: 源点云 [B, N, 3]
            tgt: 目标点云 [B, M, 3]
            
        Returns:
            R_pred: 预测的旋转矩阵 [B, 3, 3]
            t_pred: 预测的平移向量 [B, 3]
        """
        batch_size = src.shape[0]
        
        # 编码
        src_feat = self.encoder(src)  # [B, D]
        tgt_feat = self.encoder(tgt)  # [B, D]
        
        # 拼接特征
        combined = torch.cat([src_feat, tgt_feat], dim=1)  # [B, 2D]
        
        # 投影到 transformer 维度
        combined = self.project_to_transformer(combined)
        
        # 添加序列维度以适应 Transformer
        combined = combined.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Transformer 编码
        combined = self.transformer(combined)
        
        # 回归变换参数
        params = self.regressor(combined.squeeze(1))  # [B, 6]
        
        # 分离旋转和平移
        rotation_params = params[:, :3]  # 轴角表示
        translation_params = params[:, 3:]  # [B, 3]
        
        # 将轴角转换为旋转矩阵
        R_pred = self._axis_angle_to_matrix(rotation_params)
        t_pred = translation_params
        
        return R_pred, t_pred
    
    def predict_transform(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        预测完整的 4x4 变换矩阵
        
        Args:
            src: 源点云 [B, N, 3]
            tgt: 目标点云 [B, M, 3]
            
        Returns:
            transform: 4x4 变换矩阵 [B, 4, 4]
        """
        R_pred, t_pred = self.forward(src, tgt)
        
        batch_size = R_pred.shape[0]
        device = R_pred.device
        
        # 构建 4x4 变换矩阵
        transform = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        transform[:, :3, :3] = R_pred
        transform[:, :3, 3] = t_pred
        
        return transform
    
    def _axis_angle_to_matrix(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """
        轴角表示转换为旋转矩阵
        
        Args:
            axis_angle: [B, 3] 旋转向量（方向为轴，模长为角度）
        Returns:
            rotation_matrix: [B, 3, 3]
        """
        batch_size = axis_angle.shape[0]
        
        # 计算旋转角度和轴
        angle = torch.norm(axis_angle, dim=1, keepdim=True) + 1e-8
        axis = axis_angle / angle
        
        # Rodrigues 旋转公式
        cos_a = torch.cos(angle).squeeze(1)
        sin_a = torch.sin(angle).squeeze(1)
        
        # 构建反对称矩阵
        K = torch.zeros((batch_size, 3, 3), device=axis_angle.device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        # Rodrigues 公式：R = I + sin(θ)*K + (1-cos(θ))*K²
        I = torch.eye(3, device=axis_angle.device).unsqueeze(0).repeat(batch_size, 1, 1)
        R = I + \
            sin_a.view(-1, 1, 1) * K + \
            (1 - cos_a).view(-1, 1, 1) * torch.bmm(K, K)
        
        return R


def test_dcp():
    """测试 DCP 模型"""
    print("测试 DCP 模型...")
    
    config = {
        'MODEL': {
            'POINTNET': {'EMBEDDING_DIM': 512},
            'TRANSFORMER': {
                'NUM_LAYERS': 2,
                'NUM_HEADS': 4,
                'HIDDEN_DIM': 256
            },
            'REGRESSOR': {'LAYERS': [512, 256, 128, 6]}
        }
    }
    
    model = DCP(config['MODEL'])
    
    # 创建测试数据
    batch_size = 2
    src_points = torch.randn(batch_size, 1024, 3)
    tgt_points = torch.randn(batch_size, 1024, 3)
    
    # 前向传播
    with torch.no_grad():
        R_pred, t_pred = model(src_points, tgt_points)
        transform_matrix = model.predict_transform(src_points, tgt_points)
    
    print(f"✓ 输入形状：src={src_points.shape}, tgt={tgt_points.shape}")
    print(f"✓ 输出形状：R={R_pred.shape}, t={t_pred.shape}")
    print(f"✓ 变换矩阵形状：{transform_matrix.shape}")
    
    print("\nDCP 模型测试通过！")
    return model


if __name__ == '__main__':
    test_dcp()
