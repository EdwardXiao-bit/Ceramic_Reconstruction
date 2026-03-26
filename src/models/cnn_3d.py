"""
3D CNN 模型实现
用于点云互补性预测的三维卷积神经网络
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import open3d as o3d


class Voxelizer:
    """点云体素化工具"""
    
    def __init__(self, resolution=0.01, grid_size=32):
        """
        Args:
            resolution: 体素分辨率（米）
            grid_size: 体素网格大小
        """
        self.resolution = resolution
        self.grid_size = grid_size
    
    def voxelize(self, points: np.ndarray) -> np.ndarray:
        """
        将点云转换为体素网格
        
        Args:
            points: 点云 [N, 3]
        
        Returns:
            voxel_grid: 体素网格 [grid_size, grid_size, grid_size]
        """
        # 创建体素网格
        voxel_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)
        
        # 归一化点到 [0, grid_size]
        if len(points) == 0:
            return voxel_grid
        
        min_point = points.min(axis=0)
        max_point = points.max(axis=0)
        
        # 避免除零
        range_point = max_point - min_point
        range_point[range_point < 1e-8] = 1e-8
        
        # 归一化并缩放到 grid_size
        normalized_points = (points - min_point) / range_point * (self.grid_size - 1)
        
        # 转换为整数索引
        indices = np.floor(normalized_points).astype(np.int32)
        indices = np.clip(indices, 0, self.grid_size - 1)
        
        # 填充体素（占用网格）
        for idx in indices:
            voxel_grid[idx[0], idx[1], idx[2]] = 1.0
        
        return voxel_grid


class ConvBlock3D(nn.Module):
    """3D 卷积块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=True):
        super(ConvBlock3D, self).__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, 
                             kernel_size=kernel_size, padding=padding, bias=not use_batchnorm)
        self.bn = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ComplementarityPredictor(nn.Module):
    """
    3D CNN 互补性预测器
    
    架构:
    1. 双分支编码器 - 分别处理两个 patch
    2. 特征融合 - 拼接两个分支的特征
    3. 回归头 - 预测互补性得分
    """
    
    def __init__(self, input_channels=1, base_channels=32, grid_size=32):
        super(ComplementarityPredictor, self).__init__()
        
        # 分支 1 编码器
        self.branch1 = nn.Sequential(
            ConvBlock3D(input_channels, base_channels),      # [32, 16, 16, 16]
            ConvBlock3D(base_channels, base_channels * 2),   # [64, 8, 8, 8]
            ConvBlock3D(base_channels * 2, base_channels * 4) # [128, 4, 4, 4]
        )
        
        # 分支 2 编码器（权重共享）
        self.branch2 = nn.Sequential(
            ConvBlock3D(input_channels, base_channels),
            ConvBlock3D(base_channels, base_channels * 2),
            ConvBlock3D(base_channels * 2, base_channels * 4)
        )
        
        # 权重共享
        self.branch2[0].conv.weight = self.branch1[0].conv.weight
        self.branch2[0].bn = self.branch1[0].bn
        self.branch2[1].conv.weight = self.branch1[1].conv.weight
        self.branch2[1].bn = self.branch1[1].bn
        self.branch2[2].conv.weight = self.branch1[2].conv.weight
        self.branch2[2].bn = self.branch1[2].bn
        
        # 计算展平后的特征维度
        # 经过 3 次池化：grid_size -> grid_size/2 -> grid_size/4 -> grid_size/8
        final_size = grid_size // 8
        fused_features = base_channels * 4 * (final_size ** 3) * 2  # 两个分支
        
        # 特征融合后的全连接层
        self.regressor = nn.Sequential(
            nn.Linear(fused_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出 [0, 1] 范围的互补性得分
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, voxel1: torch.Tensor, voxel2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel1: patch1 的体素 [B, 1, D, H, W]
            voxel2: patch2 的体素 [B, 1, D, H, W]
        
        Returns:
            complementarity_score: 互补性得分 [B, 1]
        """
        # 编码
        feat1 = self.branch1(voxel1)  # [B, C, 4, 4, 4]
        feat2 = self.branch2(voxel2)  # [B, C, 4, 4, 4]
        
        # 展平
        feat1_flat = feat1.view(feat1.size(0), -1)  # [B, C*4*4*4]
        feat2_flat = feat2.view(feat2.size(0), -1)
        
        # 拼接
        combined = torch.cat([feat1_flat, feat2_flat], dim=1)
        
        # 预测
        score = self.regressor(combined)
        
        return score


class PointNet3DCNN:
    """
    基于 3D CNN 的点云互补性检测器
    
    使用流程:
    1. 体素化：将点云 patch 转换为 3D 网格
    2. 特征提取：使用 3D CNN 提取特征
    3. 互补性预测：输出相似度得分
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 体素化参数
        self.voxel_resolution = self.config.get('VOXEL_RESOLUTION', 0.01)
        self.voxel_grid_size = self.config.get('VOXEL_GRID_SIZE', 32)
        
        # 创建体素化器
        self.voxelizer = Voxelizer(
            resolution=self.voxel_resolution,
            grid_size=self.voxel_grid_size
        )
        
        # 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ComplementarityPredictor(
            input_channels=1,
            base_channels=self.config.get('BASE_CHANNELS', 32),
            grid_size=self.voxel_grid_size
        ).to(device)
        
        self.device = device
        self.model.eval()
    
    def voxelize_patch(self, points: np.ndarray) -> torch.Tensor:
        """
        将点云 patch 体素化
        
        Args:
            points: 点云 [N, 3]
        
        Returns:
            voxel_tensor: 体素张量 [1, 1, D, H, W]
        """
        # 体素化
        voxel_grid = self.voxelizer.voxelize(points)
        
        # 转换为 tensor
        voxel_tensor = torch.from_numpy(voxel_grid).unsqueeze(0).unsqueeze(0).float()
        
        return voxel_tensor.to(self.device)
    
    def predict_complementarity(self, patch1: np.ndarray, patch2: np.ndarray) -> float:
        """
        预测两个 patch 的互补性
        
        Args:
            patch1: patch1 点云 [N1, 3]
            patch2: patch2 点云 [N2, 3]
        
        Returns:
            score: 互补性得分 [0, 1]
        """
        # 体素化
        voxel1 = self.voxelize_patch(patch1)
        voxel2 = self.voxelize_patch(patch2)
        
        # 预测
        with torch.no_grad():
            score = self.model(voxel1, voxel2)
        
        return score.item()
    
    def load_weights(self, weight_path: str):
        """加载预训练权重"""
        checkpoint = torch.load(weight_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"[3D CNN] 权重加载成功：{weight_path}")


# 简化的轻量版本，适用于实时应用
class Light3DCNN:
    """轻量级 3D CNN"""
    
    def __init__(self):
        self.voxelizer = Voxelizer(resolution=0.02, grid_size=16)
        
        # 简化网络 - 接受双通道输入（两个 patch）
        self.model = nn.Sequential(
            # Branch 1
            nn.Conv3d(2, 16, 3, padding=1),  # 2 channels for stacked voxels
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),  # 8x8x8
            
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),  # 4x4x4
            
            nn.Flatten(),
            nn.Linear(32 * 4 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 权重共享需要在 forward 中处理
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict_complementarity(self, patch1: np.ndarray, patch2: np.ndarray) -> float:
        """快速预测互补性"""
        # 体素化
        voxel1 = self._voxelize(patch1)  # [1, D, H, W]
        voxel2 = self._voxelize(patch2)  # [1, D, H, W]
        
        with torch.no_grad():
            # 拼接两个体素作为双通道输入 [1, 2, D, H, W]
            combined = torch.stack([voxel1, voxel2], dim=1)
            score = self.model(combined)
        
        return score.item()
    
    def _voxelize(self, points: np.ndarray) -> torch.Tensor:
        """体素化并转换为 tensor"""
        voxel_grid = self.voxelizer.voxelize(points)
        voxel_tensor = torch.from_numpy(voxel_grid).unsqueeze(0).float()  # [1, D, H, W]
        return voxel_tensor.to(self.device)


if __name__ == '__main__':
    # 测试
    print("测试 3D CNN...")
    
    # 创建随机点云
    patch1 = np.random.randn(50, 3).astype(np.float32)
    patch2 = np.random.randn(50, 3).astype(np.float32)
    
    # 测试完整版
    cnn = PointNet3DCNN({'VOXEL_RESOLUTION': 0.01, 'VOXEL_GRID_SIZE': 32})
    score = cnn.predict_complementarity(patch1, patch2)
    
    print(f"完整版 3D CNN 得分：{score:.4f}")
    
    # 测试轻量版
    light_cnn = Light3DCNN()
    score_light = light_cnn.predict_complementarity(patch1, patch2)
    
    print(f"轻量版 3D CNN 得分：{score_light:.4f}")
    print("测试完成！")
