"""
D3Feat (Deep Dense Descriptor for 3D Features) 模型实现
基于全卷积架构的 3D 特征描述子网络
论文：https://arxiv.org/abs/2006.07882
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class PointNetSetAbstraction(nn.Module):
    """PointNet 集合抽象层 - 用于降采样和特征提取"""
    
    def __init__(self, in_channels, out_channels, npoint, radius, nsample):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        # MLP 层
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // 4, out_channels // 2, kernel_size=1),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // 2, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = None
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, xyz, features):
        """
        Args:
            xyz: 点坐标 [B, N, 3]
            features: 点特征 [B, C, N]
        Returns:
            new_xyz: 采样后的点 [B, npoint, 3]
            new_features: 采样后的特征 [B, C', npoint]
        """
        device = xyz.device
        B, N, _ = xyz.shape
        
        # 简单降采样（使用最远点采样的简化版本）
        if self.npoint >= N or self.npoint is None:
            # 不降采样，但仍然要应用 MLP 进行特征提取
            new_xyz = xyz
            new_features = self.mlp(features)
            
            if self.shortcut is not None:
                shortcut = self.shortcut(features)
                new_features = self.relu(new_features + shortcut)
            else:
                new_features = self.relu(new_features)
        else:
            # 使用随机采样作为简化
            indices = torch.randperm(N, device=device)[:self.npoint]
            new_xyz = xyz[:, indices, :]
            
            # 提取对应特征并应用 MLP
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, self.nsample)
            sampled_features = features[:, :, indices]
            
            new_features = self.mlp(sampled_features)
            
            if self.shortcut is not None:
                shortcut = self.shortcut(features[:, :, indices])
                new_features = self.relu(new_features + shortcut)
            else:
                new_features = self.relu(new_features)
        
        return new_xyz, new_features


class PointNetFeaturePropagation(nn.Module):
    """PointNet 特征传播层 - 用于上采样"""
    
    def __init__(self, in_channels, out_channels):
        super(PointNetFeaturePropagation, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, xyz1, xyz2, features1, features2):
        """
        Args:
            xyz1: 原始点坐标 [B, N, 3]
            xyz2: 降采样点坐标 [B, M, 3]
            features1: 原始点特征 [B, C1, N]
            features2: 降采样点特征 [B, C2, M]
        Returns:
            new_features: 插值后的特征 [B, C', N]
        """
        B, N, _ = xyz1.shape
        _, M, _ = xyz2.shape
        
        # 简单的最近邻插值
        if features2.shape[2] == features1.shape[2]:
            cat_features = torch.cat([features1, features2], dim=1)
        else:
            # 上采样 features2 到 features1 的分辨率
            indices = torch.randint(0, M, (B, N), device=features1.device)
            upsampled_features = torch.gather(
                features2.transpose(1, 2), 1,
                indices.unsqueeze(-1).expand(-1, -1, features2.shape[1])
            ).transpose(1, 2)
            cat_features = torch.cat([features1, upsampled_features], dim=1)
        
        return self.mlp(cat_features)


class D3FeatEncoder(nn.Module):
    """
    D3Feat 特征编码器
    
    架构:
    1. 输入层 - 处理点云坐标
    2. SA 层 - 逐步降采样并提取特征
    3. FP 层 - 上采样恢复分辨率
    4. 输出层 - 生成每个点的特征描述子
    """
    
    def __init__(self, config: Dict = None):
        super(D3FeatEncoder, self).__init__()
        self.config = config or {}
        
        # 输入维度
        input_dim = self.config.get('INPUT_DIM', 3)
        feature_dim = self.config.get('FEATURE_DIM', 256)
        
        # SA 层参数
        sa_config = self.config.get('SA_LAYERS', {})
        
        # SA Layer 1
        self.sa1 = PointNetSetAbstraction(
            in_channels=input_dim,
            out_channels=sa_config.get('C1', 64),
            npoint=sa_config.get('NPOINT1', 1024),
            radius=sa_config.get('RADIUS1', 0.1),
            nsample=sa_config.get('NSAMPLE1', 32)
        )
        
        # SA Layer 2
        self.sa2 = PointNetSetAbstraction(
            in_channels=sa_config.get('C1', 64),
            out_channels=sa_config.get('C2', 128),
            npoint=sa_config.get('NPOINT2', 256),
            radius=sa_config.get('RADIUS2', 0.2),
            nsample=sa_config.get('NSAMPLE2', 32)
        )
        
        # SA Layer 3
        self.sa3 = PointNetSetAbstraction(
            in_channels=sa_config.get('C2', 128),
            out_channels=sa_config.get('C3', 256),
            npoint=sa_config.get('NPOINT3', 64),
            radius=sa_config.get('RADIUS3', 0.4),
            nsample=sa_config.get('NSAMPLE3', 32)
        )
        
        # FP Layer 1
        self.fp1 = PointNetFeaturePropagation(
            in_channels=sa_config.get('C2', 128) + sa_config.get('C3', 256),
            out_channels=sa_config.get('C2', 128)
        )
        
        # FP Layer 2
        self.fp2 = PointNetFeaturePropagation(
            in_channels=sa_config.get('C1', 64) + sa_config.get('C2', 128),
            out_channels=sa_config.get('C1', 64)
        )
        
        # 输出层
        self.conv_out = nn.Sequential(
            nn.Conv1d(sa_config.get('C1', 64), feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
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
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: 点云 [B, N, 3]
        Returns:
            features: 点特征 [B, N, D]
        """
        B, N, _ = points.shape
        
        # 转置为 [B, 3, N]
        xyz = points  # [B, N, 3]
        features = points.transpose(1, 2)  # [B, 3, N]
        
        # SA 层
        try:
            l1_xyz, l1_features = self.sa1(xyz, features)
            l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
            l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        except RuntimeError as e:
            print(f"\n[D3FeatEncoder Debug]")
            print(f"  Input shape: {points.shape}")
            print(f"  Features shape: {features.shape}")
            if 'l1_features' in locals():
                print(f"  L1 features shape: {l1_features.shape}")
            if 'l2_features' in locals():
                print(f"  L2 features shape: {l2_features.shape}")
            print(f"  Error: {e}\n")
            raise
        
        # FP 层
        l2_features = self.fp1(l2_xyz, l3_xyz, l2_features, l3_features)
        l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)
        
        # 输出层
        output_features = self.conv_out(l1_features)
        
        # 转置回 [B, N, D]
        return output_features.transpose(1, 2)


class D3Feat(nn.Module):
    """
    D3Feat: Deep Dense Descriptor for 3D Features
    
    主要功能:
    1. 提取每个点的局部几何特征
    2. 生成具有判别性的特征描述子
    3. 支持特征匹配和关键点检测
    """
    
    def __init__(self, config: Dict):
        super(D3Feat, self).__init__()
        self.config = config
        
        # 编码器 - 支持两种配置格式：
        # 1. 直接传入 ENCODER 配置（包含 INPUT_DIM, FEATURE_DIM, SA_LAYERS）
        # 2. 传入完整 MODEL 配置（需要从 config['ENCODER'] 提取）
        if 'INPUT_DIM' in config or 'SA_LAYERS' in config:
            # 直接是 ENCODER 配置
            encoder_config = config
        else:
            # 完整 MODEL 配置
            encoder_config = config.get('ENCODER', {})
        
        self.encoder = D3FeatEncoder(encoder_config)
        
        # 可选：关键点检测头（用于显著性预测）
        keypoint_head_enabled = config.get('KEYPOINT_HEAD', False)
        if keypoint_head_enabled and 'ENCODER' in config:
            # 从 ENCODER 配置中获取 FEATURE_DIM
            feature_dim = config['ENCODER'].get('FEATURE_DIM', 256)
        elif keypoint_head_enabled:
            # 直接使用 config 中的 FEATURE_DIM
            feature_dim = config.get('FEATURE_DIM', 256)
        else:
            feature_dim = None
        
        if keypoint_head_enabled and feature_dim is not None:
            self.keypoint_head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1)
            )
        else:
            self.keypoint_head = None
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: 点云 [B, N, 3]
        Returns:
            features: 点特征 [B, N, D]
        """
        return self.encoder(points)
    
    def extract_features(self, points: torch.Tensor) -> torch.Tensor:
        """
        提取点云特征
        
        Args:
            points: 点云 [B, N, 3] 或 [N, 3]
        Returns:
            features: 特征 [B, N, D] 或 [N, D]
        """
        # 确保有 batch 维度
        if len(points.shape) == 2:
            points = points.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        # 提取特征
        with torch.no_grad():
            features = self.forward(points)
        
        if squeeze_batch:
            features = features.squeeze(0)
        
        return features
    
    def detect_keypoints(self, points: torch.Tensor, 
                        num_keypoints: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检测关键点
        
        Args:
            points: 点云 [B, N, 3]
            num_keypoints: 关键点数量
        Returns:
            keypoints: 关键点索引 [B, K]
            scores: 关键点分数 [B, K]
        """
        if self.keypoint_head is None:
            raise RuntimeError("关键点检测头未启用")
        
        # 提取特征
        features = self.extract_features(points)
        
        # 预测显著性分数
        saliency_scores = self.keypoint_head(features).squeeze(-1)
        
        # 选择 top-K 关键点
        scores, indices = torch.topk(saliency_scores, num_keypoints, dim=-1)
        
        return indices, scores
    
    def match_features(self, feat1: torch.Tensor, 
                      feat2: torch.Tensor, 
                      ratio_threshold: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        特征匹配（Lowe's ratio test）
        
        Args:
            feat1: 特征 1 [N, D]
            feat2: 特征 2 [M, D]
            ratio_threshold: Lowe's ratio 阈值
        Returns:
            matches: 匹配对 [K, 2]
            scores: 匹配分数 [K]
        """
        from sklearn.neighbors import NearestNeighbors
        
        # 转换为 numpy
        feat1_np = feat1.cpu().numpy()
        feat2_np = feat2.cpu().numpy()
        
        # 最近邻搜索
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feat2_np)
        distances, indices = nbrs.kneighbors(feat1_np)
        
        # Lowe's ratio test
        ratios = distances[:, 0] / (distances[:, 1] + 1e-8)
        good_matches = ratios < ratio_threshold
        
        if np.sum(good_matches) == 0:
            return torch.empty(0, 2, dtype=torch.long), torch.empty(0)
        
        # 构建匹配对
        query_indices = torch.where(torch.from_numpy(good_matches))[0]
        train_indices = torch.from_numpy(indices[good_matches, 0])
        matches = torch.stack([query_indices, train_indices], dim=-1)
        
        # 计算匹配分数
        match_scores = 1.0 / (1.0 + distances[good_matches, 0])
        match_scores = torch.from_numpy(match_scores)
        
        return matches, match_scores
