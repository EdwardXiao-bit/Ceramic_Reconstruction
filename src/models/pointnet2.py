"""
PointNet++ 模型实现
基于分层特征提取的点云学习网络
论文：https://arxiv.org/abs/1706.02413
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class PointNetSetAbstraction(nn.Module):
    """PointNet++ 集合抽象层 (SA Layer)"""
    
    def __init__(self, in_channel, mlp, npoint, radius, nsample, use_xyz=True):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        # MLP 层
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel + 3 if use_xyz else in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, kernel_size=1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.use_xyz = use_xyz
    
    def forward(self, xyz, features):
        """
        Args:
            xyz: 点坐标 [B, N, 3]
            features: 点特征 [B, C, N]
        Returns:
            new_xyz: 采样后的点 [B, npoint, 3]
            new_features: 聚合后的特征 [B, C', npoint]
        """
        B, N, _ = xyz.shape
        
        # 最远点采样（简化版本，使用随机采样）
        if self.npoint is None or self.npoint >= N:
            new_xyz = xyz
            indices = torch.arange(N, device=xyz.device).unsqueeze(0).expand(B, -1)
        else:
            # 随机采样作为简化
            indices = torch.randperm(N, device=xyz.device)[:self.npoint].unsqueeze(0).expand(B, -1)
            new_xyz = torch.gather(xyz, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
        
        # 分组和特征提取
        if self.nsample is None or self.nsample >= N:
            # 全局池化情况（SA4 层），直接应用 MLP
            # features: [B, C, N] -> 添加 nsample 维度 -> [B, C, N, 1]
            new_features = features.unsqueeze(-1)  # [B, C, N, 1]
        else:
            new_features = self._group_features(xyz, new_xyz, features, indices)
        
        # MLP 处理
        for i, conv in enumerate(self.mlp_convs):
            new_features = conv(new_features)
            new_features = self.mlp_bns[i](new_features)
            new_features = torch.relu(new_features)
        
        # 最大池化
        if new_features.dim() == 4:
            # [B, C, N, nsample] -> [B, C, N]
            new_features = torch.max(new_features, dim=-1)[0]
        elif new_features.dim() == 3:
            # [B, C, N, 1] -> [B, C, N]
            new_features = torch.max(new_features, dim=-1, keepdim=True)[0]
            new_features = new_features.squeeze(-1)
        
        return new_xyz, new_features
    
    def _group_features(self, xyz, new_xyz, features, indices):
        """简单的特征分组"""
        B, N, _ = xyz.shape
        _, M, _ = new_xyz.shape
        
        # 计算距离
        dists = self._square_distance(new_xyz, xyz)  # [B, M, N]
        
        # 获取最近的 nsample 个点
        _, idx = torch.topk(dists, k=min(self.nsample, N), dim=-1, largest=False)
        
        # 收集特征
        grouped_features_list = []
        for b in range(B):
            # 索引对应的特征
            batch_features = features[b:b+1, :, idx[b]]  # [1, C, M, nsample]
            
            if self.use_xyz:
                # 添加相对位置编码
                grouped_xyz = xyz[b:b+1, idx[b], :]  # [1, M, nsample, 3]
                # 扩展 new_xyz 以匹配维度
                new_xyz_expanded = new_xyz[b:b+1, :, :].unsqueeze(2)  # [1, M, 1, 3]
                relative_xyz = grouped_xyz - new_xyz_expanded  # [1, M, nsample, 3]
                # 转置并拼接
                relative_xyz = relative_xyz.permute(0, 3, 1, 2)  # [1, 3, M, nsample]
                grouped_features_list.append(torch.cat([batch_features, relative_xyz], dim=1))
            else:
                grouped_features_list.append(batch_features)
        
        return torch.cat(grouped_features_list, dim=0)  # [B, C', M, nsample]

    def _square_distance(self, src, dst):
        """计算点对之间的平方距离"""
        B, N, _ = src.shape
        _, M, _ = dst.shape
        
        diff = src.unsqueeze(2) - dst.unsqueeze(1)  # [B, N, M, 3]
        return torch.sum(diff ** 2, dim=-1)  # [B, N, M]


class PointNetFeaturePropagation(nn.Module):
    """PointNet++ 特征传播层 (FP Layer)"""
    
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, kernel_size=1, bias=False))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz1, xyz2, features1, features2):
        """
        Args:
            xyz1: 原始点坐标 [B, N, 3]
            xyz2: 降采样点坐标 [B, M, 3]
            features1: 原始点特征 [B, C1, N]
            features2: 降采样点特征 [B, C2, M] 或 [B, C2, M, 1]
        Returns:
            new_features: 插值后的特征 [B, C', N]
        """
        B, N, _ = xyz1.shape
        _, M, _ = xyz2.shape
        
        # 确保 features2 是 3D [B, C, M]
        if features2.dim() == 4:
            features2 = features2.squeeze(-1)  # [B, C2, M]
        
        # 最近邻插值
        dists = self._square_distance(xyz1, xyz2)
        _, idx = torch.min(dists, dim=-1)  # [B, N]
        
        # 收集特征 - idx[b, i] 表示 xyz1[b, i] 的最近邻是 xyz2[b, idx[b, i]]
        # 所以 features2[b, :, idx[b, i]] 就是对应的特征
        B, C2, M = features2.shape
        
        # 使用 gather 操作进行特征插值
        # idx: [B, N] -> 需要扩展为 [B, C2, N]
        idx_expanded = idx.unsqueeze(1).expand(-1, C2, -1)  # [B, C2, N]
        interpolated_features_list = []
        
        for b in range(B):
            # 使用 gather 在最后一个维度上收集
            feat = torch.gather(features2[b:b+1], dim=-1, index=idx_expanded[b:b+1])  # [C2, N]
            interpolated_features_list.append(feat.squeeze(0))  # [C2, N]
        
        interpolated_features = torch.stack(interpolated_features_list, dim=0) if len(interpolated_features_list) > 1 else interpolated_features_list[0].unsqueeze(0)
        
        # 拼接特征
        if features1 is not None:
            # 确保 features1 也是 3D
            if features1.dim() == 4:
                features1 = features1.squeeze(-1)
            cat_features = torch.cat([features1, interpolated_features], dim=1)
        else:
            cat_features = interpolated_features
        
        # MLP 处理
        for i, conv in enumerate(self.mlp_convs):
            cat_features = conv(cat_features)
            cat_features = self.mlp_bns[i](cat_features)
            cat_features = torch.relu(cat_features)
        
        return cat_features
    
    def _square_distance(self, src, dst):
        """计算点对之间的平方距离"""
        B, N, _ = src.shape
        _, M, _ = dst.shape
        
        diff = src.unsqueeze(2) - dst.unsqueeze(1)  # [B, N, M, 3]
        return torch.sum(diff ** 2, dim=-1)  # [B, N, M]


class PointNet2Encoder(nn.Module):
    """
    PointNet++ 特征编码器
    
    架构:
    1. SA Layer 1: 降采样到 1024 点
    2. SA Layer 2: 降采样到 256 点
    3. SA Layer 3: 降采样到 64 点
    4. SA Layer 4: 全局特征
    5. FP Layer: 上采样恢复分辨率
    """
    
    def __init__(self, config: Dict = None):
        super(PointNet2Encoder, self).__init__()
        self.config = config or {}
        
        # 输入维度
        input_dim = self.config.get('INPUT_DIM', 3)
        output_dim = self.config.get('OUTPUT_DIM', 256)
        
        # SA Layers
        self.sa1 = PointNetSetAbstraction(
            in_channel=input_dim,
            mlp=[64, 64, 128],
            npoint=1024,
            radius=0.1,
            nsample=32,
            use_xyz=True
        )
        
        self.sa2 = PointNetSetAbstraction(
            in_channel=128,
            mlp=[128, 128, 256],
            npoint=256,
            radius=0.2,
            nsample=32,
            use_xyz=True
        )
        
        self.sa3 = PointNetSetAbstraction(
            in_channel=256,
            mlp=[256, 256, 512],
            npoint=64,
            radius=0.4,
            nsample=32,
            use_xyz=True
        )
        
        # Global SA Layer
        self.sa4 = PointNetSetAbstraction(
            in_channel=512,
            mlp=[512, 512, 1024],
            npoint=None,
            radius=None,
            nsample=None,
            use_xyz=False
        )
        
        # FP Layers
        self.fp3 = PointNetFeaturePropagation(
            in_channel=1536,  # 512 + 1024
            mlp=[512, 512]
        )
        
        self.fp2 = PointNetFeaturePropagation(
            in_channel=768,  # 256 + 512
            mlp=[256, 256]
        )
        
        self.fp1 = PointNetFeaturePropagation(
            in_channel=384,  # 128 + 256
            mlp=[256, output_dim]
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
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
        
        # 准备输入
        xyz = points  # [B, N, 3]
        features = points.transpose(1, 2)  # [B, 3, N]
        
        # SA Layers
        l1_xyz, l1_features = self.sa1(xyz, features)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        l4_xyz, l4_features = self.sa4(l3_xyz, l3_features)
        
        # l4_features 从 SA4 出来是 [B, 1024, 64] (因为 npoint=64)
        # 需要保持这个形状传递给 FP 层
        
        # FP Layers - 逐步上采样
        l3_features = self.fp3(l3_xyz, l4_xyz, l3_features, l4_features)
        l2_features = self.fp2(l2_xyz, l3_xyz, l2_features, l3_features)
        l1_features = self.fp1(l1_xyz, l2_xyz, l1_features, l2_features)
        
        # 转置回 [B, N, D]
        return l1_features.transpose(1, 2)
    
    def encode(self, points: torch.Tensor) -> torch.Tensor:
        """
        提取点云特征（兼容 complementarity_checker 的接口）
        
        Args:
            points: 点云 [N, 3] 或 [B, N, 3]
        Returns:
            features: 特征 [N, D] 或 [B, N, D]
        """
        # 确保有 batch 维度
        squeeze_batch = False
        if len(points.shape) == 2:
            points = points.unsqueeze(0)
            squeeze_batch = True
        
        # 前向传播
        with torch.no_grad():
            features = self.forward(points)
        
        if squeeze_batch:
            features = features.squeeze(0)
        
        return features


class PointNet2SSG(nn.Module):
    """
    PointNet++ SSG (Single-Scale Grouping)
    简化的单尺度版本，适用于局部 patch 特征提取
    """
    
    def __init__(self, config: Dict = None):
        super(PointNet2SSG, self).__init__()
        self.config = config or {}
        
        input_dim = self.config.get('INPUT_DIM', 3)
        output_dim = self.config.get('OUTPUT_DIM', 256)
        
        # 简化的 SA 层
        self.sa1 = PointNetSetAbstraction(
            in_channel=input_dim,
            mlp=[64, 64, 128],
            npoint=None,  # 不降采样
            radius=0.05,
            nsample=32,
            use_xyz=True
        )
        
        self.sa2 = PointNetSetAbstraction(
            in_channel=128,
            mlp=[128, 128, output_dim],
            npoint=None,
            radius=0.1,
            nsample=32,
            use_xyz=True
        )
        
        # 全局池化层
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: 点云 [B, N, 3]
        Returns:
            features: 全局特征 [B, D]
        """
        B, N, _ = points.shape
        
        xyz = points
        features = points.transpose(1, 2)
        
        # SA 层
        _, f1 = self.sa1(xyz, features)
        _, f2 = self.sa2(xyz, f1)
        
        # 全局池化
        global_feat = self.global_pool(f2).squeeze(-1)
        
        return global_feat
    
    def encode(self, points: torch.Tensor) -> torch.Tensor:
        """
        提取全局特征向量
        
        Args:
            points: 点云 [N, 3] 或 [B, N, 3]
        Returns:
            features: 全局特征 [D] 或 [B, D]
        """
        squeeze_batch = False
        if len(points.shape) == 2:
            points = points.unsqueeze(0)
            squeeze_batch = True
        
        with torch.no_grad():
            features = self.forward(points)
        
        if squeeze_batch:
            features = features.squeeze(0)
        
        return features
