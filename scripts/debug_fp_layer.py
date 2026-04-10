"""
继续调试 FP 层
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.pointnet2 import PointNet2Encoder

config = {'INPUT_DIM': 3, 'OUTPUT_DIM': 256}
device = torch.device('cpu')
model = PointNet2Encoder(config).to(device)
model.eval()

points = torch.randn(1, 1024, 3)

with torch.no_grad():
    xyz = points
    features = points.transpose(1, 2)
    
    l1_xyz, l1_features = model.sa1(xyz, features)
    l2_xyz, l2_features = model.sa2(l1_xyz, l1_features)
    l3_xyz, l3_features = model.sa3(l2_xyz, l2_features)
    l4_xyz, l4_features = model.sa4(l3_xyz, l3_features)
    
    print(f"l3_xyz: {l3_xyz.shape}, l4_xyz: {l4_xyz.shape}")
    print(f"l3_features: {l3_features.shape}, l4_features: {l4_features.shape}")
    
    # 手动计算距离
    B, N, _ = l3_xyz.shape
    _, M, _ = l4_xyz.shape
    
    # dists[i,j] = ||l3_xyz[i] - l4_xyz[j]||^2
    diff = l3_xyz.unsqueeze(2) - l4_xyz.unsqueeze(1)  # [B, N, M, 3]
    dists = torch.sum(diff ** 2, dim=-1)  # [B, N, M]
    
    print(f"\n距离矩阵形状：{dists.shape}")
    
    # 找最近邻
    _, idx = torch.min(dists, dim=-1)  # [B, N]
    print(f"索引形状：{idx.shape}")
    print(f"索引范围：[{idx.min()}, {idx.max()}] (应该是 0 到 {M-1})")
    
    # 收集特征
    for b in range(B):
        feat = l4_features[b:b+1, :, idx[b]]
        print(f"\nb={b}: features2[:, idx[{b}]] 形状：{feat.shape}")
    
    # 问题可能在于 l4_features 的维度
    print(f"\nl4_features 形状：{l4_features.shape}")
    print(f"尝试直接索引...")
    try:
        result = l4_features[0, :, idx[0]]
        print(f"成功！结果形状：{result.shape}")
    except Exception as e:
        print(f"失败：{e}")
