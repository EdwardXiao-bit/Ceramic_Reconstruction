"""
测试 PointNet++ 模型
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.models.pointnet2 import PointNet2Encoder, PointNet2SSG

def test_pointnet2_encoder():
    """测试完整版 PointNet2 编码器"""
    
    config = {
        'INPUT_DIM': 3,
        'OUTPUT_DIM': 256
    }
    
    print("=" * 60)
    print("PointNet2 Encoder 测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] 使用设备：{device}")
    
    model = PointNet2Encoder(config).to(device)
    print(f"[✓] 模型创建成功")
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    总参数量：{total_params:,}")
    print(f"    可训练参数量：{trainable_params:,}")
    
    # 测试特征提取
    print("\n[2] 测试逐点特征提取...")
    batch_size = 2
    num_points = 2048
    
    points = torch.randn(batch_size, num_points, 3).to(device)
    print(f"    输入点云形状：{points.shape}")
    
    with torch.no_grad():
        features = model.encode(points)
    
    print(f"    输出特征形状：{features.shape}")
    print(f"    [✓] 逐点特征提取成功")
    
    # 测试单个 patch
    print("\n[3] 测试单个 patch 编码（兼容接口）...")
    patch = torch.randn(50, 3).to(device)  # 50 个点的局部 patch
    print(f"    输入 patch 形状：{patch.shape}")
    
    with torch.no_grad():
        feat = model.encode(patch)
    
    print(f"    输出特征形状：{feat.shape}")
    print(f"    [✓] Patch 编码成功")
    
    return True

def test_pointnet2_ssg():
    """测试简化版 PointNet2 SSG"""
    
    config = {
        'INPUT_DIM': 3,
        'OUTPUT_DIM': 256
    }
    
    print("\n" + "=" * 60)
    print("PointNet2 SSG (Single-Scale Grouping) 测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] 使用设备：{device}")
    
    model = PointNet2SSG(config).to(device)
    print(f"[✓] 模型创建成功")
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    总参数量：{total_params:,}")
    print(f"    可训练参数量：{trainable_params:,}")
    
    # 测试全局特征提取
    print("\n[2] 测试全局特征提取...")
    batch_size = 2
    num_points = 1024
    
    points = torch.randn(batch_size, num_points, 3).to(device)
    print(f"    输入点云形状：{points.shape}")
    
    with torch.no_grad():
        features = model.encode(points)
    
    print(f"    输出特征形状：{features.shape}")
    print(f"    [✓] 全局特征提取成功")
    
    # 测试 patch 互补性（模拟实际使用场景）
    print("\n[3] 测试 patch 互补性检查...")
    patch1 = torch.randn(50, 3).to(device)
    patch2 = torch.randn(50, 3).to(device)
    
    print(f"    Patch1 形状：{patch1.shape}")
    print(f"    Patch2 形状：{patch2.shape}")
    
    with torch.no_grad():
        feat1 = model.encode(patch1)
        feat2 = model.encode(patch2)
        
        # 计算余弦相似度
        feat1_norm = feat1 / (feat1.norm() + 1e-8)
        feat2_norm = feat2 / (feat2.norm() + 1e-8)
        similarity = torch.dot(feat1_norm.flatten(), feat2_norm.flatten()).item()
    
    print(f"    特征 1 形状：{feat1.shape}")
    print(f"    特征 2 形状：{feat2.shape}")
    print(f"    余弦相似度：{similarity:.4f}")
    print(f"    [✓] Patch 互补性检查成功")
    
    return True

if __name__ == '__main__':
    print("\nStarting PointNet++ tests...\n")
    
    success1 = test_pointnet2_encoder()
    success2 = test_pointnet2_ssg()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ 所有 PointNet++ 测试通过！")
    else:
        print("❌ 部分测试失败！")
    print("=" * 60)
