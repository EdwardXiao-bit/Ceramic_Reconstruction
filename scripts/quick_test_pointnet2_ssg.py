"""
快速测试 PointNet++ SSG
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.pointnet2 import PointNet2SSG

print("测试 PointNet2 SSG...")

config = {'INPUT_DIM': 3, 'OUTPUT_DIM': 256}
device = torch.device('cpu')
model = PointNet2SSG(config).to(device)

# 测试 patch 编码
patch1 = torch.randn(50, 3).to(device)
patch2 = torch.randn(50, 3).to(device)

with torch.no_grad():
    feat1 = model.encode(patch1)
    feat2 = model.encode(patch2)

print(f"✓ Patch 输入形状：{patch1.shape}")
print(f"✓ 特征输出形状：{feat1.shape}")
print(f"✓ 特征相似度：{torch.nn.functional.cosine_similarity(feat1, feat2, dim=0).item():.4f}")
print("\nPointNet++ SSG 测试成功！")
