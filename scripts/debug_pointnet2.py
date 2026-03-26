"""
调试 PointNet2 维度问题
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

# 测试小批量
points = torch.randn(1, 1024, 3)  # 减少点数
print(f"输入：{points.shape}")

with torch.no_grad():
    xyz = points
    features = points.transpose(1, 2)
    print(f"初始 features: {features.shape}")
    
    # SA1
    l1_xyz, l1_features = model.sa1(xyz, features)
    print(f"SA1: xyz={l1_xyz.shape}, features={l1_features.shape}")
    
    # SA2
    l2_xyz, l2_features = model.sa2(l1_xyz, l1_features)
    print(f"SA2: xyz={l2_xyz.shape}, features={l2_features.shape}")
    
    # SA3
    l3_xyz, l3_features = model.sa3(l2_xyz, l2_features)
    print(f"SA3: xyz={l3_xyz.shape}, features={l3_features.shape}")
    
    # SA4
    l4_xyz, l4_features = model.sa4(l3_xyz, l3_features)
    print(f"SA4: xyz={l4_xyz.shape}, features={l4_features.shape}")
    
    # FP3
    print(f"\nFP3 输入：l3_features={l3_features.shape}, l4_features={l4_features.shape}")
    try:
        l3_features_up = model.fp3(l3_xyz, l4_xyz, l3_features, l4_features)
        print(f"FP3 输出：{l3_features_up.shape}")
    except Exception as e:
        print(f"FP3 失败：{e}")
