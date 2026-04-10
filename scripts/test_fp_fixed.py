"""
测试修复后的 FP 层
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

try:
    with torch.no_grad():
        features = model.encode(points)
    
    print(f"SUCCESS!")
    print(f"Input: {points.shape}")
    print(f"Output: {features.shape}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
