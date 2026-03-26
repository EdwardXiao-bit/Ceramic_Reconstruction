"""
测试 3D CNN 模型
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.models.cnn_3d import PointNet3DCNN, Light3DCNN

def test_full_version():
    """测试完整版 3D CNN"""
    print("=" * 60)
    print("完整版 3D CNN 测试")
    print("=" * 60)
    
    # 创建配置
    config = {
        'VOXEL_RESOLUTION': 0.01,
        'VOXEL_GRID_SIZE': 32,
        'BASE_CHANNELS': 32
    }
    
    print("\n[1] 创建模型...")
    cnn = PointNet3DCNN(config)
    print(f"[✓] 模型创建成功")
    
    # 打印参数量
    total_params = sum(p.numel() for p in cnn.model.parameters())
    print(f"    总参数量：{total_params:,}")
    
    # 测试互补性预测
    print("\n[2] 测试互补性预测...")
    patch1 = np.random.randn(50, 3).astype(np.float32)
    patch2 = np.random.randn(50, 3).astype(np.float32)
    
    print(f"    Patch1: {patch1.shape}")
    print(f"    Patch2: {patch2.shape}")
    
    score = cnn.predict_complementarity(patch1, patch2)
    
    print(f"    互补性得分：{score:.4f}")
    print(f"    [✓] 预测成功")
    
    return True

def test_light_version():
    """测试轻量版 3D CNN"""
    print("\n" + "=" * 60)
    print("轻量版 3D CNN 测试")
    print("=" * 60)
    
    print("\n[1] 创建模型...")
    light_cnn = Light3DCNN()
    print(f"[✓] 模型创建成功")
    
    # 打印参数量
    total_params = sum(p.numel() for p in light_cnn.model.parameters())
    print(f"    总参数量：{total_params:,}")
    
    # 测试互补性预测
    print("\n[2] 测试互补性预测...")
    patch1 = np.random.randn(50, 3).astype(np.float32)
    patch2 = np.random.randn(50, 3).astype(np.float32)
    
    print(f"    Patch1: {patch1.shape}")
    print(f"    Patch2: {patch2.shape}")
    
    score = light_cnn.predict_complementarity(patch1, patch2)
    
    print(f"    互补性得分：{score:.4f}")
    print(f"    [✓] 预测成功")
    
    return True

def test_batch_prediction():
    """批量预测测试"""
    print("\n" + "=" * 60)
    print("批量预测测试")
    print("=" * 60)
    
    light_cnn = Light3DCNN()
    
    print("\n生成多组随机 patch...")
    scores = []
    
    for i in range(5):
        patch1 = np.random.randn(50, 3).astype(np.float32)
        patch2 = np.random.randn(50, 3).astype(np.float32)
        
        score = light_cnn.predict_complementarity(patch1, patch2)
        scores.append(score)
        print(f"    第 {i+1} 组：{score:.4f}")
    
    print(f"\n平均得分：{np.mean(scores):.4f}")
    print(f"标准差：{np.std(scores):.4f}")
    print(f"[✓] 批量测试完成")
    
    return True

if __name__ == '__main__':
    print("\nStarting 3D CNN tests...\n")
    
    try:
        success1 = test_full_version()
        success2 = test_light_version()
        success3 = test_batch_prediction()
        
        print("\n" + "=" * 60)
        if success1 and success2 and success3:
            print("✅ All 3D CNN tests passed!")
        else:
            print("❌ Some tests failed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
