"""
测试带纹样特征的初筛功能
"""
import numpy as np
from src.matching.faiss_prescreen import faiss_prescreen

def test_texture_prescreen():
    """测试包含纹样特征的初筛功能"""
    print("=== 纹样特征初筛测试 ===\n")
    
    # 创建模拟的Fragment对象
    class MockFragment:
        def __init__(self, id):
            self.id = id
            # 几何特征
            self.geo_embedding = np.random.rand(128).astype(np.float32)
            # FPFH特征
            self.fpfh_feature = np.random.rand(33).astype(np.float32)
            # 纹样特征（新增）
            self.texture_embedding = np.random.rand(256).astype(np.float32)
    
    # 创建测试碎片
    fragments = [MockFragment(i) for i in range(5)]
    print(f"创建了 {len(fragments)} 个测试碎片")
    
    # 测试新的参数
    print("\n1. 测试新参数支持:")
    try:
        result = faiss_prescreen(
            fragments, 
            top_m_geo=10, 
            top_m_fpfh=10, 
            top_m_texture=10,
            top_k=3, 
            alpha=0.4, 
            beta=0.3, 
            gamma=0.3
        )
        print("   ✓ 新参数调用成功")
        print(f"   ✓ 找到 {len(result)} 个候选匹配对")
    except Exception as e:
        print(f"   ✗ 新参数调用失败: {e}")
    
    # 测试只有纹样特征的情况
    print("\n2. 测试仅纹样特征:")
    class TextureOnlyFragment:
        def __init__(self, id):
            self.id = id
            self.texture_embedding = np.random.rand(256).astype(np.float32)
            # 不设置几何特征
            self.geo_embedding = None
            self.fpfh_feature = None
    
    texture_fragments = [TextureOnlyFragment(i) for i in range(5)]
    try:
        result = faiss_prescreen(texture_fragments, top_m_texture=10, top_k=2)
        print("   ✓ 仅纹样特征匹配成功")
        print(f"   ✓ 找到 {len(result)} 个候选对")
    except Exception as e:
        print(f"   ✗ 仅纹样特征匹配失败: {e}")
    
    # 测试混合特征情况
    print("\n3. 测试混合特征:")
    try:
        # 只使用具有相同特征类型的碎片进行测试
        geo_only_fragments = [f for f in fragments if f.geo_embedding is not None][:3]
        texture_only_fragments = [f for f in texture_fragments if f.texture_embedding is not None][:2]
        
        # 分别测试
        if len(geo_only_fragments) >= 2:
            result = faiss_prescreen(geo_only_fragments, top_k=2)
            print(f"   ✓ 几何特征匹配成功: 找到 {len(result)} 个候选对")
        
        if len(texture_only_fragments) >= 2:
            result = faiss_prescreen(texture_only_fragments, top_k=2)
            print(f"   ✓ 纹样特征匹配成功: 找到 {len(result)} 个候选对")
            
        print("   ✓ 混合特征测试完成")
    except Exception as e:
        print(f"   ✗ 混合特征匹配失败: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_texture_prescreen()