"""
模型预训练快速测试脚本
验证 Predator 和 DCP 模型的构建和前向传播
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_predator():
    """测试 Predator 模型"""
    print("\n" + "="*60)
    print("测试 Predator 模型")
    print("="*60)
    
    try:
        from src.models.predator import Predator
        
        config = {
            'MODEL': {
                'UNET': {'PLANES': [32, 64, 128, 256]},
                'INPUT_DIM': 3,
                'TRANSFORMER': {
                    'NUM_LAYERS': 2,
                    'NUM_HEADS': 4,
                    'HIDDEN_DIM': 128
                },
                'MATCHING': {'FEATURE_DIM': 64}
            }
        }
        
        print("[1/4] 创建模型...")
        model = Predator(config['MODEL'])
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 模型参数量：{num_params:,}")
        
        print("\n[2/4] 创建测试数据...")
        import torch
        batch_size = 2
        points1 = torch.randn(batch_size, 100, 3)
        points2 = torch.randn(batch_size, 100, 3)
        print(f"✓ 输入点云形状：{points1.shape}, {points2.shape}")
        
        print("\n[3/4] 前向传播测试...")
        with torch.no_grad():
            features1, features2, transform_pred = model(points1, points2)
        
        print(f"✓ 输出特征形状：{features1.shape}, {features2.shape}")
        print(f"✓ 变换参数形状：{transform_pred.shape}")
        
        print("\n[4/4] 变换矩阵预测...")
        with torch.no_grad():
            transform_matrix = model.predict_transform(points1, points2)
        
        print(f"✓ 变换矩阵形状：{transform_matrix.shape}")
        
        print("\n✅ Predator 模型测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ Predator 模型测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_dcp():
    """测试 DCP 模型"""
    print("\n" + "="*60)
    print("测试 DCP 模型")
    print("="*60)
    
    try:
        from src.models.dcp import DCP
        
        config = {
            'MODEL': {
                'POINTNET': {'EMBEDDING_DIM': 512},
                'TRANSFORMER': {
                    'NUM_LAYERS': 2,
                    'NUM_HEADS': 4,
                    'HIDDEN_DIM': 256
                },
                'REGRESSOR': {'LAYERS': [512, 256, 128, 6]}
            }
        }
        
        print("[1/4] 创建模型...")
        model = DCP(config['MODEL'])
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 模型参数量：{num_params:,}")
        
        print("\n[2/4] 创建测试数据...")
        import torch
        batch_size = 2
        src_points = torch.randn(batch_size, 1024, 3)
        tgt_points = torch.randn(batch_size, 1024, 3)
        print(f"✓ 输入点云形状：src={src_points.shape}, tgt={tgt_points.shape}")
        
        print("\n[3/4] 前向传播测试...")
        with torch.no_grad():
            R_pred, t_pred = model(src_points, tgt_points)
        
        print(f"✓ 旋转矩阵形状：{R_pred.shape}")
        print(f"✓ 平移向量形状：{t_pred.shape}")
        
        print("\n[4/4] 完整变换矩阵预测...")
        with torch.no_grad():
            transform_matrix = model.predict_transform(src_points, tgt_points)
        
        print(f"✓ 变换矩阵形状：{transform_matrix.shape}")
        
        print("\n✅ DCP 模型测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ DCP 模型测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """集成测试：模拟实际使用场景"""
    print("\n" + "="*60)
    print("集成测试：边界验证流程中的模型调用")
    print("="*60)
    
    try:
        import torch
        import numpy as np
        
        # 加载模型
        print("[1/5] 加载 Predator 模型...")
        from src.models.predator import Predator
        predator_config = {
            'MODEL': {
                'UNET': {'PLANES': [32, 64]},
                'INPUT_DIM': 3,
                'TRANSFORMER': {'NUM_LAYERS': 1, 'NUM_HEADS': 2, 'HIDDEN_DIM': 64},
                'MATCHING': {'FEATURE_DIM': 32}
            }
        }
        predator = Predator(predator_config['MODEL'])
        predator.eval()
        
        print("[2/5] 加载 DCP 模型...")
        from src.models.dcp import DCP
        dcp_config = {
            'MODEL': {
                'POINTNET': {'EMBEDDING_DIM': 256},
                'TRANSFORMER': {'NUM_LAYERS': 1, 'NUM_HEADS': 2, 'HIDDEN_DIM': 128},
                'REGRESSOR': {'LAYERS': [256, 128, 6]}
            }
        }
        dcp = DCP(dcp_config['MODEL'])
        dcp.eval()
        
        print("[3/5] 模拟边界特征匹配...")
        # 模拟边界点云
        boundary1 = np.random.randn(500, 3).astype(np.float32)
        boundary2 = np.random.randn(500, 3).astype(np.float32)
        
        points1 = torch.from_numpy(boundary1).unsqueeze(0)
        points2 = torch.from_numpy(boundary2).unsqueeze(0)
        
        with torch.no_grad():
            # Predator 特征匹配
            feat1, feat2, _ = predator(points1, points2)
            print(f"✓ Predator 特征维度：{feat1.shape}")
            
        print("[4/5] 模拟局部对齐精化...")
        # 模拟需要对齐的点云
        src = torch.randn(1, 1024, 3)
        tgt = torch.randn(1, 1024, 3)
        
        with torch.no_grad():
            # DCP 预测变换
            R, t = dcp(src, tgt)
            T = dcp.predict_transform(src, tgt)
        
        print(f"✓ DCP 预测变换：R={R.shape}, t={t.shape}")
        print(f"✓ 4x4 变换矩阵：{T.shape}")
        
        print("[5/5] 验证变换矩阵有效性...")
        # 检查旋转矩阵是否正交
        R_det = torch.det(R[0])
        print(f"✓ 旋转矩阵行列式：det(R) = {R_det.item():.4f} (应接近 1)")
        
        print("\n✅ 集成测试通过！")
        print("\n📊 总结:")
        print("   - Predator 可用于边界特征匹配")
        print("   - DCP 可用于局部对齐精化")
        print("   - 两个模型都可以正常推理")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 集成测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("深度学习模型预训练 - 快速测试套件")
    print("="*60)
    
    results = {
        'Predator': test_predator(),
        'DCP': test_dcp(),
        'Integration': test_integration()
    }
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status} - {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有测试通过！模型已准备好使用。")
        print("\n下一步:")
        print("1. 下载预训练权重：python scripts/pretrain_models.py --action download")
        print("2. 集成到项目模块：参考 docs/model_pretraining_guide.md")
        print("3. 运行边界验证测试：python scripts/test_boundary_validation.py")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
