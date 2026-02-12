#!/usr/bin/env python3
"""
SuperGlue纹理匹配运行脚本
独立于几何匹配的并行技术路线
对应算法文档中的纹样匹配流程
"""
import sys
import os
from pathlib import Path
import argparse
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.common.io import load_fragments
from src.texture_matching.superglue_integration import (
    integrate_texture_matching,
    check_superglue_availability,
    get_texture_matching_fallback
)
from src.common.base import Fragment


def main():
    """主函数 - SuperGlue纹理匹配流程"""
    parser = argparse.ArgumentParser(description='SuperGlue陶瓷碎片纹理匹配')
    parser.add_argument('--data_dir', type=str, default='data/demo',
                       help='碎片数据目录')
    parser.add_argument('--top_k', type=int, default=10,
                       help='返回Top-K匹配候选')
    parser.add_argument('--resolution', type=int, nargs=2, default=[512, 512],
                       help='纹理图像分辨率')
    parser.add_argument('--output_dir', type=str, default='results/texture_matching',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(" SuperGlue陶瓷碎片纹理匹配系统 ")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"Top-K: {args.top_k}")
    print(f"图像分辨率: {args.resolution}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    # 检查SuperGlue可用性
    print("=== 环境检查 ===")
    if check_superglue_availability():
        print("✓ SuperGlue可用")
    else:
        print("⚠ SuperGlue不可用，将使用备用方案")
    print()
    
    # 1. 加载碎片数据
    print("=== 数据加载 ===")
    try:
        fragments = load_fragments(args.data_dir)
        if not fragments:
            print(f"✗ 未在 {args.data_dir} 中找到有效碎片文件")
            return 1
        
        print(f"✓ 成功加载 {len(fragments)} 个碎片:")
        for i, frag in enumerate(fragments):
            file_name = getattr(frag, 'file_name', f'fragment_{i}')
            print(f"   碎片{i}: {file_name}")
        print()
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return 1
    
    # 2. 执行纹理匹配
    print("=== 纹理匹配流程 ===")
    try:
        candidates = integrate_texture_matching(fragments, top_k=args.top_k)
        print()
        
    except Exception as e:
        print(f"✗ 纹理匹配执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 3. 结果输出
    print("=== 匹配结果 ===")
    if candidates:
        print(f"找到 {len(candidates)} 个高相似度候选对:")
        print("-" * 50)
        print(f"{'序号':<4} {'碎片对':<15} {'相似度':<10} {'状态'}")
        print("-" * 50)
        
        for i, (idx1, idx2, similarity) in enumerate(candidates, 1):
            frag1_name = getattr(fragments[idx1], 'file_name', f'fragment_{idx1}')
            frag2_name = getattr(fragments[idx2], 'file_name', f'fragment_{idx2}')
            status = "✓ 高置信" if similarity > 0.5 else "○ 中等置信"
            
            print(f"{i:<4} {frag1_name[:7]}-{frag2_name[:7]:<8} {similarity:<10.3f} {status}")
        
        print("-" * 50)
        
        # 保存结果
        save_results(candidates, fragments, args.output_dir)
        
    else:
        print("未找到高相似度的碎片对")
    
    print()
    print("=== 流程完成 ===")
    print("SuperGlue纹理匹配流程执行完毕！")
    
    return 0


def save_results(candidates: list, fragments: list, output_dir: str):
    """保存匹配结果"""
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存匹配对信息
        result_file = os.path.join(output_dir, 'texture_matches.txt')
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("SuperGlue纹理匹配结果\n")
            f.write("=" * 50 + "\n\n")
            
            for i, (idx1, idx2, similarity) in enumerate(candidates, 1):
                frag1_name = getattr(fragments[idx1], 'file_name', f'fragment_{idx1}')
                frag2_name = getattr(fragments[idx2], 'file_name', f'fragment_{idx2}')
                
                f.write(f"{i}. 碎片对: {frag1_name} ↔ {frag2_name}\n")
                f.write(f"   相似度: {similarity:.4f}\n")
                f.write(f"   状态: {'高置信匹配' if similarity > 0.5 else '中等置信匹配'}\n\n")
        
        print(f"✓ 结果已保存至: {result_file}")
        
        # 保存详细数据
        npz_file = os.path.join(output_dir, 'texture_matches.npz')
        data_dict = {
            'candidates': np.array(candidates),
            'fragment_names': [getattr(f, 'file_name', f'fragment_{i}') 
                             for i, f in enumerate(fragments)]
        }
        np.savez(npz_file, **data_dict)
        print(f"✓ 数值数据已保存至: {npz_file}")
        
    except Exception as e:
        print(f"✗ 结果保存失败: {e}")


def demo_texture_extraction():
    """演示纹样提取功能"""
    print("\n=== 纹样提取演示 ===")
    
    # 这里可以添加交互式演示代码
    # 例如展示单个碎片的纹样提取过程
    print("纹样提取演示功能待实现...")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)