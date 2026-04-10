#!/usr/bin/env python3
"""
增强版SuperGlue纹理匹配运行脚本
提供完整的流水线执行和高级功能
"""
import sys
import os
from pathlib import Path
import argparse
import json

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.common.io import load_fragments
from src.texture_matching.advanced_matching import run_texture_matching_pipeline
from src.texture_matching.config import create_default_config_file, get_template_config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版SuperGlue陶瓷碎片纹理匹配')
    parser.add_argument('--data_dir', type=str, default='data/demo',
                       help='碎片数据目录')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--config_template', type=str, choices=['high_precision', 'fast_matching', 'balanced'],
                       default='balanced', help='预定义配置模板')
    parser.add_argument('--output_dir', type=str, default='results/texture_matching',
                       help='结果输出目录')
    parser.add_argument('--cache_dir', type=str, default='cache/texture_features',
                       help='特征缓存目录')
    parser.add_argument('--create_config', action='store_true',
                       help='创建默认配置文件')
    parser.add_argument('--use-superglue', action='store_true', default=True,
                       help='使用SuperGlue特征匹配（默认启用）')
    parser.add_argument('--no-superglue', dest='use_superglue', action='store_false',
                       help='禁用SuperGlue，使用传统ORB特征')
    
    args = parser.parse_args()
    
    # 处理互斥参数
    if hasattr(args, 'no_superglue') and args.no_superglue:
        args.use_superglue = False
    
    # 创建配置文件（如果需要）
    if args.create_config:
        config_path = args.config or 'configs/superglue_config.yaml'
        create_default_config_file(config_path)
        print(f"配置文件已创建: {config_path}")
        return 0
    
    # 确定使用的配置
    if args.config:
        config_path = args.config
    else:
        # 使用模板配置
        template_config = get_template_config(args.config_template)
        config_path = None  # 将使用模板配置
    
    print("=" * 70)
    print(" 增强版SuperGlue陶瓷碎片纹理匹配系统 ")
    print("=" * 70)
    print(f"数据目录: {args.data_dir}")
    print(f"配置模式: {args.config_template}")
    print(f"输出目录: {args.output_dir}")
    print(f"缓存目录: {args.cache_dir}")
    print()
    
    # 解决PyCharm工作目录问题：使用相对于项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, args.data_dir.replace('/', os.sep))
    
    # 加载碎片数据
    print("=== 数据加载 ===")
    try:
        fragments = load_fragments(data_dir)
        if not fragments:
            print(f"✗ 未在 {data_dir} 中找到有效碎片文件")
            print(f"  当前工作目录: {os.getcwd()}")
            print(f"  项目根目录: {project_root}")
            return 1
        
        print(f"✓ 成功加载 {len(fragments)} 个碎片:")
        for i, frag in enumerate(fragments):
            file_name = getattr(frag, 'file_name', f'fragment_{i}')
            print(f"   碎片{i}: {file_name}")
        print()
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return 1
    
    # 运行匹配流水线
    print("=== 纹理匹配流水线 ===")
    print(f"SuperGlue使用状态: {'启用' if args.use_superglue else '禁用'}")
    try:
        # 使用相对于项目根目录的输出路径
        output_dir = os.path.join(project_root, args.output_dir.replace('/', os.sep))
        
        report = run_texture_matching_pipeline(
            fragments=fragments,
            config_path=config_path,
            output_dir=output_dir,
            use_superglue=args.use_superglue  # 传递SuperGlue参数
        )
        print()
        
    except Exception as e:
        print(f"✗ 匹配流水线执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 显示结果摘要
    print("=== 匹配结果摘要 ===")
    print_summary(report)
    
    # 保存详细结果
    save_detailed_results(report, args.output_dir)
    
    print()
    print("=" * 70)
    print(" 纹理匹配流水线执行完成! ")
    print("=" * 70)
    
    return 0


def print_summary(report: dict):
    """打印匹配结果摘要"""
    print(f"处理时间: {report.get('timestamp', 'N/A')}")
    print(f"总碎片数: {report.get('total_fragments', 0)}")
    print(f"成功处理: {report.get('processed_fragments', 0)}")
    print(f"候选匹配: {report.get('total_candidates', 0)}")
    print()
    
    matches = report.get('matches', [])
    if matches:
        print("高置信匹配对:")
        print("-" * 50)
        for i, match in enumerate(matches[:10], 1):  # 显示前10个
            frag1, frag2 = match['fragment_pair']
            score = match['total_score']
            print(f"{i:2d}. {frag1[:15]:<15} ↔ {frag2[:15]:<15} (得分: {score:.3f})")
        
        if len(matches) > 10:
            print(f"... 还有 {len(matches) - 10} 个匹配对")
        print("-" * 50)
    else:
        print("未找到高置信匹配对")


def save_detailed_results(report: dict, output_dir: str):
    """保存详细结果"""
    try:
        # 保存CSV格式的结果
        csv_path = Path(output_dir) / 'matches_summary.csv'
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Rank,Fragment1,Fragment2,TotalScore,SG_Score,Emb_Score,Geo_Score\n")
            
            for i, match in enumerate(report.get('matches', []), 1):
                frag1, frag2 = match['fragment_pair']
                scores = match['component_scores']
                f.write(f"{i},{frag1},{frag2},{match['total_score']:.4f},"
                       f"{scores.get('sg_similarity', 0):.4f},"
                       f"{scores.get('emb_similarity', 0):.4f},"
                       f"{scores.get('geometric_score', 0):.4f}\n")
        
        print(f"✓ CSV结果已保存: {csv_path}")
        
        # 保存配置信息
        config_path = Path(output_dir) / 'used_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(report.get('config', {}), f, indent=2, ensure_ascii=False)
        print(f"✓ 使用的配置已保存: {config_path}")
        
    except Exception as e:
        print(f"✗ 详细结果保存失败: {e}")


def demo_advanced_features():
    """演示高级功能"""
    print("\n=== 高级功能演示 ===")
    print("1. 多尺度特征匹配")
    print("2. 几何约束匹配")
    print("3. 结果可视化")
    print("4. 性能优化")
    print("(详细实现请参考源代码)")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)