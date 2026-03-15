"""
全局拼接可视化脚本
用于查看全局拼接结果
"""

import sys
import os
import json
import numpy as np
import open3d as o3d
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def visualize_global_assembly(json_file: str):
    """从 JSON 文件加载并可视化全局拼接结果"""
    
    print(f"加载拼接结果：{json_file}")
    
    # 加载 JSON 结果
    with open(json_file, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    if not result.get('success', False):
        print("❌ 拼接失败，无法可视化")
        return
    
    # 加载碎片数据
    from src.common.io import load_fragments
    
    data_dirs = [
        os.path.join(project_root, "data", "eg1"),
        os.path.join(project_root, "data", "demo"),
        os.path.join(project_root, "data", "input")
    ]
    
    data_dir = None
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            break
    
    if data_dir is None:
        print("❌ 未找到数据目录")
        return
    
    fragments = load_fragments(data_dir)
    print(f"✓ 加载 {len(fragments)} 个碎片")
    
    # 提取位姿
    poses_dict = result['poses']
    poses = {int(k): np.array(v) for k, v in poses_dict.items()}
    
    print(f"✓ 提取 {len(poses)} 个位姿")
    
    # 创建组合点云
    combined_pcd = o3d.geometry.PointCloud()
    
    for fragment in fragments:
        frag_id = fragment.id
        if frag_id not in poses:
            continue
        
        pose = poses[frag_id]
        
        # 获取碎片点云
        if hasattr(fragment, 'point_cloud') and fragment.point_cloud is not None:
            pcd = fragment.point_cloud
        elif hasattr(fragment, 'mesh') and fragment.mesh is not None:
            # 从网格采样
            pcd = fragment.mesh.sample_points_uniformly(number_of_points=5000)
        else:
            continue
        
        # 应用变换
        transformed_pcd = pcd.transform(pose)
        combined_pcd += transformed_pcd
        
        print(f"  - 碎片 {frag_id}: {len(pcd.points)} 个点")
    
    print(f"\n组合点云总计：{len(combined_pcd.points)} 个点")
    
    # 可视化
    print("\n打开可视化窗口...")
    print("提示：按 Q 键关闭窗口")
    
    o3d.visualization.draw_geometries([combined_pcd])
    
    # 保存点云（保存到结果文件夹中）
    output_dir = os.path.dirname(json_file)  # 使用 JSON 文件所在目录
    pcd_file = os.path.join(output_dir, f"assembly_pointcloud.ply")
    o3d.io.write_point_cloud(pcd_file, combined_pcd)
    print(f"✓ 点云已保存到：{pcd_file}")
    
    # 生成统计报告
    stats = result.get('statistics', {})
    print("\n" + "=" * 60)
    print("拼接统计信息")
    print("=" * 60)
    print(f"碎片总数：{result.get('fragments_count', 0)}")
    print(f"验证对数：{result.get('validation_pairs_count', 0)}")
    print(f"连通子图：{stats.get('connected_components', 0)}")
    print(f"匹配边数：{stats.get('matched_edges', 0)}")
    print(f"优化碎片：{stats.get('optimized_fragments', 0)}")
    print(f"碰撞对数：{stats.get('collision_pairs', 0)}")
    print(f"处理时间：{stats.get('processing_time', 0):.2f}秒")
    print("=" * 60)


if __name__ == "__main__":
    import glob
    
    # 查找最新的拼接结果（支持子文件夹）
    results_dir = os.path.join(project_root, "results", "assembly")
    
    if not os.path.exists(results_dir):
        print(f"❌ 结果目录不存在：{results_dir}")
        sys.exit(1)
    
    # 查找所有 run_* 文件夹
    run_dirs = [d for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('run_')]
    
    if not run_dirs:
        # 如果没有子文件夹，直接在 assembly 目录下查找
        pattern = os.path.join(results_dir, "global_assembly_*.json")
        files = glob.glob(pattern)
    else:
        # 在最新的 run_* 文件夹中查找
        latest_run = max(run_dirs)
        run_path = os.path.join(results_dir, latest_run)
        pattern = os.path.join(run_path, "global_assembly.json")
        files = glob.glob(pattern)
        
        # 如果最新运行没有结果，查找所有 run 文件夹
        if not files:
            all_patterns = [
                os.path.join(results_dir, d, "global_assembly.json")
                for d in sorted(run_dirs, reverse=True)
            ]
            for p in all_patterns:
                files.extend(glob.glob(p))
                if files:
                    break
    
    if not files:
        print("❌ 未找到拼接结果文件")
        sys.exit(1)
    
    # 按创建时间排序，取最新的
    latest_file = max(files, key=os.path.getctime)
    
    try:
        visualize_global_assembly(latest_file)
    except Exception as e:
        print(f"\n❌ 可视化失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
