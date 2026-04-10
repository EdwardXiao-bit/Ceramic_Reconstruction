"""
全局拼接运行脚本
基于边界验证的最后一次运行结果执行全局拼接
"""

import sys
import os
import json
import numpy as np
import open3d as o3d
from datetime import datetime
import glob

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def find_latest_boundary_validation_result():
    """查找最新的边界验证结果"""
    validation_dir = os.path.join(project_root, "results", "boundary_validation")
    
    if not os.path.exists(validation_dir):
        print(f"❌ 边界验证结果目录不存在：{validation_dir}")
        return None
    
    # 查找所有验证结果文件
    pattern = os.path.join(validation_dir, "boundary_validation_test_*.json")
    files = glob.glob(pattern)
    
    if not files:
        print("❌ 未找到边界验证结果文件")
        return None
    
    # 按创建时间排序，取最新的
    latest_file = max(files, key=os.path.getctime)
    print(f"✓ 使用最新的边界验证结果：{os.path.basename(latest_file)}")
    
    return latest_file


def load_fragment_with_fallback(fragment_id, boundary_data, data_dir):
    """修复版碎片加载：确保point_cloud始终有效"""
    import open3d as o3d
    import numpy as np
    from pathlib import Path

    class MockFragment:
        def __init__(self, fid):
            self.id = fid
            self.file_name = f"{fid + 2}.obj"
            self.point_cloud = None
            self.mesh = None
            self.boundary_points = None
            self.section_patch = None

    fragment = MockFragment(fragment_id)
    frag_key = f"fragment_{fragment_id}"

    # 加载边界数据
    if frag_key in boundary_data:
        bd = boundary_data[frag_key]

        if 'boundary_points' in bd:
            bp = np.array(bd['boundary_points'])
            boundary_pcd = o3d.geometry.PointCloud()
            boundary_pcd.points = o3d.utility.Vector3dVector(bp)
            fragment.boundary_points = boundary_pcd

        if 'section_patch_points' in bd:
            sp = np.array(bd['section_patch_points'])
            section_pcd = o3d.geometry.PointCloud()
            section_pcd.points = o3d.utility.Vector3dVector(sp)
            fragment.section_patch = section_pcd

        if 'rim_curve' in bd:
            fragment.rim_curve = np.array(bd['rim_curve'])

    # 尝试加载真实网格
    obj_file = Path(data_dir) / f"{fragment_id + 2}.obj"
    if obj_file.exists():
        try:
            mesh = o3d.io.read_triangle_mesh(str(obj_file))
            if len(mesh.vertices) > 0:
                fragment.mesh = mesh
                fragment.point_cloud = mesh.sample_points_uniformly(
                    number_of_points=min(5000, len(mesh.vertices) * 2))
        except Exception as e:
            print(f"  [警告] 加载网格失败: {e}")

    # fallback：用边界点云作为point_cloud
    if fragment.point_cloud is None and fragment.boundary_points is not None:
        print(f"  [fallback] 碎片{fragment_id}用边界点云作为point_cloud")
        fragment.point_cloud = fragment.boundary_points

    # 最终fallback：创建虚拟点云
    if fragment.point_cloud is None:
        print(f"  [fallback] 碎片{fragment_id}创建虚拟点云")
        dummy_pts = np.random.randn(100, 3) * 0.1
        dummy_pcd = o3d.geometry.PointCloud()
        dummy_pcd.points = o3d.utility.Vector3dVector(dummy_pts)
        fragment.point_cloud = dummy_pcd

    return fragment


def load_fragments():
    """加载所有碎片数据"""
    from pathlib import Path
    
    # 确定数据目录
    data_dir = Path(project_root) / 'data' / 'eg1'
    if not data_dir.exists():
        print(f"⚠ 数据目录不存在: {data_dir}，尝试使用默认路径")
        data_dir = Path('data/eg1')
    
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return []
    
    # 查找所有 OBJ 文件
    obj_files = sorted(data_dir.glob('*.obj'))
    
    if not obj_files:
        print(f"❌ 在 {data_dir} 中未找到 OBJ 文件")
        return []
    
    print(f"  发现 {len(obj_files)} 个碎片文件")
    
    # 从文件名提取碎片ID（假设文件名为 2.obj, 3.obj, ...）
    fragments = []
    for obj_file in obj_files:
        try:
            # 从文件名提取 ID（例如 "2.obj" -> 0, "3.obj" -> 1）
            fid = int(obj_file.stem) - 2
            print(f"  加载碎片 {fid} ({obj_file.name})...")
            
            # 创建空的 boundary_data（因为这里不需要）
            fragment = load_fragment_with_fallback(fid, {}, data_dir)
            if fragment.point_cloud is not None:
                fragments.append(fragment)
        except Exception as e:
            print(f"  ⚠ 跳过 {obj_file.name}: {e}")
    
    print(f"✓ 成功加载 {len(fragments)} 个碎片")
    return fragments


def extract_validation_results(validation_data: dict) -> list:
    """从验证数据中提取验证结果"""
    detailed_results = validation_data.get('detailed_results', [])
    
    # 转换为全局拼接需要的格式
    formatted_results = []
    
    for result in detailed_results:
        pair = result.get('pair')
        vr = result.get('result', {})
        
        if not vr.get('success', False):
            continue
        
        # 提取关键信息
        final_scores = vr.get('final_scores', {})
        intermediate = vr.get('intermediate_results', {})
        
        formatted = {
            'pair': pair,
            'success': True,
            'total_score': final_scores.get('total_score', 0.0),
            'validation_status': final_scores.get('validation_status', 'unknown'),
            'feature_matching': intermediate.get('feature_matching', {}),
            'complementarity_check': intermediate.get('complementarity_check', {}),
            'local_alignment': intermediate.get('local_alignment', {}),
            'collision_detection': intermediate.get('collision_detection', {}),
            'processing_time': vr.get('processing_time', 0.0)
        }
        
        formatted_results.append(formatted)
    
    print(f"✓ 提取{len(formatted_results)}个有效验证对")
    return formatted_results


def run_global_assembly():
    """运行全局拼接流程"""
    print("\n" + "=" * 60)
    print("全局拼接运行脚本")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # 步骤 1: 加载边界验证结果
    print("\n【步骤 1】加载边界验证结果...")
    validation_file = find_latest_boundary_validation_result()
    
    if validation_file is None:
        print("❌ 无法加载边界验证结果，退出")
        return None
    
    with open(validation_file, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)
    
    validation_results = extract_validation_results(validation_data)
    
    if len(validation_results) == 0:
        print("❌ 没有有效的验证结果，退出")
        return None
    
    # 步骤 2: 加载碎片数据
    print("\n【步骤 2】加载碎片数据...")
    fragments = load_fragments()
    
    if len(fragments) == 0:
        print("❌ 无法加载碎片数据，退出")
        return None
    
    # 步骤 3: 配置全局拼接参数
    print("\n【步骤 3】配置参数...")
    config = {
        'use_g2o': False,  # 如果安装了 g2o 设为 True
        'use_ceres': False,  # 如果安装了 Ceres 设为 True
        'max_iterations': 100,
        'enable_texture_correction': True,
        'texture_correction': {
            'min_texture_similarity': 0.6,
            'correction_weight': 0.3
        },
        'collision_method': 'voxel',
        'voxel_size': 0.01
    }
    
    print("  配置参数:")
    print(f"    - Pose Graph 优化：g2o={config['use_g2o']}, Ceres={config['use_ceres']}")
    print(f"    - 最大迭代次数：{config['max_iterations']}")
    print(f"    - 纹样校正：{'启用' if config['enable_texture_correction'] else '禁用'}")
    print(f"    - 碰撞检测方法：{config['collision_method']}")
    
    # 步骤 4: 运行全局拼接流水线
    print("\n【步骤 4】运行全局拼接...")
    from src.assembly.pipeline import GlobalAssemblyPipeline
    
    pipeline = GlobalAssemblyPipeline(config=config)
    result = pipeline.run(fragments, validation_results)
    
    # 步骤 5: 保存结果
    print("\n【步骤 5】保存结果...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建带时间戳的运行文件夹
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]  # 包含毫秒
    results_dir = os.path.join(project_root, "results", "assembly", f"run_{run_timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存 JSON 结果
    output_json = os.path.join(results_dir, f"global_assembly.json")
    pipeline.save_result(result, output_json)
    
    # 保存简要报告
    report_path = os.path.join(results_dir, f"global_assembly_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("全局拼接结果报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"运行时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理时长：{(datetime.now() - start_time).total_seconds():.2f}秒\n\n")
        
        if result['success']:
            f.write("状态：成功 ✓\n\n")
            f.write("统计信息:\n")
            stats = result['statistics']
            f.write(f"  - 碎片总数：{result['fragments_count']}\n")
            f.write(f"  - 验证对数：{result['validation_pairs_count']}\n")
            f.write(f"  - 连通子图：{stats['connected_components']}\n")
            f.write(f"  - 匹配边数：{stats['matched_edges']}\n")
            f.write(f"  - 优化碎片：{stats['optimized_fragments']}\n")
            f.write(f"  - 碰撞对数：{stats['collision_pairs']}\n")
            f.write(f"  - 碰撞体积：{stats['total_collision_volume']:.6f}\n")
        else:
            f.write("状态：失败 ❌\n")
            if 'error' in result:
                f.write(f"错误信息：{result['error']}\n")
    
    print(f"✓ 结果已保存到：{output_json}")
    print(f"✓ 报告已保存到：{report_path}")
    
    # 步骤 6: 可视化选项
    print("\n【步骤 6】可视化...")
    if result['success']:
        visualize = input("是否可视化拼接结果？(y/n): ").strip().lower()
        
        if visualize == 'y':
            try:
                poses_dict = result['poses']
                poses = {int(k): np.array(v) for k, v in poses_dict.items()}
                
                print("正在生成可视化...")
                combined_pcd = pipeline.visualize_result(fragments, poses, show_collision=False)
                
                # 保存可视化点云（保存到当前运行文件夹）
                pcd_output = os.path.join(results_dir, f"assembly_pointcloud.ply")
                o3d.io.write_point_cloud(pcd_output, combined_pcd)
                print(f"✓ 点云已保存：{pcd_output}")
                
            except Exception as e:
                print(f"⚠ 可视化失败：{e}")
    
    print("\n" + "=" * 60)
    print("全局拼接完成!")
    print("=" * 60)
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print(f"总耗时：{total_time:.2f}秒")
    
    return result


if __name__ == "__main__":
    try:
        result = run_global_assembly()
        
        if result and result.get('success', False):
            print("\n✓ 全局拼接成功完成!")
            sys.exit(0)
        else:
            print("\n❌ 全局拼接失败!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n用户中断运行")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 运行出现异常：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
