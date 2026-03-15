#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界验证测试脚本
使用现有的匹配结果进行边界验证功能测试
"""

import sys
import os
import json
import numpy as np
import open3d as o3d
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.boundary_validation import BoundaryValidator
# from src.utils.fragment_loader import load_fragments  # 暂时注释掉
from src.boundary_validation.config import get_config


def load_latest_matching_results():
    """加载最新的匹配结果"""
    matching_dir = project_root / "results" / "matching"

    # 找到最新的运行文件夹
    run_folders = [f for f in matching_dir.iterdir() if f.is_dir() and f.name.startswith('run_')]
    if not run_folders:
        raise FileNotFoundError("未找到任何运行结果文件夹")

    # 按时间排序，获取最新的
    latest_run = sorted(run_folders)[-1]
    print(f"使用最新运行结果: {latest_run.name}")

    # 加载匹配结果
    match_file = latest_run / f"fragment_matches_{latest_run.name.split('_')[1]}_{latest_run.name.split('_')[2]}.json"

    with open(match_file, 'r', encoding='utf-8') as f:
        matches = json.load(f)

    return matches, latest_run


def create_test_pairs_from_matches(matches_data, similarity_threshold=0.3):
    """从匹配结果创建测试对，基于相似度阈值选择所有符合条件的对
    
    Args:
        matches_data: 匹配结果数据
        similarity_threshold: 相似度阈值，默认0.3
    
    Returns:
        list: 符合条件的测试对列表
    """
    # 使用集合存储已处理的碎片ID对，避免重复
    processed_pairs = set()
    valid_pairs = []
    
    # 收集所有可能的匹配对
    all_candidates = []
    
    for frag_id, candidates in matches_data.items():
        frag1_original = int(frag_id)  # 保存原始frag_id，避免修改循环变量
        if candidates:
            for candidate in candidates:
                frag2 = int(candidate['matched_fragment'])
                similarity = candidate['similarity']
                
                # 确保较小的ID在前，避免(1,2)和(2,1)重复
                frag1 = frag1_original  # 使用原始值而不是被修改的frag1
                if frag1 > frag2:
                    frag1, frag2 = frag2, frag1
                
                pair_key = (frag1, frag2)
                
                # 检查条件：相似度阈值、避免自匹配、避免重复处理
                if (similarity >= similarity_threshold and 
                    frag1 != frag2 and 
                    pair_key not in processed_pairs):
                    
                    all_candidates.append({
                        'pair': pair_key,
                        'similarity': similarity,
                        'original_order': (int(frag_id), int(candidate['matched_fragment']))
                    })
                    processed_pairs.add(pair_key)
    
    # 按相似度排序
    all_candidates.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 选择所有的对（不限制碎片使用次数）
    selected_pairs = []
    
    for candidate in all_candidates:
        frag1, frag2 = candidate['pair']
        similarity = candidate['similarity']
        
        # 直接选择所有符合条件的对，不限制碎片重复使用
        selected_pairs.append(((frag1, frag2), similarity))
        
        # 如果需要更多对，可以放松这个限制
        # 目前先保证去重效果
    
    print(f"选择相似度 >= {similarity_threshold:.2f} 的所有唯一测试对:")
    for i, ((frag1, frag2), sim) in enumerate(selected_pairs):
        print(f"  {i + 1}. 碎片{frag1} - 碎片{frag2} (相似度: {sim:.4f})")
    
    if not selected_pairs:
        print(f"警告: 没有找到相似度 >= {similarity_threshold:.2f} 的碎片对")
        print("建议降低相似度阈值或检查匹配结果")
    
    return [pair for pair, _ in selected_pairs]


def load_latest_boundary_data():
    """加载最新的边界数据"""
    output_dir = project_root / "data" / "output"
    run_folders = [f for f in output_dir.iterdir() if f.is_dir() and f.name.startswith('run_')]

    if not run_folders:
        raise FileNotFoundError("未找到任何run文件夹，请先运行run_mvp.py")

    # 选择最新的run文件夹
    latest_run = sorted(run_folders)[-1]
    boundary_file = latest_run / "boundary_data.json"

    if not boundary_file.exists():
        raise FileNotFoundError(f"在{latest_run}中未找到boundary_data.json文件")

    with open(boundary_file, 'r', encoding='utf-8') as f:
        boundary_data = json.load(f)

    print(f"[成功加载边界数据]: {boundary_file}")
    print(f"  包含 {len(boundary_data)} 个碎片的边界信息")

    return boundary_data, latest_run


def load_fragment_data_with_boundary(fragment_id, boundary_data):
    """加载包含边界数据的碎片对象"""
    data_dir = project_root / "data" / "eg1"

    # 创建碎片对象
    class MockFragment:
        def __init__(self, fid):
            self.id = fid
            self.file_name = f"{fid + 2}.obj"  # OBJ文件命名规则
            self.point_cloud = None
            self.mesh = None
            self.boundary_points = None
            self.section_patch = None
            self.rim_curve = None
            self.profile_curve = None
            self.main_axis = None
            self.fpfh_feature = None
            self.texture_embedding = None
            self.geo_embedding = None

    fragment = MockFragment(fragment_id)

    # 从边界数据中加载预提取的边界信息
    frag_key = f"fragment_{fragment_id}"
    if frag_key in boundary_data:
        frag_boundary_data = boundary_data[frag_key]

        # 加载边界点
        if 'boundary_points' in frag_boundary_data:
            boundary_points = np.array(frag_boundary_data['boundary_points'])
            boundary_pcd = o3d.geometry.PointCloud()
            boundary_pcd.points = o3d.utility.Vector3dVector(boundary_points)
            fragment.boundary_points = boundary_pcd
            print(f"  [加载碎片{fragment_id}边界点]: {len(boundary_points)}个")

        # 加载断面patch
        if 'section_patch_points' in frag_boundary_data:
            section_points = np.array(frag_boundary_data['section_patch_points'])
            section_pcd = o3d.geometry.PointCloud()
            section_pcd.points = o3d.utility.Vector3dVector(section_points)
            fragment.section_patch = section_pcd
            print(f"  [加载碎片{fragment_id}断面patch]: {len(section_points)}个点")

        # 加载Rim曲线
        if 'rim_curve' in frag_boundary_data:
            rim_points = np.array(frag_boundary_data['rim_curve'])
            rim_pcd = o3d.geometry.PointCloud()
            rim_pcd.points = o3d.utility.Vector3dVector(rim_points)
            fragment.rim_curve = rim_points
            fragment.rim_pcd = rim_pcd
            print(f"  [加载碎片{fragment_id}Rim曲线]: {len(rim_points)}个点")

        # 加载主轴
        if 'main_axis' in frag_boundary_data:
            fragment.main_axis = np.array(frag_boundary_data['main_axis'])

        # 加载轮廓曲线
        if 'profile_curve' in frag_boundary_data:
            fragment.profile_curve = np.array(frag_boundary_data['profile_curve'])

    # 尝试加载实际的网格数据
    try:
        obj_file = data_dir / f"{fragment_id + 2}.obj"
        if obj_file.exists():
            mesh = o3d.io.read_triangle_mesh(str(obj_file))
            if mesh.has_vertices():
                fragment.mesh = mesh
                fragment.point_cloud = mesh.sample_points_uniformly(number_of_points=2000)
                print(f"  [加载碎片{fragment_id}网格数据]")
            else:
                print(f"  [警告: 碎片{fragment_id}网格无顶点数据]")
        else:
            print(f"  [警告: 碎片{fragment_id}的OBJ文件不存在]: {obj_file}")
    except Exception as e:
        print(f"  [警告: 加载碎片{fragment_id}网格时出错]: {e}")

    return fragment


def test_boundary_validation_pipeline():
    """测试完整的边界验证流水线"""
    print("=" * 60)
    print("边界验证功能测试")
    print("=" * 60)

    try:
        # 1. 加载匹配结果和边界数据
        print("\n1. 加载最新的匹配结果和边界数据...")
        matches_data, run_folder = load_latest_matching_results()
        boundary_data, boundary_run_folder = load_latest_boundary_data()

        # 2. 创建测试对（取消相似度阈值过滤，确保所有碎片对都参与验证）
        print("\n2. 创建测试碎片对...")
        # 根据项目要求，取消相似度阈值过滤，让所有碎片对都参与验证
        similarity_threshold = 0.0  # 设置为0，不过滤任何对
        test_pairs = create_test_pairs_from_matches(matches_data, similarity_threshold=similarity_threshold)

        # 3. 初始化边界验证器
        print("\n3. 初始化边界验证器...")
        from src.boundary_validation.validator import BoundaryValidator
        validator = BoundaryValidator()

        # 4. 对每对碎片进行边界验证
        print("\n4. 执行边界验证测试...")
        results = []

        for i, (frag1_id, frag2_id) in enumerate(test_pairs):
            print(f"\n--- 测试对 {i + 1}: 碎片{frag1_id} vs 碎片{frag2_id} ---")

            # 加载包含边界数据的碎片对象
            frag1 = load_fragment_data_with_boundary(frag1_id, boundary_data)
            frag2 = load_fragment_data_with_boundary(frag2_id, boundary_data)

            # 执行边界验证
            try:
                result = validator.validate_fragment_pair(frag1, frag2)
                results.append({
                    'pair': (frag1_id, frag2_id),
                    'result': result
                })

                if result['success']:
                    print(f"  [验证成功]")
                    print(f"    综合得分：{result['final_scores']['total_score']:.3f}")
                    print(f"    验证状态：{result['final_scores']['validation_status']}")
                    print(f"    处理时间：{result['processing_time']:.2f}秒")
                else:
                    print(f"  [验证失败] {result['error_message']}")

            except Exception as e:
                print(f"  [验证异常] {e}")
                import traceback
                traceback.print_exc()

        # 5. 汇总结果
        print("\n" + "=" * 60)
        print("测试结果汇总:")
        print("=" * 60)

        successful_tests = [r for r in results if r['result']['success']]
        failed_tests = [r for r in results if not r['result']['success']]

        print(f"总测试对数: {len(results)}")
        print(f"成功验证: {len(successful_tests)}")
        print(f"失败验证: {len(failed_tests)}")
        print(f"成功率: {len(successful_tests) / len(results) * 100:.1f}%")

        if successful_tests:
            scores = [r['result']['final_scores']['total_score'] for r in successful_tests]
            print(f"\n得分统计:")
            print(f"  平均得分: {np.mean(scores):.3f}")
            print(f"  最高得分: {np.max(scores):.3f}")
            print(f"  最低得分: {np.min(scores):.3f}")
            print(f"  得分标准差: {np.std(scores):.3f}")

        # 6. 保存测试结果到独立文件
        print("\n6. 保存测试结果...")
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
        # 创建独立的结果文件
        test_result = {
            'test_info': {
                'timestamp': timestamp,
                'test_pairs': test_pairs,
                'total_tests': len(results),
                'successful_tests': len(successful_tests),
                'failed_tests': len(failed_tests)
            },
            'detailed_results': results
        }
                        
        # 保存到 results/boundary_validation 目录
        output_dir = project_root / "results" / "boundary_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"boundary_validation_test_{timestamp}.json"
                
        # 使用 validator 的序列化方法确保 numpy 数组可以正确保存
        validator_instance = BoundaryValidator()
        serializable_result = validator_instance._make_serializable(test_result)
                        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
                        
        print(f"  [保存成功] 测试结果已保存到：{output_file}")
                        
        # 同时保存最新的结果文件（保持向后兼容）
        latest_file = project_root / "results" / "boundary_validation_test.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        print(f"  [更新成功] 最新结果已更新：{latest_file}")

        return True

    except Exception as e:
        print(f"[测试过程中出现严重错误]: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_module_tests():
    """快速测试各个模块的基本功能"""
    print("\n" + "=" * 60)
    print("模块功能快速测试")
    print("=" * 60)

    config = get_config()

    # 测试配置加载
    print("[模块功能快速测试]")

    # 测试边界提取器（基本初始化）
    try:
        from src.boundary_validation.boundary_extractor import BoundaryExtractor
        extractor = BoundaryExtractor(config.BOUNDARY_EXTRACTION)
        print("[边界提取器初始化成功]")
    except Exception as e:
            print(f"  [初始化失败] 边界提取器：{e}")
            return False

    # 测试特征匹配器
    try:
        from src.boundary_validation.feature_matcher import FeatureMatcher
        matcher = FeatureMatcher(config.FEATURE_MATCHING)
        print("[特征匹配器初始化成功]")
    except Exception as e:
            print(f"  [初始化失败] 特征匹配器：{e}")
            return False

    # 测试互补性检查器
    try:
        from src.boundary_validation.complementarity_checker import ComplementarityChecker
        checker = ComplementarityChecker(config.COMPLEMENTARITY_CHECK)
        print("[互补性检查器初始化成功]")
    except Exception as e:
            print(f"  [初始化失败] 互补性检查器：{e}")
            return False

    # 测试局部对齐器
    try:
        from src.boundary_validation.local_aligner import LocalAligner
        aligner = LocalAligner(config.LOCAL_ALIGNMENT)
        print("[局部对齐器初始化成功]")
    except Exception as e:
            print(f"  [初始化失败] 局部对齐器：{e}")
            return False

    # 测试碰撞检测器
    try:
        from src.boundary_validation.collision_detector import CollisionDetector
        detector = CollisionDetector(config.COLLISION_DETECTION)
        print("[碰撞检测器初始化成功]")
    except Exception as e:
            print(f"  [初始化失败] 碰撞检测器：{e}")
            return False

    # 测试评分系统
    try:
        from src.boundary_validation.scoring_system import ScoringSystem
        scorer = ScoringSystem(config.FINAL_SCORING)
        print("[评分系统初始化成功]")
    except Exception as e:
            print(f"  [初始化失败] 评分系统：{e}")
            return False

    print("\n[所有模块初始化测试通过!]")
    return True


if __name__ == "__main__":
    print("开始边界验证功能测试...")

    # 首先进行模块初始化测试
    if not quick_module_tests():
        print("模块初始化测试失败，退出测试")
        sys.exit(1)

    # 然后进行完整的流水线测试
    success = test_boundary_validation_pipeline()

    if success:
        print(f"[边界验证测试完成!]")
        sys.exit(0)
    else:
        print(f"[边界验证测试失败!]")
        sys.exit(1)