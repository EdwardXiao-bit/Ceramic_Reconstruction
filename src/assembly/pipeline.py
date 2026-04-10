"""
全局拼接主流程
整合所有模块实现完整的陶瓷碎片全局拼接
"""

import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime


class GlobalAssemblyPipeline:
    """全局拼接流水线"""
    
    def __init__(self, config: dict = None):
        """
        初始化全局拼接流水线
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 导入所需模块
        from .global_assembly import (
            FragmentMatchingGraph, MatchEdge, 
            propagate_poses, build_matching_graph_from_validation_results,
            visualize_assembly_result
        )
        from .pose_graph_optimizer import (
            PoseGraphOptimizer, optimize_global_poses
        )
        from .collision_detector import (
            check_global_collisions, visualize_collision_areas
        )
        from .texture_correction import texture_assisted_correction
        
        self.FragmentMatchingGraph = FragmentMatchingGraph
        self.MatchEdge = MatchEdge
        self.propagate_poses = propagate_poses
        self.build_matching_graph = build_matching_graph_from_validation_results
        self.visualize_assembly = visualize_assembly_result
        
        self.PoseGraphOptimizer = PoseGraphOptimizer
        self.optimize_global_poses = optimize_global_poses
        
        self.check_global_collisions = check_global_collisions
        self.visualize_collision = visualize_collision_areas
        
        self.texture_assisted_correction = texture_assisted_correction

    def _extract_transformation_from_result(vr: dict) -> np.ndarray:
        """
        从验证结果中提取最佳变换矩阵
        优先级：局部对齐变换 > 特征匹配变换 > 单位矩阵
        """
        # 优先用ICP精化后的变换
        intermediate = vr.get('intermediate_results', {})

        local_align = intermediate.get('local_alignment', {})
        if local_align:
            T_local = local_align.get('refined_transformation')
            if T_local is not None:
                T = np.array(T_local)
                if T.shape == (4, 4) and not np.allclose(T, np.eye(4)):
                    return T

        # 次选：特征匹配的初始变换
        feat_match = intermediate.get('feature_matching', {})
        if feat_match:
            T_feat = feat_match.get('initial_transformation')
            if T_feat is not None:
                T = np.array(T_feat)
                if T.shape == (4, 4) and not np.allclose(T, np.eye(4)):
                    return T

        return np.eye(4)

    def run(self, fragments: List[Any], 
           validation_results: List[Dict]) -> Dict[str, Any]:
        """
        运行完整的全局拼接流程
        
        Args:
            fragments: 碎片列表
            validation_results: 边界验证结果列表
            
        Returns:
            Dict[str, Any]: 拼接结果（包含位姿、统计信息等）
        """
        print("\n" + "=" * 60)
        print("全局拼接流水线开始")
        print("=" * 60)
        
        start_time = datetime.now()
        result = {
            'success': False,
            'fragments_count': len(fragments),
            'validation_pairs_count': len(validation_results),
            'poses': {},
            'statistics': {},
            'logs': []
        }
        
        try:
            # 步骤 1: 构建碎片匹配图
            print("\n【步骤 1/6】构建碎片匹配图...")
            log_step1 = self._build_matching_graph(fragments, validation_results)
            result['logs'].append(log_step1)
            
            graph = self.FragmentMatchingGraph()
            for frag in fragments:
                graph.add_fragment(frag.id, frag)
            
            valid_pairs = 0
            for vr in validation_results:
                if not vr.get('success', False):
                    continue

                pair = vr.get('pair', [])
                if len(pair) < 2:
                    continue
                f1_id, f2_id = pair[0], pair[1]

                final_scores = vr.get('final_scores', {})
                total_score = final_scores.get('total_score', 0.0)

                # MVP阶段：接受所有成功验证的对（不过滤低分）
                # 只过滤明显无效（total_score < 0.05）
                if total_score < 0.05:
                    continue

                # 提取真实变换矩阵
                transformation = _extract_transformation_from_result(vr)

                edge = self.MatchEdge(
                    fragment1_id=f1_id,
                    fragment2_id=f2_id,
                    weight=max(total_score, 0.1),  # 保证权重不为0
                    transformation=transformation,
                    confidence=max(total_score, 0.1),
                    dcp_residual=vr.get('intermediate_results', {}).get('local_alignment', {}).get('rmse', 0.0)
                )
                graph.add_match_edge(edge)
                valid_pairs += 1
            
            result['logs'].append(f"添加{valid_pairs}个有效匹配边")
            print(f"✓ 构建完成：{len(graph.graph.nodes)}个节点，{valid_pairs}条边")
            
            # 步骤 2: 连通子图分析
            print("\n【步骤 2/6】连通子图分析...")
            components = graph.get_connected_components()
            print(f"✓ 发现{len(components)}个连通子图")
            
            for i, comp in enumerate(components):
                print(f"  子图{i+1}: {len(comp)}个碎片 - {comp}")
            
            result['logs'].append(f"发现{len(components)}个连通子图")
            
            # 步骤 3: 初始位姿传播
            print("\n【步骤 3/6】初始位姿传播...")
            poses = self.propagate_poses(graph)
            print(f"✓ 位姿传播完成：{len(poses)}个碎片获得位姿")
            
            result['logs'].append(f"位姿传播完成：{len(poses)}个碎片")
            
            # 步骤 4: Pose Graph 优化
            print("\n【步骤 4/6】Pose Graph 优化...")
            use_g2o = self.config.get('use_g2o', False)
            use_ceres = self.config.get('use_ceres', False)
            max_iterations = self.config.get('max_iterations', 100)
            
            optimized_poses = self.optimize_global_poses(
                graph, 
                max_iterations=max_iterations,
                use_g2o=use_g2o,
                use_ceres=use_ceres
            )
            
            print(f"✓ Pose Graph 优化完成")
            result['logs'].append(f"Pose Graph 优化完成（{max_iterations}次迭代）")
            
            # 步骤 5: 纹样辅助校正（如果可用）
            print("\n【步骤 5/6】纹样辅助校正...")
            enable_texture = self.config.get('enable_texture_correction', True)
            
            if enable_texture:
                corrected_poses = self.texture_assisted_correction(
                    fragments, 
                    optimized_poses,
                    config=self.config.get('texture_correction', {})
                )
                print(f"✓ 纹样校正完成")
                result['logs'].append("纹样辅助校正完成")
            else:
                corrected_poses = optimized_poses
                print(f"✓ 跳过纹样校正")
                result['logs'].append("跳过纹样校正")
            
            # 步骤 6: 冲突穿透检测
            print("\n【步骤 6/6】冲突穿透检测...")
            # 将 FragmentPose 转换为 numpy 矩阵
            poses_numpy = {}
            for frag_id, frag_pose in corrected_poses.items():
                if hasattr(frag_pose, 'get_transformation'):
                    poses_numpy[frag_id] = frag_pose.get_transformation()
                else:
                    poses_numpy[frag_id] = np.array(frag_pose) if isinstance(frag_pose, list) else frag_pose
            
            collision_result = self.check_global_collisions(
                fragments,
                poses_numpy,
                method=self.config.get('collision_method', 'voxel'),
                voxel_size=self.config.get('voxel_size', 0.01)
            )
            
            if collision_result.has_collision:
                print(f"⚠ 检测到{len(collision_result.collision_pairs)}个碰撞")
                result['logs'].append(f"警告：检测到{len(collision_result.collision_pairs)}个碰撞")
            else:
                print(f"✓ 无碰撞冲突")
                result['logs'].append("无碰撞冲突")
            
            # 整理结果（corrected_poses 已经是 numpy 矩阵）
            result['success'] = True
            result['poses'] = {
                frag_id: pose.tolist() if hasattr(pose, 'tolist') else pose
                for frag_id, pose in corrected_poses.items()
            }
            result['statistics'] = {
                'connected_components': len(components),
                'matched_edges': valid_pairs,
                'optimized_fragments': len(corrected_poses),
                'collision_pairs': len(collision_result.collision_pairs),
                'total_collision_volume': collision_result.total_collision_volume,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
            print("\n" + "=" * 60)
            print("全局拼接完成!")
            print("=" * 60)
            print(f"处理时间：{result['statistics']['processing_time']:.2f}秒")
            print(f"成功拼接：{result['statistics']['optimized_fragments']}个碎片")
            
        except Exception as e:
            print(f"\n❌ 全局拼接失败：{e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)
            result['logs'].append(f"错误：{e}")
        
        return result

    def _build_matching_graph(self, fragments: List[Any], 
                             validation_results: List[Dict]) -> str:
        """构建匹配图的辅助函数"""
        return f"开始构建匹配图：{len(fragments)}个碎片，{len(validation_results)}个验证对"
    
    def visualize_result(self, fragments: List[Any], 
                        poses: Dict[int, np.ndarray],
                        show_collision: bool = False) -> o3d.geometry.PointCloud:
        """
        可视化全局拼接结果
        
        Args:
            fragments: 碎片列表
            poses: 碎片位姿字典
            show_collision: 是否显示碰撞区域
            
        Returns:
            o3d.geometry.PointCloud: 组合点云
        """
        # 转换 poses 到 FragmentPose 格式
        from .global_assembly import FragmentPose
        
        fragment_poses = {}
        for frag_id, T in poses.items():
            if isinstance(T, list):
                T = np.array(T)
            
            frag_pose = FragmentPose(fragment_id=frag_id)
            frag_pose.rotation = T[:3, :3]
            frag_pose.translation = T[:3, 3]
            fragment_poses[frag_id] = frag_pose
        
        # 可视化装配结果（直接传入 numpy 矩阵）
        from .global_assembly import visualize_assembly_result
        combined_pcd = visualize_assembly_result(fragments, fragment_poses)
        
        if show_collision:
            # 这里需要重新计算碰撞
            print("[可视化] 碰撞区域显示功能待实现")
        
        return combined_pcd
    
    def save_result(self, result: Dict[str, Any], output_path: str):
        """
        保存拼接结果
        
        Args:
            result: 拼接结果字典
            output_path: 输出路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"[保存结果] 已保存到：{output_path}")


def global_assembly_pipeline(fragments: List[Any],
                            validation_results: List[Dict],
                            config: dict = None) -> Dict[str, Any]:
    """
    全局拼接流程的便捷函数
    
    Args:
        fragments: 碎片列表
        validation_results: 边界验证结果列表
        config: 配置参数
        
    Returns:
        Dict[str, Any]: 拼接结果
    """
    pipeline = GlobalAssemblyPipeline(config=config)
    return pipeline.run(fragments, validation_results)
