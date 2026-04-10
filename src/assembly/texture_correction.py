"""
纹样辅助全局校正模块
使用 2D SuperGlue/纹样 embedding 匹配结果进行轻量旋转平移修正
"""

import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TextureMatchConstraint:
    """纹样匹配约束"""
    fragment1_id: int
    fragment2_id: int
    texture_similarity: float  # 纹样相似度
    correspondence_points_2d: np.ndarray  # 2D 对应点 (N x 2)
    confidence: float  # 置信度


@dataclass
class TextureCorrectionResult:
    """纹样校正结果"""
    fragment_id: int
    rotation_delta: np.ndarray  # 旋转变换增量
    translation_delta: np.ndarray  # 平移变换增量
    correction_confidence: float  # 校正置信度


class TextureAssistedCorrector:
    """纹样辅助校正器"""
    
    def __init__(self, config: dict = None):
        """
        初始化校正器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.min_texture_similarity = self.config.get('min_texture_similarity', 0.6)
        self.correction_weight = self.config.get('correction_weight', 0.3)  # 校正权重
        
    def extract_texture_features(self, fragment: any) -> Optional[np.ndarray]:
        """
        提取碎片纹样特征
        
        Args:
            fragment: 碎片
            
        Returns:
            np.ndarray: 纹样特征
        """
        # 优先使用 texture_embedding（如果存在）
        if hasattr(fragment, 'texture_embedding') and fragment.texture_embedding is not None:
            return fragment.texture_embedding
        
        # 退而求其次，使用 geo_embedding（几何特征也可以用于辅助校正）
        if hasattr(fragment, 'geo_embedding') and fragment.geo_embedding is not None:
            print(f"[纹样校正] 碎片{fragment.id} 使用 geo_embedding 作为替代")
            return fragment.geo_embedding
        
        # 如果没有预计算的 embedding，尝试从图像提取
        if hasattr(fragment, 'texture_image'):
            # 这里应该调用纹样特征提取模块
            print(f"[纹样校正] 碎片{fragment.id} 需要提取纹样特征")
            return None
        
        return None
    
    def find_texture_matches(self, fragments: List[any], 
                            threshold: float = 0.6) -> List[TextureMatchConstraint]:
        """
        查找纹样匹配的碎片对
        
        Args:
            fragments: 碎片列表
            threshold: 匹配置信度阈值
            
        Returns:
            List[TextureMatchConstraint]: 纹样匹配约束列表
        """
        constraints = []
        
        # 提取所有碎片的纹样特征
        texture_features = {}
        for fragment in fragments:
            feat = self.extract_texture_features(fragment)
            if feat is not None:
                texture_features[fragment.id] = feat
        
        # 计算纹样相似度
        frag_ids = list(texture_features.keys())
        n = len(frag_ids)
        
        for i in range(n):
            id1 = frag_ids[i]
            feat1 = texture_features[id1]
            
            for j in range(i + 1, n):
                id2 = frag_ids[j]
                feat2 = texture_features[id2]
                
                # 计算余弦相似度
                similarity = np.dot(feat1.flatten(), feat2.flatten()) / (
                    np.linalg.norm(feat1.flatten()) * np.linalg.norm(feat2.flatten()) + 1e-8
                )
                
                if similarity >= threshold:
                    constraint = TextureMatchConstraint(
                        fragment1_id=id1,
                        fragment2_id=id2,
                        texture_similarity=similarity,
                        correspondence_points_2d=np.array([]),  # 这里应该有实际的对应点
                        confidence=similarity
                    )
                    constraints.append(constraint)
                    
                    print(f"[纹样匹配] 发现匹配：碎片{id1} - 碎片{id2}, "
                          f"相似度：{similarity:.4f}")
        
        return constraints
    
    def compute_correction(self, fragment1: any, fragment2: any,
                          constraint: TextureMatchConstraint,
                          current_pose1: np.ndarray, current_pose2: np.ndarray) -> Tuple[TextureCorrectionResult, TextureCorrectionResult]:
        """
        计算纹样辅助的位姿校正
        
        Args:
            fragment1: 碎片 1
            fragment2: 碎片 2
            constraint: 纹样匹配约束
            current_pose1: 碎片 1 当前位姿 (4x4)
            current_pose2: 碎片 2 当前位姿 (4x4)
            
        Returns:
            Tuple[TextureCorrectionResult, TextureCorrectionResult]: 两个碎片的校正量
        """
        # 基于纹样相似度计算校正权重
        weight = self.correction_weight * constraint.confidence
        
        # 简化的校正策略：微调旋转和平移
        # 实际应用中应该根据 2D 对应点计算最优的单应性变换
        
        # 计算小的旋转增量（绕法向量旋转）
        rotation_axis = np.array([0, 0, 1])  # 假设主要绕 Z 轴调整
        rotation_angle = (1.0 - constraint.texture_similarity) * 5.0 * np.pi / 180.0  # 小角度
        
        R_delta = self._axis_angle_to_rotation(rotation_axis, rotation_angle)
        t_delta = np.zeros(3)  # 暂时不进行平移校正
        
        # 创建校正结果
        result1 = TextureCorrectionResult(
            fragment_id=fragment1.id,
            rotation_delta=R_delta,
            translation_delta=t_delta,
            correction_confidence=constraint.confidence
        )
        
        result2 = TextureCorrectionResult(
            fragment_id=fragment2.id,
            rotation_delta=R_delta.T,  # 反向旋转
            translation_delta=-t_delta,
            correction_confidence=constraint.confidence
        )
        
        return result1, result2
    
    def apply_corrections(self, poses: Dict[int, np.ndarray],
                         corrections: List[Tuple[TextureCorrectionResult, TextureCorrectionResult]]) -> Dict[int, np.ndarray]:
        """
        应用所有校正
        
        Args:
            poses: 原始位姿字典
            corrections: 校正列表
            
        Returns:
            Dict[int, np.ndarray]: 校正后的位姿
        """
        corrected_poses = {k: v.copy() for k, v in poses.items()}
        
        for result1, result2 in corrections:
            # 应用碎片 1 的校正
            if result1.fragment_id in corrected_poses:
                T_current = corrected_poses[result1.fragment_id]
                T_correction = np.eye(4)
                T_correction[:3, :3] = result1.rotation_delta
                T_correction[:3, 3] = result1.translation_delta
                
                corrected_poses[result1.fragment_id] = T_correction @ T_current
            
            # 应用碎片 2 的校正
            if result2.fragment_id in corrected_poses:
                T_current = corrected_poses[result2.fragment_id]
                T_correction = np.eye(4)
                T_correction[:3, :3] = result2.rotation_delta
                T_correction[:3, 3] = result2.translation_delta
                
                corrected_poses[result2.fragment_id] = T_correction @ T_current
        
        return corrected_poses
    
    def _axis_angle_to_rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        轴角转旋转矩阵（Rodrigues 公式）
        
        Args:
            axis: 旋转轴 (3,)
            angle: 旋转角度（弧度）
            
        Returns:
            np.ndarray: 3x3 旋转矩阵
        """
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R


def texture_assisted_correction(fragments: List[any],
                               initial_poses: Dict[int, np.ndarray],
                               config: dict = None) -> Dict[int, np.ndarray]:
    """
    执行纹样辅助的全局校正
    
    Args:
        fragments: 碎片列表
        initial_poses: 初始位姿（numpy 矩阵或 FragmentPose）
        config: 配置参数
        
    Returns:
        Dict[int, np.ndarray]: 校正后的位姿（numpy 矩阵）
    """
    print("[纹样校正] 开始纹样辅助全局校正...")
    
    # 将 initial_poses 转换为 numpy 矩阵格式
    poses_numpy = {}
    for frag_id, pose in initial_poses.items():
        if hasattr(pose, 'get_transformation'):
            poses_numpy[frag_id] = pose.get_transformation()
        else:
            poses_numpy[frag_id] = np.array(pose) if isinstance(pose, list) else pose
    
    corrector = TextureAssistedCorrector(config=config)
    
    # 1. 查找纹样匹配
    constraints = corrector.find_texture_matches(fragments)
    
    if not constraints:
        print("[纹样校正] 未找到足够的纹样匹配，跳过校正")
        return poses_numpy
    
    # 2. 计算校正量
    all_corrections = []
    
    for constraint in constraints:
        # 获取碎片
        frag1 = next((f for f in fragments if f.id == constraint.fragment1_id), None)
        frag2 = next((f for f in fragments if f.id == constraint.fragment2_id), None)
        
        if frag1 is None or frag2 is None:
            continue
        
        # 获取当前位姿
        pose1 = initial_poses.get(constraint.fragment1_id, np.eye(4))
        pose2 = initial_poses.get(constraint.fragment2_id, np.eye(4))
        
        # 计算校正
        result1, result2 = corrector.compute_correction(frag1, frag2, constraint, pose1, pose2)
        all_corrections.append((result1, result2))
    
    # 3. 应用校正
    corrected_poses = corrector.apply_corrections(initial_poses, all_corrections)
    
    print(f"[纹样校正] 完成，应用了{len(all_corrections)}个纹样约束")
    
    return corrected_poses
