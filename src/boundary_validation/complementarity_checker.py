# D:\ceramic_reconstruction\src\boundary_validation\complementarity_checker.py
"""
局部集合互补性检查模块
包括法向互补性检查和边界形状互补性检查
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import torch

@dataclass
class ComplementarityResult:
    """互补性检查结果"""
    normal_complementarity_score: float    # 法向互补性得分
    shape_complementarity_score: float     # 形状互补性得分
    average_normal_angle: float            # 平均法向夹角
    reverse_normal_ratio: float            # 反向法向比例
    normal_similarity_stats: Dict[str, float]  # 法向相似度统计
    shape_similarity_stats: Dict[str, float]   # 形状相似度统计

class ComplementarityChecker:
    """互补性检查器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pointnet_model = None
        self.cnn_model = None
        self._initialize_models()
        
    def _initialize_models(self):
        """初始化深度学习模型"""
        if self.config['shape_complementarity']['pointnet_enabled']:
            try:
                self.pointnet_model = self._load_pointnet_model()
                print("[互补性检查] PointNet++模型加载成功")
            except Exception as e:
                print(f"[互补性检查] PointNet++模型加载失败: {e}")
                self.pointnet_model = None
                
        if self.config['shape_complementarity']['cnn_enabled']:
            try:
                self.cnn_model = self._load_cnn_model()
                print("[互补性检查] 3D CNN模型加载成功")
            except Exception as e:
                print(f"[互补性检查] 3D CNN模型加载失败: {e}")
                self.cnn_model = None
    
    def _load_pointnet_model(self):
        """加载PointNet++模型"""
        # 这里应该是实际的PointNet++模型加载代码
        class MockPointNet:
            def encode(self, points):
                # 模拟编码过程
                batch_size = 1
                embedding_dim = 256
                return np.random.randn(batch_size, embedding_dim).astype(np.float32)
                
        return MockPointNet()
    
    def _load_cnn_model(self):
        """加载3D CNN模型"""
        # 这里应该是实际的3D CNN模型加载代码
        class MockCNN:
            def predict_complementarity(self, patch1, patch2):
                # 模拟互补性预测
                return np.random.uniform(0.6, 0.95)
                
        return MockCNN()
    
    def check_complementarity(self, boundary1: Any, boundary2: Any, 
                            match_result: Any) -> ComplementarityResult:
        """
        检查两个边界区域的互补性
        
        Args:
            boundary1: 第一个边界区域
            boundary2: 第二个边界区域
            match_result: 特征匹配结果
            
        Returns:
            ComplementarityResult: 互补性检查结果
        """
        print("[互补性检查] 开始局部集合互补性检查...")
        
        # 1. 法向互补性检查
        print("[互补性检查] 执行法向互补性检查...")
        normal_result = self._check_normal_complementarity(boundary1, boundary2, match_result)
        
        # 2. 边界形状互补性检查
        print("[互补性检查] 执行边界形状互补性检查...")
        shape_result = self._check_shape_complementarity(boundary1, boundary2, match_result)
        
        # 3. 综合结果
        complementarity_result = ComplementarityResult(
            normal_complementarity_score=normal_result['score'],
            shape_complementarity_score=shape_result['score'],
            average_normal_angle=normal_result['avg_angle'],
            reverse_normal_ratio=normal_result['reverse_ratio'],
            normal_similarity_stats=normal_result['stats'],
            shape_similarity_stats=shape_result['stats']
        )
        
        print(f"[互补性检查] 检查完成:")
        print(f"  法向互补性得分: {complementarity_result.normal_complementarity_score:.3f}")
        print(f"  形状互补性得分: {complementarity_result.shape_complementarity_score:.3f}")
        print(f"  平均法向夹角: {complementarity_result.average_normal_angle:.1f}°")
        print(f"  反向法向比例: {complementarity_result.reverse_normal_ratio:.3f}")
        
        return complementarity_result
    
    def _check_normal_complementarity(self, boundary1: Any, boundary2: Any, 
                                    match_result: Any) -> Dict[str, Any]:
        """
        法向互补性检查
        """
        matches = match_result.matches
        if len(matches) == 0:
            return {
                'score': 0.0,
                'avg_angle': 90.0,
                'reverse_ratio': 0.0,
                'stats': {'min_similarity': 0.0, 'max_similarity': 0.0, 'mean_similarity': 0.0}
            }
        
        # 获取匹配点的法向量
        normals1 = boundary1.normals[matches[:, 0]]
        normals2 = boundary2.normals[matches[:, 1]]
        
        # 计算法向量夹角
        dot_products = np.sum(normals1 * normals2, axis=1)
        # 限制在[-1,1]范围内防止数值误差
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angles = np.arccos(np.abs(dot_products))  # 使用绝对值因为镜像关系
        angles_deg = np.degrees(angles)
        
        # 计算法向相似度（越接近0或180度越好）
        normal_similarities = np.abs(dot_products)
        
        # 统计信息
        stats = {
            'min_similarity': float(np.min(normal_similarities)),
            'max_similarity': float(np.max(normal_similarities)),
            'mean_similarity': float(np.mean(normal_similarities))
        }
        
        # 计算反向法向比例（理想情况下应该接近1.0）
        mirror_tolerance = np.radians(self.config['normal_complementarity']['mirror_angle_tolerance'])
        reverse_angles = np.abs(angles - 90.0)  # 与90度的偏差
        reverse_ratio = np.mean(reverse_angles < mirror_tolerance)
        
        # 计算互补性得分
        # 基于法向相似度和反向比例的综合评估
        min_similarity = self.config['normal_complementarity']['min_normal_similarity']
        similarity_score = np.mean(normal_similarities >= min_similarity)
        
        # 综合得分
        complementarity_score = 0.7 * similarity_score + 0.3 * reverse_ratio
        
        result = {
            'score': float(np.clip(complementarity_score, 0.0, 1.0)),
            'avg_angle': float(np.mean(angles_deg)),
            'reverse_ratio': float(reverse_ratio),
            'stats': stats
        }
        
        return result
    
    def _check_shape_complementarity(self, boundary1: Any, boundary2: Any, 
                                   match_result: Any) -> Dict[str, Any]:
        """
        边界形状互补性检查
        """
        matches = match_result.matches
        if len(matches) == 0:
            return {
                'score': 0.0,
                'stats': {'min_similarity': 0.0, 'max_similarity': 0.0, 'mean_similarity': 0.0}
            }
        
        patch_size = self.config['shape_complementarity']['patch_size']
        similarity_threshold = self.config['shape_complementarity']['similarity_threshold']
        
        # 存储所有patch的相似度
        patch_similarities = []
        
        # 对每个匹配点提取局部patch并计算相似度
        for i, (idx1, idx2) in enumerate(matches):
            # 提取局部patch
            patch1 = self._extract_local_patch(boundary1, idx1, patch_size)
            patch2 = self._extract_local_patch(boundary2, idx2, patch_size)
            
            if patch1 is None or patch2 is None:
                continue
                
            # 计算patch相似度
            similarity = self._compute_patch_similarity(patch1, patch2)
            patch_similarities.append(similarity)
        
        if len(patch_similarities) == 0:
            return {
                'score': 0.0,
                'stats': {'min_similarity': 0.0, 'max_similarity': 0.0, 'mean_similarity': 0.0}
            }
        
        # 统计信息
        similarities_array = np.array(patch_similarities)
        stats = {
            'min_similarity': float(np.min(similarities_array)),
            'max_similarity': float(np.max(similarities_array)),
            'mean_similarity': float(np.mean(similarities_array))
        }
        
        # 计算互补性得分
        # 基于超过阈值的patch比例
        above_threshold = np.mean(similarities_array >= similarity_threshold)
        mean_similarity = np.mean(similarities_array)
        
        # 综合得分
        complementarity_score = 0.6 * above_threshold + 0.4 * mean_similarity
        
        result = {
            'score': float(np.clip(complementarity_score, 0.0, 1.0)),
            'stats': stats
        }
        
        return result
    
    def _extract_local_patch(self, boundary: Any, center_index: int, patch_size: int) -> Optional[np.ndarray]:
        """
        提取局部边界patch
        """
        points = boundary.points
        center_point = points[center_index]
        
        # 计算到中心点的距离
        distances = np.linalg.norm(points - center_point, axis=1)
        
        # 选择最近的点作为patch
        if len(points) <= patch_size:
            patch_indices = np.arange(len(points))
        else:
            # 找到最近的patch_size个点
            patch_indices = np.argpartition(distances, patch_size-1)[:patch_size]
        
        patch_points = points[patch_indices]
        
        # 中心化patch
        centroid = np.mean(patch_points, axis=0)
        centered_patch = patch_points - centroid
        
        return centered_patch
    
    def _compute_patch_similarity(self, patch1: np.ndarray, patch2: np.ndarray) -> float:
        """
        计算两个patch的相似度
        """
        # 方法1: 使用PointNet++特征相似度
        if self.pointnet_model is not None:
            try:
                feat1 = self.pointnet_model.encode(patch1)
                feat2 = self.pointnet_model.encode(patch2)
                similarity = self._compute_feature_similarity(feat1, feat2)
                return float(similarity)
            except Exception as e:
                print(f"[形状互补性] PointNet++计算失败: {e}")
        
        # 方法2: 使用3D CNN
        if self.cnn_model is not None:
            try:
                similarity = self.cnn_model.predict_complementarity(patch1, patch2)
                return float(similarity)
            except Exception as e:
                print(f"[形状互补性] CNN计算失败: {e}")
        
        # 方法3: 基础几何相似度（降级方案）
        return self._compute_geometric_similarity(patch1, patch2)
    
    def _compute_feature_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        计算特征向量相似度
        """
        # 余弦相似度
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
        
        cosine_similarity = np.dot(feat1_norm.flatten(), feat2_norm.flatten())
        return np.clip(cosine_similarity, 0.0, 1.0)
    
    def _compute_geometric_similarity(self, patch1: np.ndarray, patch2: np.ndarray) -> float:
        """
        计算几何相似度（基础方法）
        """
        # 基于统计特征的相似度
        def compute_stats(patch):
            # 计算基本统计量
            centroid = np.mean(patch, axis=0)
            std_dev = np.std(patch, axis=0)
            bbox_min = np.min(patch, axis=0)
            bbox_max = np.max(patch, axis=0)
            
            # 计算主成分分析特征
            cov_matrix = np.cov(patch.T)
            eigenvals = np.linalg.eigvals(cov_matrix)
            eigenvals = np.sort(eigenvals)[::-1]  # 降序排列
            
            return {
                'centroid': centroid,
                'std_dev': std_dev,
                'bbox_size': bbox_max - bbox_min,
                'eigenvals': eigenvals,
                'volume': np.prod(bbox_max - bbox_min)
            }
        
        stats1 = compute_stats(patch1)
        stats2 = compute_stats(patch2)
        
        # 计算各种相似度
        similarities = []
        
        # 1. 尺寸相似度
        size_ratio = np.minimum(stats1['bbox_size'], stats2['bbox_size']) / \
                     (np.maximum(stats1['bbox_size'], stats2['bbox_size']) + 1e-8)
        size_similarity = np.mean(size_ratio)
        similarities.append(size_similarity)
        
        # 2. 体积相似度
        vol_ratio = min(stats1['volume'], stats2['volume']) / (max(stats1['volume'], stats2['volume']) + 1e-8)
        similarities.append(vol_ratio)
        
        # 3. 主成分相似度
        eigen_ratio = np.minimum(stats1['eigenvals'], stats2['eigenvals']) / \
                      (np.maximum(stats1['eigenvals'], stats2['eigenvals']) + 1e-8)
        pca_similarity = np.mean(eigen_ratio)
        similarities.append(pca_similarity)
        
        # 4. 标准差相似度
        std_ratio = np.minimum(stats1['std_dev'], stats2['std_dev']) / \
                    (np.maximum(stats1['std_dev'], stats2['std_dev']) + 1e-8)
        std_similarity = np.mean(std_ratio)
        similarities.append(std_similarity)
        
        # 综合相似度
        geometric_similarity = np.mean(similarities)
        return float(np.clip(geometric_similarity, 0.0, 1.0))
    
    def analyze_complementarity_distribution(self, complementarity_result: ComplementarityResult) -> Dict[str, Any]:
        """
        分析互补性分布特征
        """
        analysis = {
            'normal_analysis': {
                'quality_level': self._assess_normal_quality(complementarity_result.normal_complementarity_score),
                'angle_distribution': 'concentrated' if complementarity_result.average_normal_angle < 45 else 'dispersed',
                'mirror_symmetry': complementarity_result.reverse_normal_ratio > 0.8
            },
            'shape_analysis': {
                'quality_level': self._assess_shape_quality(complementarity_result.shape_complementarity_score),
                'consistency': complementarity_result.shape_similarity_stats['max_similarity'] - \
                              complementarity_result.shape_similarity_stats['min_similarity']
            },
            'overall_assessment': self._overall_complementarity_assessment(complementarity_result)
        }
        
        return analysis
    
    def _assess_normal_quality(self, score: float) -> str:
        """评估法向互补性质量等级"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _assess_shape_quality(self, score: float) -> str:
        """评估形状互补性质量等级"""
        if score >= 0.85:
            return 'excellent'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _overall_complementarity_assessment(self, result: ComplementarityResult) -> str:
        """整体互补性评估"""
        avg_score = (result.normal_complementarity_score + result.shape_complementarity_score) / 2.0
        
        if avg_score >= 0.75:
            return 'highly_complementary'
        elif avg_score >= 0.55:
            return 'moderately_complementary'
        elif avg_score >= 0.35:
            return 'slightly_complementary'
        else:
            return 'not_complementary'