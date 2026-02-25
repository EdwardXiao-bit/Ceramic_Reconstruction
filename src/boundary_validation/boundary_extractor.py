# D:\ceramic_reconstruction\src\boundary_validation\boundary_extractor.py
"""
边界区域提取模块
提取点云中断裂边界(Rim)或边界strip，保留曲率变化大/表面粗糙度高/深度不连续的区域
"""

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class BoundaryRegion:
    """边界区域数据类"""
    points: np.ndarray           # 边界点坐标
    normals: np.ndarray          # 法向量
    curvature: np.ndarray        # 曲率值
    roughness: np.ndarray        # 粗糙度值
    depth_values: np.ndarray     # 深度值
    indices: np.ndarray          # 原始索引
    confidence: float            # 提取置信度

class BoundaryExtractor:
    """边界区域提取器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def extract_boundary_regions(self, fragment1: Any, fragment2: Any) -> Tuple[Optional[BoundaryRegion], Optional[BoundaryRegion]]:
        """
        提取两个碎片的边界区域
        
        Args:
            fragment1: 第一个碎片对象
            fragment2: 第二个碎片对象
            
        Returns:
            tuple: (边界区域1, 边界区域2)
        """
        print("[边界提取] 开始提取碎片边界区域...")
        
        # 提取第一个碎片的边界
        boundary1 = self._extract_single_boundary(fragment1)
        if boundary1 is None:
            print("[边界提取] 碎片1边界提取失败")
            return None, None
            
        # 提取第二个碎片的边界
        boundary2 = self._extract_single_boundary(fragment2)
        if boundary2 is None:
            print("[边界提取] 碎片2边界提取失败")
            return None, None
            
        print(f"[边界提取] 成功提取两个边界区域:")
        print(f"  碎片1: {len(boundary1.points)} 个边界点")
        print(f"  碎片2: {len(boundary2.points)} 个边界点")
        
        return boundary1, boundary2
    
    def _extract_single_boundary(self, fragment: Any) -> Optional[BoundaryRegion]:
        """
        提取单个碎片的边界区域
        """
        # 获取点云数据
        if hasattr(fragment, 'point_cloud') and fragment.point_cloud is not None:
            pcd = fragment.point_cloud
        elif hasattr(fragment, 'mesh') and fragment.mesh is not None:
            pcd = fragment.mesh.sample_points_uniformly(number_of_points=10000)
        else:
            print(f"[边界提取] 碎片{getattr(fragment, 'id', 'unknown')}缺少点云数据")
            return None
            
        points = np.asarray(pcd.points)
        
        # 1. 计算法向量和曲率
        print("[边界提取] 计算法向量和曲率...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        normals = np.asarray(pcd.normals)
        
        # 计算曲率（基于法向量变化）
        curvature = self._compute_curvature(points, normals)
        
        # 2. 计算表面粗糙度
        print("[边界提取] 计算表面粗糙度...")
        roughness = self._compute_roughness(points, k_neighbors=30)
        
        # 3. 计算深度值（相对于重心的距离）
        print("[边界提取] 计算深度不连续性...")
        depth_values = self._compute_depth_discontinuity(points)
        
        # 4. 综合判断疑似断裂面区域
        print("[边界提取] 识别疑似断裂面区域...")
        boundary_mask = self._identify_suspicious_boundary(
            curvature=curvature,
            roughness=roughness, 
            depth_values=depth_values
        )
        
        if np.sum(boundary_mask) < self.config['min_boundary_points']:
            print(f"[边界提取] 边界点数不足: {np.sum(boundary_mask)} < {self.config['min_boundary_points']}")
            return None
            
        # 5. 聚类去噪
        print("[边界提取] 聚类去噪...")
        clustered_indices = self._cluster_boundary_points(points[boundary_mask])
        
        if len(clustered_indices) == 0:
            print("[边界提取] 聚类后无有效边界点")
            return None
            
        # 6. 构建边界区域对象
        boundary_points = points[boundary_mask][clustered_indices]
        boundary_normals = normals[boundary_mask][clustered_indices]
        boundary_curvature = curvature[boundary_mask][clustered_indices]
        boundary_roughness = roughness[boundary_mask][clustered_indices]
        boundary_depth = depth_values[boundary_mask][clustered_indices]
        original_indices = np.where(boundary_mask)[0][clustered_indices]
        
        # 计算置信度
        confidence = self._calculate_extraction_confidence(
            boundary_curvature, boundary_roughness, boundary_depth
        )
        
        boundary_region = BoundaryRegion(
            points=boundary_points,
            normals=boundary_normals,
            curvature=boundary_curvature,
            roughness=boundary_roughness,
            depth_values=boundary_depth,
            indices=original_indices,
            confidence=confidence
        )
        
        print(f"[边界提取] 提取完成，置信度: {confidence:.3f}")
        return boundary_region
    
    def _compute_curvature(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """
        计算点云曲率（基于法向量变化）
        """
        n_points = len(points)
        curvature = np.zeros(n_points)
        
        # 构建KD树
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        tree = o3d.geometry.KDTreeFlann(pcd)
        
        # 对每个点计算曲率
        for i in range(n_points):
            # 找到k近邻
            k = min(20, n_points - 1)
            _, indices, _ = tree.search_knn_vector_3d(points[i], k)
            
            if len(indices) < 5:
                curvature[i] = 0
                continue
                
            # 计算法向量变化
            neighbor_normals = normals[indices]
            ref_normal = normals[i]
            
            # 计算法向量夹角变化
            angles = []
            for j in range(len(neighbor_normals)):
                if j != 0:  # 不与自己比较
                    dot_product = np.clip(np.dot(ref_normal, neighbor_normals[j]), -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    angles.append(angle)
            
            if angles:
                curvature[i] = np.mean(angles)
            else:
                curvature[i] = 0
                
        return curvature
    
    def _compute_roughness(self, points: np.ndarray, k_neighbors: int = 30) -> np.ndarray:
        """
        计算表面粗糙度（基于局部平面拟合残差）
        """
        n_points = len(points)
        roughness = np.zeros(n_points)
        
        # 构建KD树
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        tree = o3d.geometry.KDTreeFlann(pcd)
        
        for i in range(n_points):
            # 找到k近邻
            k = min(k_neighbors, n_points - 1)
            _, indices, _ = tree.search_knn_vector_3d(points[i], k)
            
            if len(indices) < 5:
                roughness[i] = 0
                continue
                
            # 局部点集
            local_points = points[indices]
            
            # 平面拟合
            centroid = np.mean(local_points, axis=0)
            centered_points = local_points - centroid
            
            # SVD分解计算最佳拟合平面
            try:
                _, _, vt = np.linalg.svd(centered_points)
                normal = vt[-1, :]  # 最小奇异值对应的向量作为法向量
                
                # 计算点到平面的距离（粗糙度）
                distances = np.abs(np.dot(centered_points, normal))
                roughness[i] = np.mean(distances)
                
            except np.linalg.LinAlgError:
                roughness[i] = 0
                
        return roughness
    
    def _compute_depth_discontinuity(self, points: np.ndarray) -> np.ndarray:
        """
        计算深度不连续性（基于点到重心距离的变化）
        """
        # 计算重心
        centroid = np.mean(points, axis=0)
        
        # 计算每个点到重心的距离
        distances = np.linalg.norm(points - centroid, axis=1)
        
        # 计算距离的标准差作为深度不连续性指标
        global_std = np.std(distances)
        
        # 局部深度变化
        n_points = len(points)
        depth_discontinuity = np.zeros(n_points)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        tree = o3d.geometry.KDTreeFlann(pcd)
        
        for i in range(n_points):
            # 找到近邻点
            k = min(20, n_points - 1)
            _, indices, _ = tree.search_knn_vector_3d(points[i], k)
            
            local_distances = distances[indices]
            local_std = np.std(local_distances)
            
            # 相对于全局变化的比例
            if global_std > 1e-8:
                depth_discontinuity[i] = local_std / global_std
            else:
                depth_discontinuity[i] = 0
                
        return depth_discontinuity
    
    def _identify_suspicious_boundary(self, curvature: np.ndarray, roughness: np.ndarray, 
                                    depth_values: np.ndarray) -> np.ndarray:
        """
        识别疑似断裂面区域
        """
        # 标准化特征值到[0,1]范围
        def normalize(arr):
            arr_min, arr_max = np.min(arr), np.max(arr)
            if arr_max - arr_min > 1e-8:
                return (arr - arr_min) / (arr_max - arr_min)
            return np.zeros_like(arr)
            
        norm_curvature = normalize(curvature)
        norm_roughness = normalize(roughness)
        norm_depth = normalize(depth_values)
        
        # 综合评分
        combined_score = (
            0.4 * norm_curvature +      # 曲率权重
            0.3 * norm_roughness +      # 粗糙度权重  
            0.3 * norm_depth            # 深度不连续权重
        )
        
        # 应用阈值
        threshold = (
            self.config['curvature_threshold'] * 0.4 +
            self.config['roughness_threshold'] * 0.3 +
            self.config['depth_discontinuity_threshold'] * 0.3
        )
        
        boundary_mask = combined_score > threshold
        print(f"[边界提取] 初始边界点数: {np.sum(boundary_mask)}")
        
        return boundary_mask
    
    def _cluster_boundary_points(self, boundary_points: np.ndarray) -> np.ndarray:
        """
        对边界点进行聚类去噪
        """
        if len(boundary_points) < self.config['min_cluster_size']:
            return np.arange(len(boundary_points))
            
        # DBSCAN聚类
        clustering = DBSCAN(
            eps=self.config['clustering_eps'],
            min_samples=self.config['min_cluster_size']
        )
        
        labels = clustering.fit_predict(boundary_points)
        
        # 选择最大的聚类
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[unique_labels != -1]  # 排除噪声点
        
        if len(valid_labels) == 0:
            return np.array([], dtype=int)
            
        # 找到最大的聚类
        largest_cluster_idx = np.argmax(counts[unique_labels != -1])
        largest_label = valid_labels[largest_cluster_idx]
        
        # 返回该聚类的索引
        clustered_indices = np.where(labels == largest_label)[0]
        
        print(f"[边界提取] 聚类后保留点数: {len(clustered_indices)}")
        return clustered_indices
    
    def _calculate_extraction_confidence(self, curvature: np.ndarray, 
                                       roughness: np.ndarray, 
                                       depth_values: np.ndarray) -> float:
        """
        计算边界提取的置信度
        """
        # 基于特征值的统计特性计算置信度
        curvature_score = np.mean(curvature) / (np.std(curvature) + 1e-8)
        roughness_score = np.mean(roughness) / (np.std(roughness) + 1e-8)
        depth_score = np.mean(depth_values) / (np.std(depth_values) + 1e-8)
        
        # 综合置信度
        confidence = np.clip(
            (curvature_score + roughness_score + depth_score) / 3.0,
            0.0, 1.0
        )
        
        return confidence