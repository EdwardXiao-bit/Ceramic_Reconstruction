# D:\ceramic_reconstruction\src\boundary_validation\collision_detector.py
"""
碰撞与穿透检查模块
使用SDF或voxel occupancy检查边界附近是否有物理穿透可能性
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy.spatial import cKDTree
import trimesh

@dataclass
class CollisionResult:
    """碰撞检测结果"""
    collision_score: float                 # 碰撞得分（越小越好）
    penetration_depth: float              # 穿透深度
    collision_points: np.ndarray          # 碰撞点坐标
    collision_volumes: Dict[str, float]   # 不同类型的碰撞体积
    safety_margin: float                  # 安全边界
    detailed_analysis: Dict[str, Any]     # 详细分析结果

class CollisionDetector:
    """碰撞检测器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sdf_calculator = None
        self.voxel_grid = None
        
    def check_collision(self, fragment1: Any, fragment2: Any, 
                       transformation: np.ndarray) -> CollisionResult:
        """
        检查两个碎片在给定变换下的碰撞情况
        
        Args:
            fragment1: 第一个碎片
            fragment2: 第二个碎片  
            transformation: 碎片2相对于碎片1的变换矩阵
            
        Returns:
            CollisionResult: 碰撞检测结果
        """
        print("[碰撞检测] 开始碰撞与穿透检查...")
        
        # 获取点云数据
        points1 = self._get_fragment_points(fragment1)
        points2 = self._get_fragment_points(fragment2)
        
        if points1 is None or points2 is None:
            print("[碰撞检测] 点云数据获取失败")
            return self._create_failed_result()
        
        # 应用变换到碎片2
        transformed_points2 = self._apply_transformation(points2, transformation)
        
        collision_score = 0.0
        penetration_depth = 0.0
        collision_points = np.array([]).reshape(0, 3)
        collision_volumes = {}
        detailed_analysis = {}
        
        # 1. 使用SDF进行碰撞检测（如果启用）
        if self.config['sdf_enabled']:
            print("[碰撞检测] 使用SDF进行碰撞检测...")
            sdf_result = self._check_collision_with_sdf(points1, transformed_points2)
            collision_score += sdf_result['score'] * 0.7
            penetration_depth = max(penetration_depth, sdf_result['penetration'])
            collision_points = np.vstack([collision_points, sdf_result['collision_points']])
            collision_volumes['sdf_volume'] = sdf_result['volume']
            detailed_analysis['sdf_analysis'] = sdf_result['analysis']
        
        # 2. 使用体素占用进行碰撞检测（如果启用）
        if self.config['voxel_enabled']:
            print("[碰撞检测] 使用体素占用进行碰撞检测...")
            voxel_result = self._check_collision_with_voxels(points1, transformed_points2)
            collision_score += voxel_result['score'] * 0.3
            penetration_depth = max(penetration_depth, voxel_result['penetration'])
            collision_points = np.vstack([collision_points, voxel_result['collision_points']])
            collision_volumes['voxel_volume'] = voxel_result['volume']
            detailed_analysis['voxel_analysis'] = voxel_result['analysis']
        
        # 3. 计算安全边界
        safety_margin = self._calculate_safety_margin(points1, transformed_points2)
        
        # 4. 综合碰撞结果
        collision_result = CollisionResult(
            collision_score=float(np.clip(collision_score, 0.0, 1.0)),
            penetration_depth=float(penetration_depth),
            collision_points=collision_points,
            collision_volumes=collision_volumes,
            safety_margin=float(safety_margin),
            detailed_analysis=detailed_analysis
        )
        
        print(f"[碰撞检测] 检测完成:")
        print(f"  碰撞得分: {collision_result.collision_score:.3f}")
        print(f"  穿透深度: {collision_result.penetration_depth:.6f}")
        print(f"  碰撞点数: {len(collision_result.collision_points)}")
        print(f"  安全边界: {collision_result.safety_margin:.6f}")
        
        return collision_result
    
    def _get_fragment_points(self, fragment: Any) -> Optional[np.ndarray]:
        """获取碎片点云数据"""
        if hasattr(fragment, 'point_cloud') and fragment.point_cloud is not None:
            return np.asarray(fragment.point_cloud.points)
        elif hasattr(fragment, 'mesh') and fragment.mesh is not None:
            # 从网格采样点
            try:
                pcd = fragment.mesh.sample_points_uniformly(number_of_points=2000)
                return np.asarray(pcd.points)
            except:
                # 如果采样失败，使用顶点
                return np.asarray(fragment.mesh.vertices)
        else:
            return None
    
    def _apply_transformation(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """应用变换矩阵到点云"""
        homogeneous_points = np.hstack([points, np.ones((len(points), 1))])
        transformed_points = (transform @ homogeneous_points.T).T
        return transformed_points[:, :3]
    
    def _check_collision_with_sdf(self, points1: np.ndarray, points2: np.ndarray) -> Dict[str, Any]:
        """
        使用符号距离场(SDF)进行碰撞检测
        """
        try:
            # 创建点云对象
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(points1)
            
            # 计算SDF
            resolution = self.config['resolution']
            padding = self.config['padding_factor']
            
            # 计算包围盒
            all_points = np.vstack([points1, points2])
            bbox_min = np.min(all_points, axis=0)
            bbox_max = np.max(all_points, axis=0)
            bbox_size = bbox_max - bbox_min
            bbox_center = (bbox_min + bbox_max) / 2
            
            # 扩展包围盒
            extended_size = bbox_size * padding
            grid_min = bbox_center - extended_size / 2
            grid_max = bbox_center + extended_size / 2
            
            # 创建3D网格
            x = np.linspace(grid_min[0], grid_max[0], resolution)
            y = np.linspace(grid_min[1], grid_max[1], resolution)
            z = np.linspace(grid_min[2], grid_max[2], resolution)
            
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
            
            # 计算到点云1的符号距离
            tree1 = cKDTree(points1)
            distances1, _ = tree1.query(grid_points)
            
            # 简化的SDF计算（实际应用中应该使用更精确的方法）
            # 这里使用点到表面的近似距离
            sdf_values = distances1
            
            # 检查点云2中的点是否在碎片1内部
            tree_grid = cKDTree(grid_points)
            distances_to_grid, indices = tree_grid.query(points2)
            
            # 穿透检测阈值
            penetration_threshold = self.config['penetration_threshold']
            collision_mask = sdf_values[indices] < penetration_threshold
            
            collision_points = points2[collision_mask]
            penetration_depths = np.abs(sdf_values[indices][collision_mask])
            
            # 计算碰撞体积（简单近似）
            voxel_volume = np.prod(extended_size) / (resolution ** 3)
            collision_volume = np.sum(collision_mask) * voxel_volume
            
            # 碰撞得分（基于穿透点比例和平均穿透深度）
            penetration_ratio = np.sum(collision_mask) / len(points2)
            avg_penetration = np.mean(penetration_depths) if len(penetration_depths) > 0 else 0
            
            collision_score = penetration_ratio * (1.0 + avg_penetration * 100)  # 放大穿透深度的影响
            
            result = {
                'score': float(np.clip(collision_score, 0.0, 1.0)),
                'penetration': float(avg_penetration),
                'collision_points': collision_points,
                'volume': float(collision_volume),
                'analysis': {
                    'penetration_points': int(np.sum(collision_mask)),
                    'total_points': len(points2),
                    'penetration_ratio': float(penetration_ratio),
                    'avg_penetration_depth': float(avg_penetration)
                }
            }
            
            return result
            
        except Exception as e:
            print(f"[SDF碰撞检测] 失败: {e}")
            return self._create_failed_sdf_result()
    
    def _check_collision_with_voxels(self, points1: np.ndarray, points2: np.ndarray) -> Dict[str, Any]:
        """
        使用体素占用进行碰撞检测
        """
        try:
            # 创建体素网格
            voxel_size = self._estimate_voxel_size(points1, points2)
            
            # 为两个点云创建体素化表示
            voxel_grid1 = self._voxelize_points(points1, voxel_size)
            voxel_grid2 = self._voxelize_points(points2, voxel_size)
            
            # 检查重叠的体素
            collision_voxels = set(voxel_grid1.keys()) & set(voxel_grid2.keys())
            
            if not collision_voxels:
                return {
                    'score': 0.0,
                    'penetration': 0.0,
                    'collision_points': np.array([]).reshape(0, 3),
                    'volume': 0.0,
                    'analysis': {'collision_voxels': 0, 'total_voxels_1': len(voxel_grid1), 'total_voxels_2': len(voxel_grid2)}
                }
            
            # 计算碰撞点
            collision_points = []
            for voxel_key in collision_voxels:
                # 获取碰撞体素中的点
                points_in_voxel1 = voxel_grid1[voxel_key]
                points_in_voxel2 = voxel_grid2[voxel_key]
                
                # 简单的点对点距离检查
                for p1 in points_in_voxel1:
                    for p2 in points_in_voxel2:
                        if np.linalg.norm(p1 - p2) < voxel_size:
                            collision_points.append(p2)  # 记录碎片2的碰撞点
            
            collision_points = np.array(collision_points)
            
            # 计算碰撞统计
            collision_volume = len(collision_voxels) * (voxel_size ** 3)
            penetration_ratio = len(collision_points) / len(points2) if len(points2) > 0 else 0
            
            # 穿透深度估算
            if len(collision_points) > 0:
                # 计算碰撞点到最近的非碰撞点的距离作为穿透深度
                non_collision_points2 = points2[~np.isin(range(len(points2)), 
                    [i for i, p in enumerate(points2) if any(np.linalg.norm(p - cp) < voxel_size for cp in collision_points)])]
                
                if len(non_collision_points2) > 0:
                    tree_non_collision = cKDTree(non_collision_points2)
                    distances, _ = tree_non_collision.query(collision_points)
                    avg_penetration = np.mean(distances)
                else:
                    avg_penetration = voxel_size
            else:
                avg_penetration = 0.0
            
            collision_score = penetration_ratio * (1.0 + avg_penetration * 1000)  # 体素方法需要更大权重
            
            result = {
                'score': float(np.clip(collision_score, 0.0, 1.0)),
                'penetration': float(avg_penetration),
                'collision_points': collision_points,
                'volume': float(collision_volume),
                'analysis': {
                    'collision_voxels': len(collision_voxels),
                    'total_voxels_1': len(voxel_grid1),
                    'total_voxels_2': len(voxel_grid2),
                    'penetration_ratio': float(penetration_ratio)
                }
            }
            
            return result
            
        except Exception as e:
            print(f"[体素碰撞检测] 失败: {e}")
            return self._create_failed_voxel_result()
    
    def _estimate_voxel_size(self, points1: np.ndarray, points2: np.ndarray) -> float:
        """估算合适的体素大小"""
        all_points = np.vstack([points1, points2])
        bbox_size = np.max(all_points, axis=0) - np.min(all_points, axis=0)
        max_extent = np.max(bbox_size)
        
        # 基于分辨率估算体素大小
        resolution = self.config['resolution']
        voxel_size = max_extent / resolution * 0.5  # 稍微细化一些
        
        return max(voxel_size, 0.001)  # 最小体素大小
    
    def _voxelize_points(self, points: np.ndarray, voxel_size: float) -> Dict[tuple, list]:
        """将点云体素化"""
        voxel_grid = {}
        
        for point in points:
            # 计算体素坐标
            voxel_coords = tuple(np.floor(point / voxel_size).astype(int))
            
            if voxel_coords not in voxel_grid:
                voxel_grid[voxel_coords] = []
            voxel_grid[voxel_coords].append(point)
            
        return voxel_grid
    
    def _calculate_safety_margin(self, points1: np.ndarray, points2: np.ndarray) -> float:
        """
        计算安全边界（最小分离距离）
        """
        try:
            # 使用最近邻搜索计算最小距离
            tree1 = cKDTree(points1)
            distances, _ = tree1.query(points2)
            
            min_distance = np.min(distances)
            return float(min_distance)
            
        except Exception as e:
            print(f"[安全边界计算] 失败: {e}")
            return 0.0
    
    def _create_failed_result(self) -> CollisionResult:
        """创建失败的碰撞检测结果"""
        return CollisionResult(
            collision_score=1.0,  # 最差得分
            penetration_depth=float('inf'),
            collision_points=np.array([]).reshape(0, 3),
            collision_volumes={'failed': 0.0},
            safety_margin=0.0,
            detailed_analysis={'status': 'failed'}
        )
    
    def _create_failed_sdf_result(self) -> Dict[str, Any]:
        """创建失败的SDF结果"""
        return {
            'score': 1.0,
            'penetration': float('inf'),
            'collision_points': np.array([]).reshape(0, 3),
            'volume': 0.0,
            'analysis': {'status': 'sdf_failed'}
        }
    
    def _create_failed_voxel_result(self) -> Dict[str, Any]:
        """创建失败的体素结果"""
        return {
            'score': 1.0,
            'penetration': float('inf'),
            'collision_points': np.array([]).reshape(0, 3),
            'volume': 0.0,
            'analysis': {'status': 'voxel_failed'}
        }
    
    def advanced_collision_analysis(self, fragment1: Any, fragment2: Any,
                                  transformation: np.ndarray) -> Dict[str, Any]:
        """
        高级碰撞分析
        """
        print("[高级碰撞分析] 执行详细的碰撞分析...")
        
        # 执行基本碰撞检测
        basic_result = self.check_collision(fragment1, fragment2, transformation)
        
        # 获取详细点云数据
        points1 = self._get_fragment_points(fragment1)
        points2 = self._get_fragment_points(fragment2)
        transformed_points2 = self._apply_transformation(points2, transformation)
        
        # 1. 穿透深度分布分析
        penetration_analysis = self._analyze_penetration_distribution(
            points1, transformed_points2, basic_result.collision_points
        )
        
        # 2. 碰撞区域分析
        region_analysis = self._analyze_collision_regions(basic_result.collision_points)
        
        # 3. 力学稳定性分析
        stability_analysis = self._analyze_mechanical_stability(points1, transformed_points2)
        
        # 4. 优化建议
        optimization_suggestions = self._generate_optimization_suggestions(basic_result)
        
        advanced_analysis = {
            'basic_collision': {
                'score': basic_result.collision_score,
                'penetration_depth': basic_result.penetration_depth,
                'safety_margin': basic_result.safety_margin
            },
            'penetration_analysis': penetration_analysis,
            'region_analysis': region_analysis,
            'stability_analysis': stability_analysis,
            'optimization_suggestions': optimization_suggestions,
            'overall_assessment': self._assess_collision_severity(basic_result.collision_score)
        }
        
        print(f"[高级碰撞分析] 完成，总体评估: {advanced_analysis['overall_assessment']}")
        return advanced_analysis
    
    def _analyze_penetration_distribution(self, points1: np.ndarray, points2: np.ndarray,
                                        collision_points: np.ndarray) -> Dict[str, Any]:
        """分析穿透深度分布"""
        if len(collision_points) == 0:
            return {'distribution': 'no_collision', 'statistics': {}}
        
        # 计算每个碰撞点的穿透深度
        tree1 = cKDTree(points1)
        distances, _ = tree1.query(collision_points)
        
        # 统计分析
        stats = {
            'min_depth': float(np.min(distances)),
            'max_depth': float(np.max(distances)),
            'mean_depth': float(np.mean(distances)),
            'std_depth': float(np.std(distances)),
            'median_depth': float(np.median(distances))
        }
        
        # 深度分布分类
        if stats['max_depth'] < 0.001:
            distribution_type = 'surface_contact'
        elif stats['mean_depth'] < 0.01:
            distribution_type = 'light_penetration'
        elif stats['mean_depth'] < 0.05:
            distribution_type = 'moderate_penetration'
        else:
            distribution_type = 'heavy_penetration'
        
        return {
            'distribution': distribution_type,
            'statistics': stats,
            'depth_histogram': self._compute_histogram(distances)
        }
    
    def _analyze_collision_regions(self, collision_points: np.ndarray) -> Dict[str, Any]:
        """分析碰撞区域特征"""
        if len(collision_points) == 0:
            return {'regions': 0, 'region_characteristics': []}
        
        # 使用DBSCAN聚类识别碰撞区域
        from sklearn.cluster import DBSCAN
        
        clustering = DBSCAN(eps=0.02, min_samples=5).fit(collision_points)
        labels = clustering.labels_
        
        unique_labels = np.unique(labels[labels != -1])  # 排除噪声点
        n_regions = len(unique_labels)
        
        region_characteristics = []
        for label in unique_labels:
            region_points = collision_points[labels == label]
            centroid = np.mean(region_points, axis=0)
            size = len(region_points)
            
            # 计算区域的紧密程度
            distances_to_centroid = np.linalg.norm(region_points - centroid, axis=1)
            compactness = 1.0 / (1.0 + np.std(distances_to_centroid))
            
            region_characteristics.append({
                'centroid': centroid.tolist(),
                'size': int(size),
                'compactness': float(compactness)
            })
        
        return {
            'regions': int(n_regions),
            'region_characteristics': region_characteristics,
            'total_collision_points': len(collision_points)
        }
    
    def _analyze_mechanical_stability(self, points1: np.ndarray, points2: np.ndarray) -> Dict[str, Any]:
        """分析力学稳定性"""
        # 简化的稳定性分析
        # 检查重心位置和支撑情况
        
        # 计算两个碎片的重心
        centroid1 = np.mean(points1, axis=0)
        centroid2 = np.mean(points2, axis=0)
        
        # 计算相对高度差
        height_difference = abs(centroid2[2] - centroid1[2])  # 假设Z轴向上
        
        # 检查接触面积（简化的）
        tree1 = cKDTree(points1)
        close_points2, _ = tree1.query(points2, distance_upper_bound=0.02)
        contact_points = np.sum(~np.isnan(close_points2))
        contact_ratio = contact_points / len(points2)
        
        stability_score = 0.5 * (1.0 - min(height_difference, 0.1) / 0.1) + 0.5 * contact_ratio
        
        return {
            'stability_score': float(np.clip(stability_score, 0.0, 1.0)),
            'height_difference': float(height_difference),
            'contact_ratio': float(contact_ratio),
            'assessment': 'stable' if stability_score > 0.7 else 'unstable' if stability_score < 0.3 else 'marginally_stable'
        }
    
    def _generate_optimization_suggestions(self, collision_result: CollisionResult) -> list:
        """生成优化建议"""
        suggestions = []
        
        if collision_result.collision_score > 0.5:
            suggestions.append("严重碰撞：需要大幅调整相对位置")
        
        if collision_result.penetration_depth > 0.01:
            suggestions.append("穿透深度较大：建议沿法向方向分离碎片")
        
        if collision_result.safety_margin < 0.005:
            suggestions.append("安全边界不足：增加碎片间距离")
        
        if collision_result.collision_score < 0.3:
            suggestions.append("碰撞风险较低：可进行微调优化")
        
        return suggestions
    
    def _assess_collision_severity(self, collision_score: float) -> str:
        """评估碰撞严重程度"""
        if collision_score < 0.2:
            return 'no_collision'
        elif collision_score < 0.4:
            return 'minor_collision'
        elif collision_score < 0.7:
            return 'moderate_collision'
        else:
            return 'severe_collision'
    
    def _compute_histogram(self, values: np.ndarray, bins: int = 20) -> Dict[str, list]:
        """计算直方图"""
        if len(values) == 0:
            return {'bins': [], 'counts': []}
        
        counts, bin_edges = np.histogram(values, bins=bins)
        return {
            'bins': bin_edges.tolist(),
            'counts': counts.tolist()
        }