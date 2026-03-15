"""
全局冲突穿透检测模块
使用 SDF（Signed Distance Field）或 Voxel Grid 进行快速碰撞检测
"""

import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CollisionResult:
    """碰撞检测结果"""
    has_collision: bool
    collision_pairs: List[Tuple[int, int]]  # 碰撞碎片对
    penetration_depths: Dict[Tuple[int, int], float]  # 穿透深度
    collision_points: Dict[Tuple[int, int], np.ndarray]  # 碰撞点
    total_collision_volume: float  # 总碰撞体积


class SDFCollisionDetector:
    """基于 SDF 的碰撞检测器"""
    
    def __init__(self, voxel_size: float = 0.01, margin: float = 0.02):
        """
        初始化 SDF 碰撞检测器
        
        Args:
            voxel_size: 体素大小
            margin: 安全边界距离
        """
        self.voxel_size = voxel_size
        self.margin = margin
        
    def build_sdf(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.VoxelGrid:
        """
        构建网格的 SDF
        
        Args:
            mesh: 输入网格
            
        Returns:
            o3d.geometry.VoxelGrid: 体素化的 SDF
        """
        # 创建体素网格
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
            mesh, 
            voxel_size=self.voxel_size
        )
        
        return voxel_grid
    
    def compute_signed_distance(self, point: np.ndarray, 
                               sdf: o3d.geometry.VoxelGrid) -> float:
        """
        计算点到 SDF 的符号距离
        
        Args:
            point: 查询点 (3,)
            sdf: 符号距离场
            
        Returns:
            float: 符号距离（正表示外部，负表示内部）
        """
        # 找到最近的体素
        voxel_coord = sdf.get_voxel_coordinate(point)
        
        if sdf.check_if_in_bounds(voxel_coord):
            voxel = sdf.get_voxel(voxel_coord)
            return voxel[0]  # 假设第一个通道存储距离
        
        return float('inf')  # 超出范围
    
    def detect_collision(self, fragment1: any, pose1: np.ndarray,
                        fragment2: any, pose2: np.ndarray) -> CollisionResult:
        """
        检测两个碎片之间的碰撞
        
        Args:
            fragment1: 碎片 1（带 mesh）
            pose1: 碎片 1 的全局位姿 (4x4)
            fragment2: 碎片 2（带 mesh）
            pose2: 碎片 2 的全局位姿 (4x4)
            
        Returns:
            CollisionResult: 碰撞检测结果
        """
        # 获取网格并变换到全局坐标
        mesh1 = fragment1.mesh.transform(pose1)
        mesh2 = fragment2.mesh.transform(pose2)
        
        # 构建 SDF
        sdf1 = self.build_sdf(mesh1)
        
        # 采样 mesh2 的点并检查是否在 mesh1 内部
        points2 = np.asarray(mesh2.vertices)
        
        collision_points = []
        penetration_depths = []
        
        for point in points2:
            distance = self.compute_signed_distance(point, sdf1)
            
            if distance < -self.margin:  # 在内部且超过安全边界
                collision_points.append(point)
                penetration_depths.append(-distance)
        
        has_collision = len(collision_points) > 0
        
        result = CollisionResult(
            has_collision=has_collision,
            collision_pairs=[(fragment1.id, fragment2.id)] if has_collision else [],
            penetration_depths={(fragment1.id, fragment2.id): np.mean(penetration_depths) if penetration_depths else 0.0},
            collision_points={(fragment1.id, fragment2.id): np.array(collision_points)},
            total_collision_volume=np.sum(penetration_depths) * self.voxel_size**3 if penetration_depths else 0.0
        )
        
        return result


class VoxelCollisionDetector:
    """基于体素占用的碰撞检测器"""
    
    def __init__(self, voxel_size: float = 0.01):
        """
        初始化体素碰撞检测器
        
        Args:
            voxel_size: 体素大小
        """
        self.voxel_size = voxel_size
        
    def voxelize_fragment(self, fragment: any, pose: np.ndarray) -> set:
        """
        将碎片体素化
        
        Args:
            fragment: 碎片
            pose: 全局位姿 (4x4)
            
        Returns:
            set: 占据的体素坐标集合
        """
        # 变换点云到全局坐标
        if hasattr(fragment, 'point_cloud'):
            pcd = fragment.point_cloud.transform(pose)
        elif hasattr(fragment, 'mesh'):
            pcd = fragment.mesh.sample_points_uniformly(number_of_points=10000)
            pcd = pcd.transform(pose)
        else:
            return set()
        
        # 体素化
        voxels = set()
        points = np.asarray(pcd.points)
        
        for point in points:
            voxel_coord = tuple((point / self.voxel_size).astype(int))
            voxels.add(voxel_coord)
        
        return voxels
    
    def detect_collision(self, fragment1: any, pose1: np.ndarray,
                        fragment2: any, pose2: np.ndarray) -> CollisionResult:
        """
        检测两个碎片之间的碰撞
        
        Args:
            fragment1: 碎片 1
            pose1: 碎片 1 的全局位姿
            fragment2: 碎片 2
            pose2: 碎片 2 的全局位姿
            
        Returns:
            CollisionResult: 碰撞检测结果
        """
        # 体素化两个碎片
        voxels1 = self.voxelize_fragment(fragment1, pose1)
        voxels2 = self.voxelize_fragment(fragment2, pose2)
        
        # 检测体素重叠
        collision_voxels = voxels1.intersection(voxels2)
        
        has_collision = len(collision_voxels) > 0
        
        # 估算穿透深度（简化为重叠体素数 * 体素大小）
        penetration_depth = len(collision_voxels) * self.voxel_size if has_collision else 0.0
        
        result = CollisionResult(
            has_collision=has_collision,
            collision_pairs=[(fragment1.id, fragment2.id)] if has_collision else [],
            penetration_depths={(fragment1.id, fragment2.id): penetration_depth},
            collision_points={(fragment1.id, fragment2.id): np.array(list(collision_voxels)) * self.voxel_size},
            total_collision_volume=len(collision_voxels) * self.voxel_size**3
        )
        
        return result


def check_global_collisions(fragments: List[any],
                           poses: Dict[int, np.ndarray],
                           method: str = 'voxel',
                           voxel_size: float = 0.01) -> CollisionResult:
    """
    检查所有碎片之间的全局碰撞
    
    Args:
        fragments: 碎片列表
        poses: 碎片位姿字典 {id: 4x4 变换矩阵}
        method: 检测方法 ('voxel' 或 'sdf')
        voxel_size: 体素大小
        
    Returns:
        CollisionResult: 全局碰撞检测结果
    """
    print(f"[全局碰撞检测] 开始检测，{len(fragments)}个碎片")
    
    # 选择检测器
    if method == 'sdf':
        detector = SDFCollisionDetector(voxel_size=voxel_size)
    else:
        detector = VoxelCollisionDetector(voxel_size=voxel_size)
    
    all_collision_pairs = []
    all_penetration_depths = {}
    all_collision_points = {}
    total_volume = 0.0
    
    # 检查所有碎片对
    n = len(fragments)
    checked_count = 0
    
    for i in range(n):
        frag1 = fragments[i]
        id1 = frag1.id
        
        for j in range(i + 1, n):
            frag2 = fragments[j]
            id2 = frag2.id
            
            # 获取位姿（确保是 numpy 数组）
            pose1_data = poses.get(id1, np.eye(4))
            pose2_data = poses.get(id2, np.eye(4))
            
            # 如果是 FragmentPose 对象，转换为 numpy 矩阵
            if hasattr(pose1_data, 'get_transformation'):
                pose1 = pose1_data.get_transformation()
            else:
                pose1 = np.array(pose1_data) if isinstance(pose1_data, list) else pose1_data
            
            if hasattr(pose2_data, 'get_transformation'):
                pose2 = pose2_data.get_transformation()
            else:
                pose2 = np.array(pose2_data) if isinstance(pose2_data, list) else pose2_data
            
            # 检测碰撞
            result = detector.detect_collision(frag1, pose1, frag2, pose2)
            
            if result.has_collision:
                all_collision_pairs.extend(result.collision_pairs)
                all_penetration_depths.update(result.penetration_depths)
                all_collision_points.update(result.collision_points)
                total_volume += result.total_collision_volume
                
                print(f"[碰撞检测] 发现碰撞：碎片{id1} - 碎片{id2}, "
                      f"穿透深度：{result.penetration_depths[(id1, id2)]:.6f}")
            
            checked_count += 1
    
    print(f"[全局碰撞检测] 完成，检查{checked_count}对，发现{len(all_collision_pairs)}个碰撞")
    
    result = CollisionResult(
        has_collision=len(all_collision_pairs) > 0,
        collision_pairs=all_collision_pairs,
        penetration_depths=all_penetration_depths,
        collision_points=all_collision_points,
        total_collision_volume=total_volume
    )
    
    return result


def visualize_collision_areas(fragments: List[any],
                             poses: Dict[int, np.ndarray],
                             collision_result: CollisionResult) -> o3d.geometry.PointCloud:
    """
    可视化碰撞区域
    
    Args:
        fragments: 碎片列表
        poses: 位姿字典
        collision_result: 碰撞检测结果
        
    Returns:
        o3d.geometry.PointCloud: 可视化点云
    """
    vis_pcd = o3d.geometry.PointCloud()
    
    # 添加原始碎片（半透明蓝色）
    for fragment in fragments:
        pose = poses.get(fragment.id, np.eye(4))
        
        if hasattr(fragment, 'point_cloud'):
            pcd = fragment.point_cloud.transform(pose)
        elif hasattr(fragment, 'mesh'):
            pcd = fragment.mesh.sample_points_uniformly(number_of_points=5000)
            pcd = pcd.transform(pose)
        else:
            continue
        
        # 设置颜色为蓝色
        colors = np.zeros((len(pcd.points), 3))
        colors[:, 2] = 0.7  # 蓝色
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        vis_pcd += pcd
    
    # 添加碰撞点（红色）
    for (id1, id2), points in collision_result.collision_points.items():
        if len(points) > 0:
            collision_pcd = o3d.geometry.PointCloud()
            collision_pcd.points = o3d.utility.Vector3dVector(points)
            
            # 设置颜色为红色
            colors = np.ones((len(points), 3))
            colors[:, 0] = 1.0  # 红色
            colors[:, 1:] = 0.0
            collision_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            vis_pcd += collision_pcd
    
    # 可视化
    o3d.visualization.draw_geometries([vis_pcd])
    
    return vis_pcd
