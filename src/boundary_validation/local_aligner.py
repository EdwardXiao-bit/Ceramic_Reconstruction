# src/boundary_validation/local_aligner.py
"""
局部对齐精化模块（修复版）
核心修改：
1. 禁用失效的DCP（无预训练权重时输出无意义）
2. 用多初始化ICP代替，大幅提升对齐成功率
3. 用FPFH粗配准作为ICP初始化
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import torch.nn as nn


@dataclass
class AlignmentResult:
    refined_transformation: np.ndarray
    alignment_error: float
    fitness_score: float
    rmse: float
    iterations_used: int
    convergence_status: str


class LocalAligner:
    """局部对齐精化器（修复版）"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        print("[局部对齐] 初始化（使用多初始化ICP）")

    def refine_alignment(self, fragment1: Any, fragment2: Any,
                         initial_transformation: np.ndarray) -> AlignmentResult:
        """
        精化对齐结果
        策略：FPFH粗配准 + 多初始化ICP
        """
        print("[局部对齐] 开始局部对齐精化...")

        points1 = self._get_fragment_points(fragment1)
        points2 = self._get_fragment_points(fragment2)

        if points1 is None or points2 is None:
            print("[局部对齐] 点云数据获取失败")
            return self._create_failed_result()

        best_result = None
        best_fitness = -1.0

        # === 策略1: 以传入变换为初始化的ICP ===
        r1 = self._icp_with_init(points1, points2, initial_transformation)
        if r1.fitness_score > best_fitness:
            best_result, best_fitness = r1, r1.fitness_score

        # === 策略2: FPFH全局粗配准 + ICP ===
        T_fpfh = self._fpfh_global_registration(points1, points2)
        if T_fpfh is not None:
            r2 = self._icp_with_init(points1, points2, T_fpfh)
            if r2.fitness_score > best_fitness:
                best_result, best_fitness = r2, r2.fitness_score

        # === 策略3: 多随机初始化ICP（粗暴但有效）===
        for _ in range(5):
            T_rand = self._random_init_transform(points1, points2)
            r3 = self._icp_with_init(points1, points2, T_rand)
            if r3.fitness_score > best_fitness:
                best_result, best_fitness = r3, r3.fitness_score

        if best_result is None:
            return self._create_failed_result()

        print(f"[局部对齐] 完成: fitness={best_result.fitness_score:.3f}, "
              f"rmse={best_result.rmse:.4f}, status={best_result.convergence_status}")
        return best_result

    def _fpfh_global_registration(self, points1: np.ndarray, points2: np.ndarray) -> Optional[np.ndarray]:
        """FPFH特征 + RANSAC全局粗配准"""
        try:
            # 估算体素大小
            all_pts = np.vstack([points1, points2])
            bbox = np.max(all_pts, axis=0) - np.min(all_pts, axis=0)
            voxel_size = max(np.max(bbox) * 0.05, 0.01)

            def preprocess(pts):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd_down = pcd.voxel_down_sample(voxel_size)
                if len(pcd_down.points) < 5:
                    pcd_down = pcd
                pcd_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
                fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    pcd_down,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
                return pcd_down, fpfh

            pcd1_down, fpfh1 = preprocess(points1)
            pcd2_down, fpfh2 = preprocess(points2)

            if len(pcd1_down.points) < 4 or len(pcd2_down.points) < 4:
                return None

            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                pcd1_down, pcd2_down, fpfh1, fpfh2,
                mutual_filter=True,
                max_correspondence_distance=voxel_size * 3,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 3)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
            )

            if result.fitness > 0.01:
                print(f"[FPFH全局配准] fitness={result.fitness:.3f}")
                return result.transformation
            return None
        except Exception as e:
            print(f"[FPFH全局配准] 失败: {e}")
            return None

    def _random_init_transform(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """生成随机初始变换（基于两点云范围）"""
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)

        # 随机旋转
        angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis) + 1e-8

        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        # 平移：将pcd1中心对齐到pcd2中心，再加随机扰动
        t = c2 - R @ c1
        bbox_size = np.max(np.max(points1, axis=0) - np.min(points1, axis=0))
        t += np.random.randn(3) * bbox_size * 0.1

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _icp_with_init(self, points1: np.ndarray, points2: np.ndarray,
                       init_transform: np.ndarray) -> AlignmentResult:
        """以给定初始变换执行ICP"""
        icp_config = self.config.get('icp', {})
        max_iter = icp_config.get('max_iterations', 50)

        # 自适应距离阈值
        bbox = np.max(points1, axis=0) - np.min(points1, axis=0)
        avg_extent = np.mean(np.max(bbox))
        threshold = max(avg_extent * 0.1, icp_config.get('distance_threshold', 0.05))

        try:
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(points1)
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(points2)

            result = o3d.pipelines.registration.registration_icp(
                source=source,
                target=target,
                max_correspondence_distance=threshold,
                init=init_transform,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=max_iter
                )
            )

            fitness = result.fitness
            rmse = result.inlier_rmse if result.inlier_rmse > 0 else float('inf')

            # 判断收敛
            fitness_thresh = icp_config.get('fitness_threshold', 0.3)
            rmse_thresh = icp_config.get('rmse_threshold', 0.05)

            if fitness >= fitness_thresh:
                status = 'converged'
            elif fitness >= 0.1:
                status = 'partial'
            else:
                status = 'failed'

            return AlignmentResult(
                refined_transformation=result.transformation,
                alignment_error=1.0 - fitness,
                fitness_score=fitness,
                rmse=rmse,
                iterations_used=max_iter,
                convergence_status=status
            )
        except Exception as e:
            print(f"[ICP] 失败: {e}")
            return AlignmentResult(
                refined_transformation=init_transform,
                alignment_error=1.0,
                fitness_score=0.0,
                rmse=float('inf'),
                iterations_used=0,
                convergence_status='failed'
            )

    def _get_fragment_points(self, fragment: Any) -> Optional[np.ndarray]:
        if hasattr(fragment, 'point_cloud') and fragment.point_cloud is not None:
            return np.asarray(fragment.point_cloud.points)
        elif hasattr(fragment, 'mesh') and fragment.mesh is not None:
            try:
                pcd = fragment.mesh.sample_points_uniformly(number_of_points=3000)
                return np.asarray(pcd.points)
            except Exception:
                return np.asarray(fragment.mesh.vertices)
        return None

    def _create_failed_result(self) -> AlignmentResult:
        return AlignmentResult(
            refined_transformation=np.eye(4),
            alignment_error=1.0,
            fitness_score=0.0,
            rmse=float('inf'),
            iterations_used=0,
            convergence_status='failed'
        )

    # 保留接口兼容性
    def multi_scale_alignment(self, fragment1, fragment2, initial_transformation):
        return self.refine_alignment(fragment1, fragment2, initial_transformation)

    def refine_alignment_at_scale(self, points1, points2, transform):
        return self._icp_with_init(points1, points2, transform)