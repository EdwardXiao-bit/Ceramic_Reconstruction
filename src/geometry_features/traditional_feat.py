"""
传统几何特征（FPFH）作为 fallback
文档：当深度模型置信度不足时使用 FPFH/SHOT 进行匹配
"""
import numpy as np
import open3d as o3d


def compute_patch_fpfh(patch_pcd, knn=20):
    """
    对断面 patch 计算 FPFH 特征，聚合为全局 33 维描述子
    :param patch_pcd: o3d.PointCloud，需已估计法向（normalize 阶段已完成）
    :param knn: FPFH 计算的 K 近邻数
    :return: (33,) numpy 数组，L2 归一化；若失败返回 None
    """
    if patch_pcd is None or len(patch_pcd.points) < 3:
        return None

    # 确保有法向
    if not patch_pcd.has_normals():
        patch_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=min(knn, len(patch_pcd.points)))
        )
        patch_pcd.orient_normals_consistent_tangent_plane(k=min(knn, len(patch_pcd.points)))

    try:
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            patch_pcd,
            o3d.geometry.KDTreeSearchParamKNN(knn=min(knn, len(patch_pcd.points)))
        )
    except Exception:
        return None

    if fpfh is None or fpfh.data is None:
        return None

    # fpfh.data shape: (33, N)
    desc = np.asarray(fpfh.data)
    if desc.size == 0:
        return None

    # 聚合：均值池化得到全局 33 维
    global_desc = np.mean(desc, axis=1)
    norm = np.linalg.norm(global_desc)
    if norm > 1e-8:
        global_desc = global_desc / norm
    return global_desc.astype(np.float32)
