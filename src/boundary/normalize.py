"""
边界规范化模块
依据文档(一)第2节：尺度归一化、局部坐标系、重采样、法向一致化
"""
import numpy as np
import open3d as o3d


# 默认重采样点数（文档建议 2048-4096，Predator 适配）
DEFAULT_PATCH_POINTS = 2048


def normalize_patch(fragment, n_points=DEFAULT_PATCH_POINTS):
    """
    对断面 patch 执行完整规范化流程
    :param fragment: Fragment 对象，需已包含 section_patch
    :param n_points: 重采样目标点数
    :return: 规范化后的 o3d.PointCloud | None
    """
    if not hasattr(fragment, "section_patch") or fragment.section_patch is None:
        return None

    patch = fragment.section_patch
    pts = np.asarray(patch.points)
    if len(pts) < 3:
        return None

    # ===== 1. 尺度归一化（单位球）=====
    center = pts.mean(axis=0)
    pts_centered = pts - center
    max_dist = np.linalg.norm(pts_centered, axis=1).max()
    if max_dist < 1e-8:
        return None
    scale = max_dist
    pts_scaled = pts_centered / scale

    # 保存原始尺度与中心，便于后续恢复
    if not hasattr(fragment, "patch_norm_meta"):
        fragment.patch_norm_meta = {}
    fragment.patch_norm_meta["scale"] = float(scale)
    fragment.patch_norm_meta["center"] = center.copy()

    # ===== 2. 局部坐标系（PCA）=====
    _, _, Vt = np.linalg.svd(pts_scaled)
    axes = Vt  # (3,3)，行为主方向
    pts_local = pts_scaled @ axes.T  # 变换到局部坐标系
    fragment.patch_norm_meta["axes"] = axes.copy()

    # ===== 3. 点云重采样（Poisson/voxel 思想，此处用均匀采样保证点数）=====
    if len(pts_local) >= n_points:
        idx = np.random.choice(len(pts_local), n_points, replace=False)
        pts_r = pts_local[idx]
    else:
        idx = np.random.choice(len(pts_local), n_points, replace=True)
        pts_r = pts_local[idx]

    pcd_resampled = o3d.geometry.PointCloud()
    pcd_resampled.points = o3d.utility.Vector3dVector(pts_r)

    # ===== 4. 法向估计与一致化 =====
    pcd_resampled.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
    )
    pcd_resampled.orient_normals_consistent_tangent_plane(k=20)

    # 更新 fragment
    fragment.section_patch = pcd_resampled

    return pcd_resampled


def denormalize_patch_point(p_local, fragment):
    """
    将局部坐标系中的点还原到原始坐标系
    :param p_local: (3,) 或 (N,3) 局部坐标
    :param fragment: 包含 patch_norm_meta 的 Fragment
    :return: 原始坐标系中的点
    """
    if not hasattr(fragment, "patch_norm_meta"):
        return p_local
    meta = fragment.patch_norm_meta
    scale = meta.get("scale", 1.0)
    center = meta.get("center", np.zeros(3))
    axes = meta.get("axes", np.eye(3))
    p_orig = (np.asarray(p_local) @ axes) * scale + center
    return p_orig
