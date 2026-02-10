import numpy as np
import open3d as o3d


def normalize_fragment(fragment, voxel_size=0.005):
    """
    修正：接收Fragment对象，归一化其点云
    :param fragment: Fragment对象（包含point_cloud）
    :param voxel_size: 体素下采样尺寸
    :return: 归一化后的点云 + 元数据
    """
    if fragment.point_cloud is None:
        print(f"碎片{fragment.id}无点云数据，跳过归一化")
        return None, None

    # 保存原始点云副本供边界检测使用
    original_pcd = fragment.point_cloud
    original_points = np.asarray(original_pcd.points)

    # 动态调整体素大小，避免信息丢失
    if voxel_size is None:
        # 如果没有指定体素大小，根据点云大小自动选择
        n_points = len(original_points)
        if n_points > 100000:
            voxel_size = 0.01   # 大点云使用较大的体素，保留足够点
        elif n_points > 50000:
            voxel_size = 0.008
        else:
            voxel_size = 0.005

    pcd = fragment.point_cloud
    centroid = pcd.get_center()
    pcd.translate(-centroid)

    pts = np.asarray(pcd.points)
    scale = np.linalg.norm(pts, axis=1).max()
    pcd.scale(1.0 / scale, center=(0, 0, 0))

    # 为边界检测保存未下采样的点云
    fragment.unscaled_pcd = pcd  # 保存未下采用的点云

    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)

    # 更新Fragment的归一化状态和点云
    fragment.normalized = True
    fragment.scale = scale
    fragment.centroid = centroid
    fragment.original_points = original_points  # 保存原始坐标
    fragment.point_cloud = pcd

    meta = {
        "centroid": centroid,
        "scale": scale
    }
    return pcd, meta