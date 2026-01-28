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

    pcd = fragment.point_cloud
    centroid = pcd.get_center()
    pcd.translate(-centroid)

    pts = np.asarray(pcd.points)
    scale = np.linalg.norm(pts, axis=1).max()
    pcd.scale(1.0 / scale, center=(0, 0, 0))

    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)

    # 更新Fragment的归一化状态和点云
    fragment.normalized = True
    fragment.scale = scale
    fragment.point_cloud = pcd

    meta = {
        "centroid": centroid,
        "scale": scale
    }
    return pcd, meta