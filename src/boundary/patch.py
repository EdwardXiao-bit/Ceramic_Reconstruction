# D:\ceramic_reconstruction\src\boundary\patch.py
import numpy as np
import open3d as o3d


def extract_section_patch(fragment, k_neighbors=50, visualize=False):
    """
    基于边界点的K近邻提取陶瓷碎片的断面patch（断裂局部区域）
    :param fragment: Fragment对象，需包含point_cloud和boundary_pts属性
    :param k_neighbors: 每个边界点的近邻数，默认50（碎片点云密则调大，疏则调小）
    :param visualize: 是否可视化断面patch，默认False
    :return: 断面patch点云o3d.PointCloud | None
    """
    # 容错1：无点云/无边界点直接返回
    if fragment.point_cloud is None or len(fragment.point_cloud.points) == 0:
        print(f"[断面提取] 碎片{fragment.id}无有效点云数据，跳过断面patch提取")
        return None
    if not hasattr(fragment, "boundary_pts") or fragment.boundary_pts is None:
        print(f"[断面提取] 碎片{fragment.id}未检测到边界，先执行边界检测再提取断面")
        return None

    pcd = fragment.point_cloud
    boundary_pts = fragment.boundary_pts
    pcd_pts = np.asarray(pcd.points)  # 整体点云坐标 (N,3)

    # 构建KD树，快速查找近邻（Open3D内置高效KD树）
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    # 核心逻辑：遍历所有边界点，提取每个点的K近邻，合并为断面候选点
    patch_vert_indices = set()
    for pt in boundary_pts:
        # 查找近邻：返回（近邻数，近邻索引，近邻距离）
        _, idx, _ = pcd_tree.search_knn_vector_3d(pt, k_neighbors)
        patch_vert_indices.update(idx)  # 用集合去重

    patch_vert_indices = np.array(list(patch_vert_indices))
    # 容错2：无有效断面点
    if len(patch_vert_indices) == 0:
        print(f"[断面提取] 碎片{fragment.id}无有效断面点，跳过")
        return None

    # 提取断面patch点云（蓝色，方便可视化）
    section_patch = o3d.geometry.PointCloud()
    section_patch.points = o3d.utility.Vector3dVector(pcd_pts[patch_vert_indices])
    section_patch.paint_uniform_color([0, 0, 1])  # 蓝色标记断面

    # 可视化（整体点云+红色边界+蓝色断面）
    if visualize:
        o3d.visualization.draw_geometries(
            [pcd, fragment.boundary_pcd, section_patch],
            window_name=f"碎片{fragment.id} - 断面patch提取结果",
            width=800, height=600
        )

    # 更新Fragment对象属性
    fragment.section_patch = section_patch

    print(f"[断面提取] 碎片{fragment.id}完成，提取到{len(patch_vert_indices)}个断面点（k={k_neighbors}）")
    return section_patch