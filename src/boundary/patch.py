import open3d as o3d
import numpy as np


def extract_section_patch(fragment, k_neighbors=50, thickness_ratio=0.3,
                          normal_to_surface_thresh=50.0, visualize=False):
    """
    基于边界点的K近邻提取陶瓷碎片的断面patch
    可视化统一由 run_mvp.py 的 visualize_section_patch 负责
    """
    # 容错1：无点云/无边界点直接返回
    if fragment.point_cloud is None or len(fragment.point_cloud.points) == 0:
        print(f"[断面提取] 碎片{fragment.id}无有效点云数据，跳过断面patch提取")
        return None
    if not hasattr(fragment, "boundary_pts") or fragment.boundary_pts is None or len(fragment.boundary_pts) == 0:
        print(f"[断面提取] 碎片{fragment.id}未检测到有效边界，跳过断面patch提取")
        return None

    pcd = fragment.point_cloud
    boundary_pts = fragment.boundary_pts
    pcd_pts = np.asarray(pcd.points)  # 整体点云坐标 (N,3)

    # 构建KD树，快速查找近邻（Open3D内置高效KD树）
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    # 核心逻辑：遍历所有边界点，提取每个点的K近邻，合并为断面候选点（集合去重）
    patch_vert_indices = set()
    for pt in boundary_pts:
        # 查找近邻：返回（近邻数，近邻索引，近邻距离）
        _, idx, distances = pcd_tree.search_knn_vector_3d(pt, k_neighbors)
        patch_vert_indices.update(idx)

    patch_vert_indices = np.array(list(patch_vert_indices))
    # 容错2：无有效断面点
    if len(patch_vert_indices) == 0:
        print(f"[断面提取] 碎片{fragment.id}无有效断面点，跳过")
        return None

    patch_indices_arr = np.array(list(patch_vert_indices), dtype=np.int64)
    if len(patch_indices_arr) == 0:
        print(f"[断面提取] 碎片{fragment.id}无有效断面点，跳过")
        return None

    # 厚度比例过滤
    thickness = getattr(fragment, 'thickness', None)
    if thickness is not None and thickness > 0:
        half_thresh = thickness * thickness_ratio / 2.0
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            )
        global_normal = np.mean(np.asarray(pcd.normals), axis=0)
        global_normal /= np.linalg.norm(global_normal) + 1e-9
        centroid = pcd_pts.mean(axis=0)
        proj = (pcd_pts[patch_indices_arr] - centroid) @ global_normal
        patch_indices_arr = patch_indices_arr[np.abs(proj) <= half_thresh]
        print(f"[断面提取] 厚度过滤（±{half_thresh:.4f}）后剩余 {len(patch_indices_arr)} 点")

    # 法向角度过滤
    if pcd.has_normals() and len(patch_indices_arr) > 0:
        normals = np.asarray(pcd.normals)
        global_normal = np.mean(normals, axis=0)
        global_normal /= np.linalg.norm(global_normal) + 1e-9
        candidate_normals = normals[patch_indices_arr]
        cos_angles = np.clip(
            np.abs(candidate_normals @ global_normal)
            / (np.linalg.norm(candidate_normals, axis=1) + 1e-9),
            0.0, 1.0
        )
        angles_deg = np.degrees(np.arccos(cos_angles))
        cross_section_min_angle = 90.0 - normal_to_surface_thresh
        patch_indices_arr = patch_indices_arr[angles_deg >= cross_section_min_angle]
        print(f"[断面提取] 法向过滤（≥{cross_section_min_angle:.1f}°）后剩余 {len(patch_indices_arr)} 点")

    if len(patch_indices_arr) == 0:
        print(f"[断面提取] 碎片{fragment.id}过滤后无有效断面点，跳过")
        return None

    section_patch = pcd.select_by_index(patch_indices_arr.tolist())
    fragment.section_patch = section_patch

    print(f"[断面提取] 碎片{fragment.id}完成，断面点数={len(patch_indices_arr)}，k={k_neighbors}")
    return section_patch

    # 可视化（整体点云+红色边界+蓝色断面）
    if visualize:
        # 仅可视化与当前 point_cloud 同坐标系的数据，避免 mesh/pcd 坐标系不一致导致错位
        # 使用boundary_points而不是boundary_pcd以保持一致性
        vis_list = [pcd, fragment.boundary_points, section_patch]
        o3d.visualization.draw_geometries(
            vis_list,
            window_name=f"碎片{fragment.id} - 断面patch提取结果",
            width=800, height=600
        )

    # 更新Fragment对象属性
    fragment.section_patch = section_patch

    print(f"[断面提取] 碎片{fragment.id}完成，提取到{len(patch_vert_indices)}个断面点（k={k_neighbors}）")
    return section_patch