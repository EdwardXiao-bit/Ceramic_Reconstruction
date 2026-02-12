# D:\ceramic_reconstruction\src\boundary\dual_boundary_rim.py
"""
双边界rim曲线提取模块
用于处理两条边界线之间区域的rim曲线提取
"""
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


def extract_patch_between_boundaries(fragment, boundary1_pts, boundary2_pts, 
                                   k_neighbors=30, expand_factor=1.5, visualize=False):
    """
    在两条边界线之间提取patch区域
    :param fragment: Fragment对象
    :param boundary1_pts: 第一条边界点集 (N1, 3)
    :param boundary2_pts: 第二条边界点集 (N2, 3)
    :param k_neighbors: 近邻搜索参数
    :param expand_factor: 扩展因子，控制patch区域大小
    :param visualize: 是否可视化
    :return: patch点云
    """
    if fragment.point_cloud is None:
        print("[双边界Patch] 无效点云数据")
        return None

    pcd = fragment.point_cloud
    pcd_pts = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # 合并两条边界线的所有点
    all_boundary_pts = np.vstack([boundary1_pts, boundary2_pts])
    
    # 提取边界点附近的点作为种子点
    seed_indices = set()
    for pt in all_boundary_pts:
        _, idx, _ = pcd_tree.search_knn_vector_3d(pt, k_neighbors)
        seed_indices.update(idx)
    
    seed_indices = list(seed_indices)
    
    # 使用区域增长算法扩展patch区域
    patch_indices = set(seed_indices)
    frontier = set(seed_indices)
    
    # 计算边界点的平均距离作为增长阈值
    if len(all_boundary_pts) > 1:
        distances = []
        for i in range(min(100, len(all_boundary_pts))):
            for j in range(i+1, min(i+10, len(all_boundary_pts))):
                distances.append(np.linalg.norm(all_boundary_pts[i] - all_boundary_pts[j]))
        avg_distance = np.mean(distances) if distances else 0.1
        growth_threshold = avg_distance * expand_factor
    else:
        growth_threshold = 0.1

    # 区域增长
    while frontier:
        current_idx = frontier.pop()
        current_pt = pcd_pts[current_idx]
        
        # 查找当前点的近邻
        _, neighbor_idx, _ = pcd_tree.search_radius_vector_3d(current_pt, growth_threshold)
        
        for idx in neighbor_idx:
            if idx not in patch_indices:
                # 检查该点是否在两条边界之间
                if _is_between_boundaries(pcd_pts[idx], boundary1_pts, boundary2_pts):
                    patch_indices.add(idx)
                    frontier.add(idx)

    patch_indices = np.array(list(patch_indices))
    
    if len(patch_indices) == 0:
        print("[双边界Patch] 未找到有效patch点")
        return None

    # 创建patch点云
    patch_pcd = o3d.geometry.PointCloud()
    patch_pcd.points = o3d.utility.Vector3dVector(pcd_pts[patch_indices])
    patch_pcd.paint_uniform_color([0, 0, 1])  # 蓝色

    # 可视化
    if visualize:
        # 创建边界线可视化
        boundary1_pcd = o3d.geometry.PointCloud()
        boundary1_pcd.points = o3d.utility.Vector3dVector(boundary1_pts)
        boundary1_pcd.paint_uniform_color([1, 0, 0])  # 红色
        
        boundary2_pcd = o3d.geometry.PointCloud()
        boundary2_pcd.points = o3d.utility.Vector3dVector(boundary2_pts)
        boundary2_pcd.paint_uniform_color([0, 1, 0])  # 绿色

        o3d.visualization.draw_geometries(
            [pcd, boundary1_pcd, boundary2_pcd, patch_pcd],
            window_name="双边界Patch提取结果",
            width=1000, height=800
        )

    return patch_pcd


def _is_between_boundaries(point, boundary1_pts, boundary2_pts, threshold=0.05):
    """
    判断点是否在两条边界线之间
    :param point: 待判断的点 (3,)
    :param boundary1_pts: 第一条边界点集
    :param boundary2_pts: 第二条边界点集
    :param threshold: 距离阈值
    :return: bool
    """
    # 计算点到两条边界线的最小距离
    dist_to_boundary1 = np.min([np.linalg.norm(point - bp) for bp in boundary1_pts])
    dist_to_boundary2 = np.min([np.linalg.norm(point - bp) for bp in boundary2_pts])
    
    # 如果点到两条边界的距离都不为0，且相对较小，则认为在边界之间
    return dist_to_boundary1 > threshold and dist_to_boundary2 > threshold


def extract_rim_from_dual_boundary_patch(fragment, boundary1_pts, boundary2_pts,
                                       patch_k_neighbors=30, expand_factor=1.5,
                                       rim_samples=200, visualize=False):
    """
    完整流程：从双边界间的patch提取rim曲线
    :param fragment: Fragment对象
    :param boundary1_pts: 第一条边界点集
    :param boundary2_pts: 第二条边界点集
    :param patch_k_neighbors: patch提取参数
    :param expand_factor: patch扩展因子
    :param rim_samples: rim曲线重采样点数
    :param visualize: 是否可视化
    :return: rim曲线点集, rim点云
    """
    print("[双边界Rim] 开始双边界rim曲线提取...")
    
    # 步骤1: 提取两条边界间的patch
    patch_pcd = extract_patch_between_boundaries(
        fragment, boundary1_pts, boundary2_pts, 
        k_neighbors=patch_k_neighbors, expand_factor=expand_factor, visualize=False
    )
    
    if patch_pcd is None:
        print("[双边界Rim] Patch提取失败")
        return None, None

    # 步骤2: 从patch中提取rim边界点
    patch_pts = np.asarray(patch_pcd.points)
    
    # 使用法向变化检测边界点
    patch_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    normals = np.asarray(patch_pcd.normals)
    
    # 计算曲率（法向变化）
    tree = o3d.geometry.KDTreeFlann(patch_pcd)
    curvature = np.zeros(len(patch_pts))
    
    for i, pt in enumerate(patch_pts):
        _, idx, _ = tree.search_knn_vector_3d(pt, 20)
        nbr_normals = normals[idx]
        mean_normal = nbr_normals.mean(axis=0)
        curvature[i] = np.mean(np.linalg.norm(nbr_normals - mean_normal, axis=1))
    
    # 选择高曲率点作为rim候选点
    num_rim_candidates = max(len(patch_pts) // 10, 50)  # 选择10%或至少50个点
    rim_candidate_indices = np.argsort(curvature)[-num_rim_candidates:]
    rim_candidate_pts = patch_pts[rim_candidate_indices]
    
    # 步骤3: 聚类去除噪声，保留主要rim结构
    if len(rim_candidate_pts) > 10:
        clustering = DBSCAN(eps=0.02, min_samples=5).fit(rim_candidate_pts)
        labels = clustering.labels_
        unique_labels = [l for l in np.unique(labels) if l != -1]
        
        if unique_labels:
            # 选择最大的簇作为rim
            best_label = max(unique_labels, key=lambda l: np.sum(labels == l))
            rim_pts = rim_candidate_pts[labels == best_label]
        else:
            rim_pts = rim_candidate_pts
    else:
        rim_pts = rim_candidate_pts

    if len(rim_pts) < 10:
        print("[双边界Rim] 未找到足够的rim点")
        return None, None

    # 步骤4: 应用标准rim曲线提取算法
    center = rim_pts.mean(axis=0)
    pts_centered = rim_pts - center
    
    # PCA平面拟合
    _, _, Vt = np.linalg.svd(pts_centered)
    normal = Vt[-1]
    x_axis = Vt[0]
    y_axis = np.cross(normal, x_axis)
    
    # 极角排序
    x = pts_centered @ x_axis
    y = pts_centered @ y_axis
    theta = np.arctan2(y, x)
    order = np.argsort(theta)
    rim_sorted = rim_pts[order]
    
    # 等弧长重采样
    diffs = np.linalg.norm(np.diff(rim_sorted, axis=0), axis=1)
    arc = np.insert(np.cumsum(diffs), 0, 0)
    arc /= arc[-1]
    
    target = np.linspace(0, 1, rim_samples)
    rim_resampled = np.vstack([
        np.interp(target, arc, rim_sorted[:, i])
        for i in range(3)
    ]).T

    # 创建结果对象
    rim_pcd = o3d.geometry.PointCloud()
    rim_pcd.points = o3d.utility.Vector3dVector(rim_resampled)
    rim_pcd.paint_uniform_color([1, 0, 1])  # 紫色表示最终rim曲线

    rim_lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        rim_pcd, rim_pcd,
        np.array([[i, i + 1] for i in range(rim_samples - 1)])
    )
    rim_lines.paint_uniform_color([1, 0, 1])

    # 可视化完整结果
    if visualize:
        # 创建边界可视化
        boundary1_vis = o3d.geometry.PointCloud()
        boundary1_vis.points = o3d.utility.Vector3dVector(boundary1_pts)
        boundary1_vis.paint_uniform_color([1, 0, 0])
        
        boundary2_vis = o3d.geometry.PointCloud()
        boundary2_vis.points = o3d.utility.Vector3dVector(boundary2_pts)
        boundary2_vis.paint_uniform_color([0, 1, 0])
        
        # 创建patch可视化
        patch_vis = o3d.geometry.PointCloud()
        patch_vis.points = o3d.utility.Vector3dVector(patch_pts)
        patch_vis.paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries(
            [fragment.point_cloud, boundary1_vis, boundary2_vis, patch_vis, rim_pcd, rim_lines],
            window_name="双边界Rim提取完整流程",
            width=1200, height=900
        )

    # 更新fragment属性
    fragment.dual_boundary_patch = patch_pcd
    fragment.dual_boundary_rim_curve = rim_resampled
    fragment.dual_boundary_rim_pcd = rim_pcd
    fragment.dual_boundary_rim_lines = rim_lines

    print(f"[双边界Rim] 提取完成，rim曲线点数: {len(rim_resampled)}")
    return rim_resampled, rim_pcd