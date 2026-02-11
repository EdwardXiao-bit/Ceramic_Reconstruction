# D:\ceramic_reconstruction\src\boundary\detect.py
import numpy as np
import open3d as o3d
from collections import defaultdict
from sklearn.cluster import DBSCAN


def _select_rim_cluster(boundary_pts):
    """
    从所有 boundary 点中，筛选最可能是 rim 的那一簇
    判据：最大连通簇 + 最大空间跨度
    """
    if len(boundary_pts) < 50:
        return boundary_pts

    # DBSCAN 聚类（rim 通常是一整圈，密度高、规模大）
    clustering = DBSCAN(eps=0.01, min_samples=10).fit(boundary_pts)
    labels = clustering.labels_

    unique_labels = [l for l in np.unique(labels) if l != -1]
    if len(unique_labels) == 0:
        return boundary_pts

    best_score = -1
    best_cluster = None

    for lb in unique_labels:
        pts = boundary_pts[labels == lb]
        if len(pts) < 30:
            continue

        # 空间跨度作为 rim 判据（rim 是最大的一圈）
        bbox = pts.max(axis=0) - pts.min(axis=0)
        span = np.linalg.norm(bbox)

        score = span * len(pts)
        if score > best_score:
            best_score = score
            best_cluster = pts

    return best_cluster if best_cluster is not None else boundary_pts


def detect_boundary(fragment, visualize=False, curvature_thresh=0.1):
    """
    陶瓷碎片 rim 边界检测：
    - 优先 mesh 拓扑
    - fallback 到点云几何
    - 最终筛选 rim 主边界
    - 返回原始坐标系中的边界点
    """
    if fragment.point_cloud is None or len(fragment.point_cloud.points) == 0:
        print(f"[边界检测] 碎片{fragment.id}无有效点云数据")
        return None, None

    pcd = fragment.point_cloud
    boundary_pts = None

    # ========= 模式 1：网格拓扑 =========
    if fragment.mesh is not None and len(fragment.mesh.triangles) > 0:
        print(f"[边界检测] 碎片{fragment.id}使用网格拓扑法")
        mesh = fragment.mesh
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        edge_count = defaultdict(int)
        for tri in triangles:
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for e in edges:
                edge_count[tuple(sorted(e))] += 1

        boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
        if len(boundary_edges) > 0:
            idx = np.unique(np.array(boundary_edges).flatten())
            boundary_pts = vertices[idx]

    # ========= 模式 2：点云几何 =========
    if boundary_pts is None or len(boundary_pts) == 0:
        print(f"[边界检测] 碎片{fragment.id}使用点云几何法")

        # 优先使用原始点云进行边界检测，避免归一化导致的信息丢失
        if hasattr(fragment, 'original_points') and fragment.original_points is not None:
            # 使用原始点云
            detection_pcd = o3d.geometry.PointCloud()
            detection_pcd.points = o3d.utility.Vector3dVector(fragment.original_points)
            detection_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
            )
            pts = np.asarray(detection_pcd.points)
            normals = np.asarray(detection_pcd.normals)
            print(f"  使用原始点云（{len(pts)} 个点）进行边界检测")
        else:
            # 使用归一化的点云
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
            )
            pts = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            print(f"  使用归一化点云（{len(pts)} 个点）进行边界检测")

        # Open3D 没有稳定曲率 API，用邻域法向变化近似
        pts = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        # 使用原始点云的KD树
        detection_pcd = o3d.geometry.PointCloud()
        detection_pcd.points = o3d.utility.Vector3dVector(pts)
        tree = o3d.geometry.KDTreeFlann(detection_pcd)
        curvature = np.zeros(len(pts))

        # 计算曲率
        print(f"  计算曲率...")
        for i, pt in enumerate(pts):
            _, idx, _ = tree.search_knn_vector_3d(pt, 20)
            nbr_normals = normals[idx]
            mean_normal = nbr_normals.mean(axis=0)
            curvature[i] = np.mean(
                np.linalg.norm(nbr_normals - mean_normal, axis=1)
            )

        # 陶瓷碎片边界检测策略
        # 选择曲率最低的点作为边界（陶瓷碎片边界通常曲率较低）
        num_boundary = max(len(pts) // 20, 100)  # 至少100个点，或总数的5%
        boundary_idx = np.argsort(curvature)[:num_boundary]
        print(f"  选择最低曲率 {num_boundary} 个点作为候选边界")

        boundary_pts = pts[boundary_idx]

    # ========= 关键新增：rim 主边界筛选 =========
    rim_pts_normalized = _select_rim_cluster(boundary_pts)

    # ========= 坐标系转换：归一化坐标 -> 原始坐标 =========
    if hasattr(fragment, 'original_points') and fragment.original_points is not None:
        # 如果有保存的原始坐标，使用反归一化
        scale = getattr(fragment, 'scale', 1.0)
        centroid = getattr(fragment, 'centroid', np.zeros(3))
        rim_pts_original = rim_pts_normalized * scale + centroid

        print(f"[边界检测] 已将边界点从归一化坐标系转换回原始坐标系")
        print(f"  归一化边界点范围: [{rim_pts_normalized.min():.3f}, {rim_pts_normalized.max():.3f}]")
        print(f"  原始边界点范围: [{rim_pts_original.min():.3f}, {rim_pts_original.max():.3f}]")
    else:
        # 如果没有归一化，直接使用检测到的边界点
        rim_pts_original = rim_pts_normalized

    # 创建边界点云（使用原始坐标）
    boundary_pcd = o3d.geometry.PointCloud()
    boundary_pcd.points = o3d.utility.Vector3dVector(rim_pts_original)
    boundary_pcd.paint_uniform_color([1, 0, 0])

    # 可视化时显示原始点云（如果可用）
    if visualize:
        if hasattr(fragment, 'original_points') and fragment.original_points is not None:
            # 创建原始点云的可视化对象
            original_pcd = o3d.geometry.PointCloud()
            original_pcd.points = o3d.utility.Vector3dVector(fragment.original_points)
            original_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色

            # 如果有归一化的当前点云，也显示
            if hasattr(fragment, 'point_cloud') and fragment.point_cloud is not None:
                o3d.visualization.draw_geometries(
                    [original_pcd, boundary_pcd],
                    window_name=f"碎片{fragment.id} - Rim 边界（原始坐标系）",
                    width=800,
                    height=600,
                    zoom=0.5
                )
            else:
                o3d.visualization.draw_geometries(
                    [original_pcd, boundary_pcd],
                    window_name=f"碎片{fragment.id} - Rim 边界（原始坐标系）",
                    width=800,
                    height=600,
                    zoom=0.5
                )
        else:
            o3d.visualization.draw_geometries(
                [pcd, boundary_pcd],
                window_name=f"碎片{fragment.id} - Rim 边界",
                width=800,
                height=600
            )

    # 存储边界点（原始坐标系）
    fragment.boundary_pts = rim_pts_original
    fragment.boundary_pcd = boundary_pcd

    print(f"[边界检测] 碎片{fragment.id} rim 点数：{len(rim_pts_original)}")
    return boundary_pcd, rim_pts_original