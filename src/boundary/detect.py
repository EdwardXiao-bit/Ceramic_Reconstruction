import numpy as np
import open3d as o3d
from collections import defaultdict


def detect_boundary_robust(fragment,
                           smooth_iter=5,
                           angle_thresh=35.0,  # 主阈值（用于找强边界）
                           low_angle_thresh=20.0,  # 副阈值（用于找圆润边界）
                           min_cluster_size=20,
                           visualize=False):
    """
    【双阈值 + 连通生长版】
    解决“一边锐利、一边圆润”导致只检出一半的问题。
    """
    if fragment.mesh is None:
        return None, None

    # 1. 预处理：轻微平滑 (不要平滑太多，否则圆润边更难找)
    mesh_compute = o3d.geometry.TriangleMesh(fragment.mesh)
    mesh_compute = mesh_compute.filter_smooth_laplacian(
        number_of_iterations=smooth_iter, lambda_filter=0.5)
    mesh_compute.compute_triangle_normals()

    triangles = np.asarray(mesh_compute.triangles)
    triangle_normals = np.asarray(mesh_compute.triangle_normals)
    vertices = np.asarray(mesh_compute.vertices)

    # 2. 计算所有边的“锐利度” (1 - dot_product)
    # sharp_score 越高越锐利。范围 [0, 2]。
    # 垂直时 dot=0 -> score=1.0; 钝角120度 dot=-0.5 -> score=1.5
    edge_scores = {}
    edge_to_triangles = defaultdict(list)

    for i, tri in enumerate(triangles):
        edges = [tuple(sorted((tri[0], tri[1]))), tuple(sorted((tri[1], tri[2]))), tuple(sorted((tri[2], tri[0])))]
        for edge in edges:
            edge_to_triangles[edge].append(i)

    # 计算分数的阈值
    high_score_thresh = 1.0 - np.cos(np.deg2rad(angle_thresh))
    low_score_thresh = 1.0 - np.cos(np.deg2rad(low_angle_thresh))

    strong_edges = set()
    weak_edges = set()

    for edge, tri_indices in edge_to_triangles.items():
        score = 0.0
        if len(tri_indices) == 1:  # 边界边 (无限锐利)
            score = 2.0
        elif len(tri_indices) == 2:
            n1 = triangle_normals[tri_indices[0]]
            n2 = triangle_normals[tri_indices[1]]
            dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
            score = 1.0 - dot

        if score >= high_score_thresh:
            strong_edges.add(edge)
        elif score >= low_score_thresh:
            weak_edges.add(edge)

    # 3. 结果合并：强边缘 + (连接到强边缘的)弱边缘
    # 由于拓扑复杂，这里简化策略：
    # 只要弱边缘点在空间上靠近强边缘点（断裂面厚度范围内），就保留。

    strong_indices = set()
    for e in strong_edges:
        strong_indices.add(e[0]);
        strong_indices.add(e[1])

    weak_indices = set()
    for e in weak_edges:
        weak_indices.add(e[0]);
        weak_indices.add(e[1])

    # 如果完全没找到强边缘，就降级用弱边缘
    if len(strong_indices) == 0:
        final_indices_set = weak_indices
    else:
        # 这里用一种简单有效的策略：全盘接受弱边缘
        # 因为我们后续有 DBSCAN 聚类去噪，孤立的弱边缘（表面噪点）会被去掉
        # 而位于断裂面边缘的弱边缘（圆润边）通常会连成一片
        final_indices_set = strong_indices.union(weak_indices)

    if len(final_indices_set) == 0:
        return None, None

    candidate_indices = np.array(list(final_indices_set), dtype=int)

    # 4. 强力去噪 (DBSCAN)
    # 这里的关键是：真边界（无论是锐利还是圆润）是连贯的长线条
    # 表面噪点是零散的
    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(vertices[candidate_indices])

    # 估算 eps: 平均边长的 2-3 倍
    # 假设模型单位是 mm，且非常精细，这里可能需要手动调节
    # 如果你知道模型的大致尺寸，可以直接写死，比如 0.5 或 1.0
    avg_edge_len = np.linalg.norm(vertices[triangles[0][0]] - vertices[triangles[0][1]])
    cluster_eps = avg_edge_len * 3.0

    labels = np.array(pcd_temp.cluster_dbscan(eps=cluster_eps, min_points=10, print_progress=False))

    if len(labels) == 0: return None, None

    final_indices_list = []
    max_label = labels.max()

    for label in range(max_label + 1):
        mask = (labels == label)
        # 只要簇足够大，就保留
        if np.sum(mask) > min_cluster_size:
            final_indices_list.append(candidate_indices[mask])

    if not final_indices_list:
        return None, None

    final_indices = np.concatenate(final_indices_list)

    # 5. 构建结果
    original_vertices = np.asarray(fragment.mesh.vertices)
    boundary_pts = original_vertices[final_indices]

    boundary_pcd = o3d.geometry.PointCloud()
    boundary_pcd.points = o3d.utility.Vector3dVector(boundary_pts)
    boundary_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    fragment.boundary_points = boundary_pcd
    fragment.boundary_indices = final_indices
    # 兼容性：同时设置boundary_pts属性
    fragment.boundary_pts = boundary_pts
    # 检查当前点云的点数是否与 Mesh 顶点数一致
    current_pcd_points = np.asarray(fragment.point_cloud.points) if fragment.point_cloud else []

    if len(original_vertices) != len(current_pcd_points):
        print(f"[数据同步] Mesh顶点数({len(original_vertices)}) != 点云数({len(current_pcd_points)})。正在同步...")

        # 用 Mesh 的顶点重新生成一个 PointCloud
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(original_vertices)

        # 如果 Mesh 有法线，也同步过来（这对 Patch 提取很重要）
        if fragment.mesh.has_vertex_normals():
            new_pcd.normals = fragment.mesh.vertex_normals
        elif fragment.mesh.has_triangle_normals():
            fragment.mesh.compute_vertex_normals()
            new_pcd.normals = fragment.mesh.vertex_normals

        # 强制替换
        fragment.point_cloud = new_pcd
        print(f"[数据同步] Fragment点云已更新为 Mesh 顶点。")
        print(f"[边界检测] 完成。去噪后剩余点数: {len(final_indices)}")
    if visualize:
        vis_mesh = o3d.geometry.TriangleMesh(fragment.mesh)
        vis_mesh.compute_vertex_normals()
        vis_mesh.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([vis_mesh, boundary_pcd], window_name="Dual Threshold")

    return boundary_pcd, final_indices

# 为了兼容旧代码的导入，保留旧函数名，但你可以不用它们
def detect_boundary(*args, **kwargs):
    print("Warning: Calling deprecated detect_boundary")
    return None, None


def detect_sharp_edges(*args, **kwargs):
    # 简单的转接，如果你不想改 run.py 的 import
    return detect_boundary_robust(*args, **kwargs)
