# D:\ceramic_reconstruction\src\boundary\detect.py
import numpy as np
import open3d as o3d
from collections import defaultdict

def detect_boundary(fragment, visualize=False, curvature_thresh=0.1):
    """
    陶瓷碎片边界检测：优先网格拓扑法，无网格则用点云几何法（曲率/法向突变）
    :param fragment: Fragment对象，需包含point_cloud（必选），mesh（可选）
    :param visualize: 是否可视化，默认False
    :param curvature_thresh: 点云边界检测的曲率阈值，越大提取的边界越少，默认0.1
    :return: 边界点云o3d.PointCloud | None，边界点数组np.ndarray | None
    """
    # 容错：无点云直接返回（网格也依赖点云，且点云是必选属性）
    if fragment.point_cloud is None or len(fragment.point_cloud.points) == 0:
        print(f"[边界检测] 碎片{fragment.id}无有效点云数据，跳过边界检测")
        return None, None

    pcd = fragment.point_cloud
    boundary_pcd = None
    boundary_pts = None

    # ===== 模式1：网格拓扑法（优先，精度更高）=====
    if fragment.mesh is not None and len(fragment.mesh.triangles) > 0:
        print(f"[边界检测] 碎片{fragment.id}使用网格拓扑法提取边界")
        mesh = fragment.mesh
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # 统计边的面数，筛选边界边
        edge_count = defaultdict(int)
        for tri in triangles:
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for e in edges:
                edge_count[tuple(sorted(e))] += 1

        boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
        boundary_vert_indices = np.unique(np.array([i for e in boundary_edges for i in e]))

        if len(boundary_vert_indices) > 0:
            boundary_pts = vertices[boundary_vert_indices]
            boundary_pcd = o3d.geometry.PointCloud()
            boundary_pcd.points = o3d.utility.Vector3dVector(boundary_pts)
            boundary_pcd.paint_uniform_color([1, 0, 0])  # 红色

    # ===== 模式2：点云几何法（无网格时使用，基于曲率）=====
    if boundary_pcd is None or len(boundary_pts) == 0:
        print(f"[边界检测] 碎片{fragment.id}无有效网格，使用点云几何法提取边界")
        # 步骤1：计算点云法向和曲率（Open3D内置）
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        pcd.compute_curvatures()  # 计算每个点的曲率

        # 步骤2：提取曲率大于阈值的点作为边界点（曲率大=表面变化剧烈=断裂边界）
        curvatures = np.asarray(pcd.curvatures)  # 每个点的曲率值 (N,)
        boundary_vert_indices = np.where(curvatures > curvature_thresh)[0]

        if len(boundary_vert_indices) == 0:
            print(f"[边界检测] 碎片{fragment.id}点云曲率均低于阈值{curvature_thresh}，未提取到边界")
            return None, None

        # 步骤3：构建边界点云
        pcd_pts = np.asarray(pcd.points)
        boundary_pts = pcd_pts[boundary_vert_indices]
        boundary_pcd = o3d.geometry.PointCloud()
        boundary_pcd.points = o3d.utility.Vector3dVector(boundary_pts)
        boundary_pcd.paint_uniform_color([1, 0, 0])  # 红色

    # 生成边界线集（可视化用）
    boundary_lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        boundary_pcd, boundary_pcd, np.array([[i, i+1] for i in range(len(boundary_pts)-1)])
    )
    boundary_lines.paint_uniform_color([1, 0, 0])

    # 可视化（点云+边界点+边界线，无网格则只显示点云）
    if visualize:
        vis_list = [pcd, boundary_pcd, boundary_lines]
        if fragment.mesh is not None:
            vis_list.insert(0, fragment.mesh)
        o3d.visualization.draw_geometries(
            vis_list,
            window_name=f"碎片{fragment.id} - 边界检测结果",
            width=800, height=600
        )

    # 更新Fragment属性
    fragment.boundary_pcd = boundary_pcd
    fragment.boundary_pts = boundary_pts
    fragment.boundary_lines = boundary_lines

    print(f"[边界检测] 碎片{fragment.id}完成，提取到{len(boundary_pts)}个边界点")
    return boundary_pcd, boundary_pts