import numpy as np
import open3d as o3d
from collections import defaultdict


def detect_boundary(fragment):
    """
    修正：接收Fragment对象，检测其网格的边界点云
    :param fragment: Fragment对象（包含mesh）
    :return: 边界点云
    """
    if fragment.mesh is None:
        print(f"碎片{fragment.id}无网格数据，跳边界检测")
        return None

    mesh = fragment.mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    edge_count = defaultdict(int)
    for tri in triangles:
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edge_count[tuple(sorted(e))] += 1

    boundary_edges = [e for e, c in edge_count.items() if c == 1]

    boundary_pts = np.unique(
        np.array([vertices[i] for e in boundary_edges for i in e]),
        axis=0
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(boundary_pts)

    # 保存边界点云到Fragment对象
    fragment.boundary_patch = pcd
    return pcd