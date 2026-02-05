# D:\ceramic_reconstruction\src\boundary\rim.py
import numpy as np
import open3d as o3d


def extract_rim_curve(fragment, n_samples=200, visualize=False):
    """
    从 rim 边界点拟合有序 rim 曲线
    :param fragment: Fragment对象，需包含 boundary_pts
    :param n_samples: 重采样点数
    """

    if fragment.boundary_pts is None or len(fragment.boundary_pts) < 50:
        print(f"[Rim提取] 碎片{fragment.id} rim 点不足")
        return None, None

    rim_pts = fragment.boundary_pts
    center = rim_pts.mean(axis=0)

    # ===== 1. PCA 求 rim 平面 =====
    pts_centered = rim_pts - center
    _, _, Vt = np.linalg.svd(pts_centered)
    normal = Vt[-1]
    x_axis = Vt[0]
    y_axis = np.cross(normal, x_axis)

    # ===== 2. 投影到 rim 平面，极角排序 =====
    x = pts_centered @ x_axis
    y = pts_centered @ y_axis
    theta = np.arctan2(y, x)

    order = np.argsort(theta)
    rim_sorted = rim_pts[order]

    # ===== 3. 等弧长重采样 =====
    diffs = np.linalg.norm(np.diff(rim_sorted, axis=0), axis=1)
    arc = np.insert(np.cumsum(diffs), 0, 0)
    arc /= arc[-1]

    target = np.linspace(0, 1, n_samples)
    rim_resampled = np.vstack([
        np.interp(target, arc, rim_sorted[:, i])
        for i in range(3)
    ]).T

    rim_pcd = o3d.geometry.PointCloud()
    rim_pcd.points = o3d.utility.Vector3dVector(rim_resampled)
    rim_pcd.paint_uniform_color([0, 1, 0])

    rim_lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        rim_pcd, rim_pcd,
        np.array([[i, i + 1] for i in range(n_samples - 1)])
    )
    rim_lines.paint_uniform_color([0, 1, 0])

    if visualize:
        o3d.visualization.draw_geometries(
            [rim_pcd, rim_lines],
            window_name=f"碎片{fragment.id} - Rim 曲线",
            width=800,
            height=600
        )

    fragment.rim_curve = rim_resampled
    fragment.rim_pcd = rim_pcd
    fragment.rim_lines = rim_lines

    print(f"[Rim提取] 碎片{fragment.id} rim 曲线点数：{len(rim_resampled)}")
    return rim_resampled, rim_pcd
