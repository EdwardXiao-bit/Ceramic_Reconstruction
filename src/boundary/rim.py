import numpy as np
import open3d as o3d


def extract_rim_curve(fragment, n_samples=200, visualize=False):
    """
    从 rim 边界点拟合有序、平滑的 rim 曲线
    :param fragment: Fragment对象，需包含 boundary_pts (N x 3)
    :param n_samples: 重采样点数
    :param visualize: 是否可视化（保留接口，不在内部强依赖）
    """

    # ========== 内部小工具函数（低耦合） ==========

    def _moving_average_smooth(points, window_size=5, cyclic=True):
        """
        对 3D 点序列做简单滑动平均平滑。
        :param points: (N, 3)
        :param window_size: 奇数窗口，越大越平滑，过大会损失细节
        :param cyclic: 是否将曲线视作闭合，进行首尾循环平滑
        """
        if window_size < 3 or window_size % 2 == 0 or len(points) < window_size:
            return points

        half = window_size // 2
        pts = points

        if cyclic:
            # 首尾拼接，实现循环平滑
            extended = np.vstack([pts[-half:], pts, pts[:half]])
        else:
            # 仅在两端做边缘填充
            extended = np.vstack([np.repeat(pts[0:1], half, axis=0),
                                  pts,
                                  np.repeat(pts[-1:], half, axis=0)])

        kernel = np.ones(window_size, dtype=np.float64) / window_size
        smoothed = np.zeros_like(pts)

        # 对每一维独立做 1D 卷积
        for d in range(3):
            conv = np.convolve(extended[:, d], kernel, mode='valid')
            smoothed[:, d] = conv

        return smoothed

    def _resample_equal_arc(points, n_samples):
        """
        等弧长重采样（假定 points 已经是按顺序的）。
        :param points: (N, 3)
        :param n_samples: 目标采样点数
        """
        if len(points) < 2:
            return points

        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.insert(np.cumsum(diffs), 0, 0.0)
        if arc[-1] == 0:
            # 所有点重合，直接返回重复点
            return np.repeat(points[:1], n_samples, axis=0)

        arc /= arc[-1]
        target = np.linspace(0.0, 1.0, n_samples)

        # 按维度插值
        resampled = np.vstack([
            np.interp(target, arc, points[:, dim])
            for dim in range(3)
        ]).T
        return resampled

    # ========== 0. 基本检查 ==========
    if fragment.boundary_pts is None or len(fragment.boundary_pts) < 50:
        print("[Rim提取] 碎片{fragment.id} rim 点不足")
        return None, None

    rim_pts = np.asarray(fragment.boundary_pts, dtype=np.float64)
    center = rim_pts.mean(axis=0)

    # （可选）在 PCA 之前做一次轻度平滑，降低噪声对法向估计的影响
    rim_pts_smooth_for_plane = _moving_average_smooth(rim_pts, window_size=5, cyclic=True)

    # ========== 1. PCA 求 rim 平面 ==========
    pts_centered = rim_pts_smooth_for_plane - center
    _, _, Vt = np.linalg.svd(pts_centered)
    normal = Vt[-1]
    x_axis = Vt[0]
    y_axis = np.cross(normal, x_axis)

    # 归一化坐标轴，避免数值问题
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    normal /= np.linalg.norm(normal)

    # ========== 2. 投影到 rim 平面，极角排序 ==========
    pts_centered_full = rim_pts - center
    x = pts_centered_full @ x_axis
    y = pts_centered_full @ y_axis
    theta = np.arctan2(y, x)

    order = np.argsort(theta)
    rim_sorted = rim_pts[order]

    rim_sorted_smooth = rim_sorted
    # ========== 3. 等弧长重采样 ==========
    rim_resampled = _resample_equal_arc(rim_sorted_smooth, n_samples)

    # 对重采样结果再做轻度平滑，进一步抑制高频噪声
    rim_resampled = _moving_average_smooth(rim_resampled, window_size=5, cyclic=True)

    # ========== 4. 构建 Open3D 几何 ==========
    rim_pcd = o3d.geometry.PointCloud()
    rim_pcd.points = o3d.utility.Vector3dVector(rim_resampled)
    rim_pcd.paint_uniform_color([0, 1, 0])

    # 线段连接为开曲线；如果是闭合曲线，可在最后再连回起点
    line_indices = np.array([[i, i + 1] for i in range(n_samples - 1)], dtype=np.int32)
    # 若需要闭合，可使用：
    # line_indices = np.array([[i, i + 1] for i in range(n_samples - 1)] + [[n_samples - 1, 0]], dtype=np.int32)

    rim_lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        rim_pcd, rim_pcd, line_indices
    )
    rim_lines.paint_uniform_color([0, 1, 0])

    fragment.rim_curve = rim_resampled
    fragment.rim_pcd = rim_pcd
    fragment.rim_lines = rim_lines

    print("[Rim提取] 碎片{fragment.id} rim 曲线点数：{len(rim_resampled)}")


    return rim_resampled, rim_pcd
