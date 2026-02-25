# D:\ceramic_reconstruction\src\boundary\rim.py
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


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
        print(f"[Rim提取] 碎片{fragment.id} rim 点不足")
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
    rim_pcd.paint_uniform_color([0.2, 0.6, 0.2])  # 更柔和的绿色

    # 线段连接为开曲线；如果是闭合曲线，可在最后再连回起点
    line_indices = np.array([[i, i + 1] for i in range(n_samples - 1)], dtype=np.int32)
    # 若需要闭合，可使用：
    # line_indices = np.array([[i, i + 1] for i in range(n_samples - 1)] + [[n_samples - 1, 0]], dtype=np.int32)

    rim_lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        rim_pcd, rim_pcd, line_indices
    )
    rim_lines.paint_uniform_color([0.2, 0.6, 0.2])  # 更柔和的绿色

    fragment.rim_curve = rim_resampled
    fragment.rim_pcd = rim_pcd
    fragment.rim_lines = rim_lines

    print(f"[Rim提取] 碎片{fragment.id} rim 曲线点数：{len(rim_resampled)}")


    return rim_resampled, rim_pcd


def extract_centerline_rim_from_boundaries(fragment, n_samples=200, visualize=False):
    """
    基于两条边界线提取中心线rim曲线
    
    :param fragment: Fragment对象，需要包含boundary_points和section_patch
    :param n_samples: 重采样点数
    :param visualize: 是否可视化
    :return: 中心线rim曲线点集, rim点云
    """
    
    # 检查必要数据是否存在
    if not hasattr(fragment, 'boundary_points') or fragment.boundary_points is None:
        print(f"[中心线Rim] 碎片{fragment.id} 缺少边界点数据")
        return None, None
    
    if not hasattr(fragment, 'section_patch') or fragment.section_patch is None:
        print(f"[中心线Rim] 碎片{fragment.id} 缺少断面patch数据")
        return None, None
    
    print(f"[中心线Rim] 开始提取碎片{fragment.id}的中心线rim曲线...")
    
    # 获取边界点和patch点
    boundary_pcd = fragment.boundary_points
    boundary_pts = np.asarray(boundary_pcd.points)
    patch_pcd = fragment.section_patch
    patch_pts = np.asarray(patch_pcd.points)
    
    # 分离两条边界线
    print("[中心线Rim] 分离两条边界线...")
    boundary_lines = _separate_boundary_lines(boundary_pts, eps=0.02, min_samples=10)
    
    if len(boundary_lines) < 2:
        print(f"[中心线Rim] 未能分离出两条边界线，找到{len(boundary_lines)}条线")
        return None, None
    
    # 选择最长的两条边界线
    boundary_lines.sort(key=len, reverse=True)
    line1_pts = boundary_lines[0]
    line2_pts = boundary_lines[1]
    
    print(f"[中心线Rim] 找到两条边界线，长度分别为: {len(line1_pts)}, {len(line2_pts)}")
    
    # 计算两条边界线之间的对应点对
    print("[中心线Rim] 计算边界线对应点...")
    correspondence_pairs = _find_boundary_correspondences(line1_pts, line2_pts, patch_pts)
    
    if len(correspondence_pairs) < 10:
        print(f"[中心线Rim] 对应点对不足: {len(correspondence_pairs)}")
        return None, None
    
    # 通过对应点对计算中心线点
    print("[中心线Rim] 计算中心线点...")
    centerline_points = []
    for pt1, pt2 in correspondence_pairs:
        center_pt = (pt1 + pt2) / 2.0  # 中点即为中心点
        centerline_points.append(center_pt)
    
    centerline_points = np.array(centerline_points)
    
    # 对中心线点进行排序和重采样
    print("[中心线Rim] 排序和重采样中心线...")
    ordered_centerline = _order_and_resample_centerline(centerline_points, n_samples)
    
    # 创建结果对象
    rim_pcd = o3d.geometry.PointCloud()
    rim_pcd.points = o3d.utility.Vector3dVector(ordered_centerline)
    rim_pcd.paint_uniform_color([1, 0.5, 0])  # 橙色表示中心线rim
    
    rim_lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        rim_pcd, rim_pcd,
        np.array([[i, i + 1] for i in range(n_samples - 1)])
    )
    rim_lines.paint_uniform_color([1, 0.5, 0])
    
    # 可视化
    if visualize:
        # 创建边界线可视化
        line1_pcd = o3d.geometry.PointCloud()
        line1_pcd.points = o3d.utility.Vector3dVector(line1_pts)
        line1_pcd.paint_uniform_color([1, 0, 0])  # 红色
        
        line2_pcd = o3d.geometry.PointCloud()
        line2_pcd.points = o3d.utility.Vector3dVector(line2_pts)
        line2_pcd.paint_uniform_color([0, 0, 1])  # 蓝色
        
        # 显示原始点云、边界线和中心线
        geometries = [fragment.point_cloud, line1_pcd, line2_pcd, rim_pcd, rim_lines]
        
        # 如果有patch，也显示出来
        if hasattr(fragment, 'section_patch') and fragment.section_patch is not None:
            patch_vis = o3d.geometry.PointCloud(fragment.section_patch.points)
            patch_vis.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色
            geometries.append(patch_vis)
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"碎片{fragment.id} - 中心线Rim曲线提取结果",
            width=1200,
            height=900
        )
    
    # 更新fragment属性
    fragment.centerline_rim_curve = ordered_centerline
    fragment.centerline_rim_pcd = rim_pcd
    fragment.centerline_rim_lines = rim_lines
    
    print(f"[中心线Rim] 提取完成，中心线rim曲线点数: {len(ordered_centerline)}")
    return ordered_centerline, rim_pcd


def _separate_boundary_lines(boundary_points, eps=0.02, min_samples=10):
    """
    使用DBSCAN聚类将边界点分离成不同的边界线
    """
    from sklearn.cluster import DBSCAN
    
    # 使用DBSCAN聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(boundary_points)
    labels = clustering.labels_
    
    # 分离不同簇的点
    unique_labels = np.unique(labels)
    boundary_lines = []
    
    for label in unique_labels:
        if label != -1:  # -1表示噪声点
            line_points = boundary_points[labels == label]
            if len(line_points) > 5:  # 有效边界线：点数>5
                boundary_lines.append(line_points)
    
    return boundary_lines


def _find_boundary_correspondences(line1_pts, line2_pts, patch_pts, k_neighbors=10):
    """
    在两条边界线间寻找对应点对
    """
    # 构建patch的KD树用于快速近邻搜索
    patch_tree = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree')
    patch_tree.fit(patch_pts)
    
    correspondence_pairs = []
    
    # 对第一条边界线的每个点，寻找第二条边界线上的对应点
    for pt1 in line1_pts:
        # 在patch中找到pt1的近邻点
        distances, indices = patch_tree.kneighbors([pt1], n_neighbors=k_neighbors)
        neighbor_pts = patch_pts[indices[0]]
        
        # 在第二条边界线上寻找最近的对应点
        distances_to_line2 = np.linalg.norm(line2_pts - pt1, axis=1)
        closest_idx = np.argmin(distances_to_line2)
        pt2 = line2_pts[closest_idx]
        
        # 检查两点间是否有patch点连接（避免跨越太远）
        midpoint = (pt1 + pt2) / 2.0
        dist_to_midpoint = np.linalg.norm(neighbor_pts - midpoint, axis=1)
        min_dist = np.min(dist_to_midpoint)
        
        # 如果中点附近有足够的patch点，则认为是一对有效对应点
        if min_dist < 0.05:  # 阈值可根据实际情况调整
            correspondence_pairs.append((pt1, pt2))
    
    return correspondence_pairs


def _order_and_resample_centerline(centerline_points, n_samples):
    """
    对中心线点进行排序和重采样
    """
    if len(centerline_points) < 3:
        return centerline_points
    
    # 使用PCA找到主方向进行排序
    center = centerline_points.mean(axis=0)
    centered_pts = centerline_points - center
    
    # PCA
    _, _, Vt = np.linalg.svd(centered_pts)
    main_axis = Vt[0]
    
    # 投影到主轴方向进行排序
    projections = centered_pts @ main_axis
    sort_indices = np.argsort(projections)
    ordered_points = centerline_points[sort_indices]
    
    # 等弧长重采样
    diffs = np.linalg.norm(np.diff(ordered_points, axis=0), axis=1)
    if len(diffs) == 0:
        return ordered_points
    
    arc_length = np.insert(np.cumsum(diffs), 0, 0)
    arc_length_normalized = arc_length / arc_length[-1]
    
    # 等间距采样
    target_arc = np.linspace(0, 1, n_samples)
    
    resampled_points = np.zeros((n_samples, 3))
    for i in range(3):
        resampled_points[:, i] = np.interp(target_arc, arc_length_normalized, ordered_points[:, i])
    
    return resampled_points