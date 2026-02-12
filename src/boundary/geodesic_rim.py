# D:\ceramic_reconstruction\src\boundary\geodesic_rim.py
"""
基于测地线的rim曲线提取模块
实现技术文档中要求的geodesic shortest path方法
"""
import numpy as np
import open3d as o3d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors


def extract_geodesic_rim_curve(fragment, patch_pcd=None, n_samples=200, visualize=False):
    """
    基于测地线最短路径的rim曲线提取（符合技术文档要求）
    :param fragment: Fragment对象
    :param patch_pcd: patch点云（如果为None，则从fragment.section_patch获取）
    :param n_samples: 重采样点数
    :param visualize: 是否可视化
    :return: rim曲线点集, rim点云
    """
    # 获取patch点云
    if patch_pcd is None:
        if not hasattr(fragment, 'section_patch') or fragment.section_patch is None:
            print("[测地线Rim] 未找到patch点云")
            return None, None
        patch_pcd = fragment.section_patch

    patch_points = np.asarray(patch_pcd.points)
    n_points = len(patch_points)

    if n_points < 10:
        print("[测地线Rim] patch点数不足")
        return None, None

    print(f"[测地线Rim] 开始处理 {n_points} 个patch点")

    # 步骤1: 构建k近邻图
    k = min(10, n_points - 1)  # 邻居数
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(patch_points)
    distances, indices = nbrs.kneighbors(patch_points)

    # 步骤2: 构建稀疏距离矩阵
    rows, cols, data = [], [], []
    for i in range(n_points):
        for j in range(k):
            neighbor_idx = indices[i, j]
            if i != neighbor_idx:  # 排除自己到自己的连接
                rows.append(i)
                cols.append(neighbor_idx)
                # 使用欧几里得距离作为边权重
                data.append(distances[i, j])

    # 创建稀疏矩阵
    n_points = len(patch_points)
    graph = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

    # 步骤3: 找到边界点作为起点和终点
    boundary_indices = _find_patch_boundary_indices(patch_pcd)
    
    if len(boundary_indices) < 2:
        print("[测地线Rim] 未找到足够的边界点")
        return None, None

    # 选择两个最远的边界点作为起点和终点
    start_idx, end_idx = _find_farthest_boundary_points(patch_points, boundary_indices)

    # 步骤4: 计算测地线最短路径
    print(f"[测地线Rim] 计算从点{start_idx}到点{end_idx}的测地线路径...")
    
    # 使用Dijkstra算法计算最短路径
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, 
                                            indices=start_idx, return_predecessors=True)
    
    # 重构路径
    path = _reconstruct_path(predecessors, start_idx, end_idx)
    
    if len(path) < 2:
        print("[测地线Rim] 无法找到有效路径")
        return None, None

    # 步骤5: 提取路径上的点
    rim_points = patch_points[path]
    
    # 步骤6: 重采样获得均匀分布
    rim_resampled = _resample_curve(rim_points, n_samples)

    # 步骤7: 附加几何属性
    geometric_attributes = _compute_geometric_attributes(rim_resampled, patch_pcd)

    # 创建结果对象
    rim_pcd = o3d.geometry.PointCloud()
    rim_pcd.points = o3d.utility.Vector3dVector(rim_resampled)
    rim_pcd.paint_uniform_color([1, 0.5, 0])  # 橙色表示测地线rim

    rim_lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        rim_pcd, rim_pcd,
        np.array([[i, i + 1] for i in range(n_samples - 1)])
    )
    rim_lines.paint_uniform_color([1, 0.5, 0])

    # 可视化
    if visualize:
        # 显示patch、边界点和rim曲线
        boundary_pcd = o3d.geometry.PointCloud()
        boundary_points = patch_points[boundary_indices]
        boundary_pcd.points = o3d.utility.Vector3dVector(boundary_points)
        boundary_pcd.paint_uniform_color([1, 0, 0])  # 红色边界点

        start_end_pcd = o3d.geometry.PointCloud()
        start_end_points = patch_points[[start_idx, end_idx]]
        start_end_pcd.points = o3d.utility.Vector3dVector(start_end_points)
        start_end_pcd.paint_uniform_color([0, 1, 0])  # 绿色起终点

        o3d.visualization.draw_geometries(
            [patch_pcd, boundary_pcd, start_end_pcd, rim_pcd, rim_lines],
            window_name="测地线Rim曲线提取结果",
            width=1200, height=900
        )

    # 更新fragment属性
    fragment.geodesic_rim_curve = rim_resampled
    fragment.geodesic_rim_pcd = rim_pcd
    fragment.geodesic_rim_lines = rim_lines
    fragment.geodesic_attributes = geometric_attributes

    print(f"[测地线Rim] 提取完成，rim曲线点数: {len(rim_resampled)}")
    print(f"[测地线Rim] 几何属性维度: {geometric_attributes.shape}")
    
    return rim_resampled, rim_pcd


def _find_patch_boundary_indices(patch_pcd, boundary_ratio=0.1):
    """
    找到patch的边界点索引
    :param patch_pcd: patch点云
    :param boundary_ratio: 边界点比例
    :return: 边界点索引数组
    """
    points = np.asarray(patch_pcd.points)
    n_points = len(points)
    
    # 计算法向量
    patch_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    normals = np.asarray(patch_pcd.normals)
    
    # 计算曲率（法向变化）
    tree = o3d.geometry.KDTreeFlann(patch_pcd)
    curvature = np.zeros(n_points)
    
    for i in range(n_points):
        _, idx, _ = tree.search_knn_vector_3d(points[i], 20)
        nbr_normals = normals[idx]
        mean_normal = nbr_normals.mean(axis=0)
        curvature[i] = np.mean(np.linalg.norm(nbr_normals - mean_normal, axis=1))
    
    # 选择高曲率点作为边界候选
    num_boundary = max(int(n_points * boundary_ratio), 10)
    boundary_indices = np.argsort(curvature)[-num_boundary:]
    
    return boundary_indices


def _find_farthest_boundary_points(points, boundary_indices):
    """
    找到边界点中最远的两个点
    :param points: 所有点坐标
    :param boundary_indices: 边界点索引
    :return: 起点索引, 终点索引
    """
    boundary_points = points[boundary_indices]
    n_boundary = len(boundary_points)
    
    max_distance = 0
    start_idx = 0
    end_idx = 1
    
    # 找到最远的两点
    for i in range(n_boundary):
        for j in range(i + 1, n_boundary):
            distance = np.linalg.norm(boundary_points[i] - boundary_points[j])
            if distance > max_distance:
                max_distance = distance
                start_idx = boundary_indices[i]
                end_idx = boundary_indices[j]
    
    return start_idx, end_idx


def _reconstruct_path(predecessors, start_idx, end_idx):
    """
    从前驱矩阵重构路径
    :param predecessors: 前驱矩阵
    :param start_idx: 起点索引
    :param end_idx: 终点索引
    :return: 路径索引列表
    """
    path = [end_idx]
    current = end_idx
    
    while current != start_idx and predecessors[current] != -9999:
        current = predecessors[current]
        path.append(current)
        if current == end_idx:  # 避免循环
            break
    
    return path[::-1]  # 反转得到正确顺序


def _resample_curve(points, n_samples):
    """
    对曲线进行等弧长重采样
    :param points: 原始点序列
    :param n_samples: 目标点数
    :return: 重采样后的点序列
    """
    if len(points) < 2:
        return points
    
    # 计算累积弧长
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_length = np.insert(np.cumsum(diffs), 0, 0)
    arc_length_normalized = arc_length / arc_length[-1]
    
    # 等间距采样
    target_arc = np.linspace(0, 1, n_samples)
    
    resampled_points = np.zeros((n_samples, 3))
    for i in range(3):  # 对每个坐标分量插值
        resampled_points[:, i] = np.interp(target_arc, arc_length_normalized, points[:, i])
    
    return resampled_points


def _compute_geometric_attributes(curve_points, patch_pcd):
    """
    计算rim曲线的几何属性
    :param curve_points: rim曲线点
    :param patch_pcd: 原始patch点云
    :return: 几何属性矩阵 (n_points, n_attributes)
    """
    n_points = len(curve_points)
    attributes = np.zeros((n_points, 6))  # 6个几何属性
    
    # 为每个rim点计算属性
    tree = o3d.geometry.KDTreeFlann(patch_pcd)
    
    for i, point in enumerate(curve_points):
        # 查找近邻点
        _, idx, distances = tree.search_knn_vector_3d(point, 20)
        neighbor_points = np.asarray(patch_pcd.points)[idx]
        
        # 1. 到patch边界的平均距离
        boundary_distances = distances
        attributes[i, 0] = np.mean(boundary_distances)
        
        # 2. 局部密度（近邻点数）
        attributes[i, 1] = len(idx)
        
        # 3. 曲率估计
        if len(neighbor_points) > 3:
            # 简单的平面拟合残差作为曲率近似
            center = neighbor_points.mean(axis=0)
            centered_points = neighbor_points - center
            try:
                _, singular_values, _ = np.linalg.svd(centered_points)
                # 最小奇异值反映偏离平面的程度
                attributes[i, 2] = singular_values[-1] / (singular_values[0] + 1e-8)
            except:
                attributes[i, 2] = 0.0
        
        # 4. 到质心的距离
        patch_center = np.asarray(patch_pcd.points).mean(axis=0)
        attributes[i, 3] = np.linalg.norm(point - patch_center)
        
        # 5. 高度（z坐标相对patch最小值）
        patch_z_min = np.asarray(patch_pcd.points)[:, 2].min()
        attributes[i, 4] = point[2] - patch_z_min
        
        # 6. 切线方向的一致性
        if i > 0 and i < n_points - 1:
            tangent_prev = curve_points[i] - curve_points[i-1]
            tangent_next = curve_points[i+1] - curve_points[i]
            tangent_prev = tangent_prev / (np.linalg.norm(tangent_prev) + 1e-8)
            tangent_next = tangent_next / (np.linalg.norm(tangent_next) + 1e-8)
            attributes[i, 5] = np.dot(tangent_prev, tangent_next)
        else:
            attributes[i, 5] = 1.0
    
    return attributes