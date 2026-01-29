# D:\ceramic_reconstruction\src\boundary\rim.py
import numpy as np
import open3d as o3d


def extract_rim_curve(fragment, n_bins=100, visualize=False):
    """
    提取陶瓷碎片的rim曲线（器型轮廓边缘）：PCA找主轴线→投影→极坐标拟合
    :param fragment: Fragment对象，需包含point_cloud属性（已归一化更佳）
    :param n_bins: 极坐标角度分箱数，默认100（轮廓越圆，分箱数可越大）
    :param visualize: 是否可视化rim曲线，默认False
    :return: rim曲线点数组np.ndarray | None，rim曲线点云o3d.PointCloud | None
    """
    # 容错1：无点云数据直接返回
    if fragment.point_cloud is None or len(fragment.point_cloud.points) == 0:
        print(f"[Rim提取] 碎片{fragment.id}无有效点云数据，跳过rim曲线提取")
        return None, None

    pcd = fragment.point_cloud
    pts = np.asarray(pcd.points)  # (N,3)
    # 容错2：点云数量过少
    if len(pts) < 100:
        print(f"[Rim提取] 碎片{fragment.id}点云数量不足（{len(pts)}<100），跳过")
        return None, None

    # 步骤1：PCA主成分分析，找到器型主轴线（如碗沿的中心轴线）
    pts_centered = pts - pts.mean(axis=0)  # 点云中心化
    _, _, Vt = np.linalg.svd(pts_centered)  # SVD分解求主成分
    main_axis = Vt[0]  # 第一主成分 = 器型主轴线（rim曲线的法向）

    # 步骤2：将点云投影到「垂直于主轴线的平面」（rim曲线所在平面）
    # 投影逻辑：去除主轴线方向的分量，保留平面分量
    proj_pts = pts_centered - np.outer(pts_centered @ main_axis, main_axis)  # (N,3)

    # 步骤3：极坐标转换，按角度分箱求平均半径（拟合光滑rim曲线）
    # 计算每个投影点的极角和极径
    theta = np.arctan2(proj_pts[:, 1], proj_pts[:, 0])  # 极角 (-π, π)
    r = np.linalg.norm(proj_pts[:, :2], axis=1)  # 极径（平面内到中心的距离）

    # 角度分箱，遍历每个箱求平均极径
    theta_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    rim_curve_pts = []
    for i in range(n_bins):
        # 筛选当前角度箱内的点
        bin_mask = (theta >= theta_bins[i]) & (theta < theta_bins[i + 1])
        bin_r = r[bin_mask]
        if len(bin_r) < 5:  # 每个箱至少5个点，避免噪声
            continue
        # 计算平均极径，还原为笛卡尔坐标
        avg_r = bin_r.mean()
        avg_theta = (theta_bins[i] + theta_bins[i + 1]) / 2
        x = avg_r * np.cos(avg_theta)
        y = avg_r * np.sin(avg_theta)
        rim_curve_pts.append([x, y, 0])  # z=0（投影平面）

    rim_curve_pts = np.array(rim_curve_pts)  # (M,3)
    # 容错3：拟合后rim曲线点过少
    if len(rim_curve_pts) < 20:
        print(f"[Rim提取] 碎片{fragment.id}拟合rim曲线点不足（{len(rim_curve_pts)}<20），跳过")
        return None, None

    # 步骤4：构建rim曲线点云（绿色，方便可视化）
    rim_pcd = o3d.geometry.PointCloud()
    rim_pcd.points = o3d.utility.Vector3dVector(rim_curve_pts)
    rim_pcd.paint_uniform_color([0, 1, 0])  # 绿色标记rim曲线
    # 生成rim曲线线集，可视化更直观
    rim_lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        rim_pcd, rim_pcd, np.array([[i, i + 1] for i in range(len(rim_curve_pts) - 1)])
    )
    rim_lines.paint_uniform_color([0, 1, 0])

    # 可视化（原始点云+绿色rim曲线）
    if visualize:
        o3d.visualization.draw_geometries(
            [pcd, rim_pcd, rim_lines],
            window_name=f"碎片{fragment.id} - Rim曲线提取结果",
            width=800, height=600
        )

    # 更新Fragment对象属性
    fragment.rim_curve = rim_curve_pts  # rim曲线点数组
    fragment.rim_pcd = rim_pcd  # rim曲线点云
    fragment.rim_lines = rim_lines  # rim曲线线集
    fragment.main_axis = main_axis  # 器型主轴线（复用至其他模块）

    print(f"[Rim提取] 碎片{fragment.id}完成，拟合出{len(rim_curve_pts)}个rim曲线点")
    return rim_curve_pts, rim_pcd