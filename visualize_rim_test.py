"""
Rim边界提取可视化验证
"""
import sys
sys.path.append('src')
import numpy as np
import open3d as o3d
from src.boundary.detect import detect_boundary
from src.boundary.rim import extract_rim_curve
from src.common.base import Fragment

def create_ceramic_fragment():
    """创建一个模拟陶瓷碎片的点云"""
    # 创建椭圆柱形碎片（更接近真实陶瓷形状）
    n_points = 3000
    
    # 底部圆形边缘（rim区域）
    rim_points = 800
    theta_rim = np.linspace(0, 2*np.pi, rim_points, endpoint=False)
    r_rim = 1.0 + 0.1 * np.sin(3*theta_rim)  # 添加轻微波动模拟真实边缘
    x_rim = r_rim * np.cos(theta_rim)
    y_rim = r_rim * np.sin(theta_rim)
    z_rim = np.full_like(x_rim, -0.5)  # 底部
    
    # 上表面点
    surface_points = 1200
    r_surf = np.random.uniform(0, 0.9, surface_points)
    theta_surf = np.random.uniform(0, 2*np.pi, surface_points)
    x_surf = r_surf * np.cos(theta_surf)
    y_surf = r_surf * np.sin(theta_surf)
    z_surf = np.random.uniform(-0.4, 0.3, surface_points)
    
    # 侧面点
    side_points = 1000
    theta_side = np.random.uniform(0, 2*np.pi, side_points)
    z_side = np.random.uniform(-0.5, 0.3, side_points)
    r_side = 0.9 + 0.1 * np.random.random(side_points)  # 半径略有变化
    x_side = r_side * np.cos(theta_side)
    y_side = r_side * np.sin(theta_side)
    
    # 组合所有点
    x = np.concatenate([x_rim, x_surf, x_side])
    y = np.concatenate([y_rim, y_surf, y_side])
    z = np.concatenate([z_rim, z_surf, z_side])
    
    points = np.column_stack([x, y, z])
    
    # 添加少量噪声
    noise = np.random.normal(0, 0.02, points.shape)
    points += noise
    
    return points

def main():
    print("=== Rim边界提取可视化验证 ===")
    
    # 创建测试碎片
    fragment = Fragment(id=0, file_path='', file_name='ceramic_test.obj')
    points = create_ceramic_fragment()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    fragment.point_cloud = pcd
    
    print(f"创建测试点云: {len(points)} 个点")
    
    # 执行边界检测
    print("\n1. 执行边界检测...")
    boundary_pcd, boundary_pts = detect_boundary(fragment, visualize=False)
    
    if boundary_pts is None:
        print("❌ 边界检测失败")
        return
    
    print(f"✓ 检测到边界点: {len(boundary_pts)} 个")
    
    # 执行rim提取
    print("\n2. 执行rim曲线提取...")
    rim_curve, rim_pcd = extract_rim_curve(fragment, n_samples=200, visualize=False)
    
    if rim_curve is None:
        print("❌ Rim曲线提取失败")
        return
    
    print(f"✓ 提取rim曲线: {len(rim_curve)} 个点")
    
    # 可视化结果
    print("\n3. 显示结果...")
    
    # 原始点云（灰色）
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(points)
    original_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    
    # 边界点（红色）
    boundary_vis = o3d.geometry.PointCloud()
    boundary_vis.points = o3d.utility.Vector3dVector(boundary_pts)
    boundary_vis.paint_uniform_color([1, 0, 0])
    
    # Rim曲线点（绿色）
    rim_points_vis = o3d.geometry.PointCloud()
    rim_points_vis.points = o3d.utility.Vector3dVector(rim_curve)
    rim_points_vis.paint_uniform_color([0, 1, 0])
    
    # Rim连线（青色）
    rim_lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        rim_points_vis, rim_points_vis,
        np.array([[i, (i+1) % len(rim_curve)] for i in range(len(rim_curve))])
    )
    rim_lines.paint_uniform_color([0, 1, 1])
    
    # 显示所有元素
    geometries = [original_pcd, boundary_vis, rim_points_vis, rim_lines]
    
    print("显示内容:")
    print("- 灰色: 原始点云")
    print("- 红色点: 检测到的边界点")
    print("- 绿色点: Rim曲线采样点") 
    print("- 青色线: Rim曲线连接")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Rim边界提取验证",
        width=1000,
        height=800
    )
    
    # 输出统计信息
    print(f"\n=== 结果统计 ===")
    print(f"原始点云点数: {len(points)}")
    print(f"边界点数: {len(boundary_pts)} ({len(boundary_pts)/len(points)*100:.1f}%)")
    print(f"Rim曲线点数: {len(rim_curve)}")
    
    # 验证rim曲线质量
    distances = np.linalg.norm(np.diff(rim_curve, axis=0), axis=1)
    print(f"点间平均距离: {np.mean(distances):.4f}")
    print(f"点间距离标准差: {np.std(distances):.4f}")
    print(f"首尾点距离: {np.linalg.norm(rim_curve[0] - rim_curve[-1]):.4f}")

if __name__ == "__main__":
    main()