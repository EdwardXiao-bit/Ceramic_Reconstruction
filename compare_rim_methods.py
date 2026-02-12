# D:\ceramic_reconstruction\compare_rim_methods.py
"""
rim曲线提取方法对比测试
比较传统PCA方法与测地线方法的效果
使用 data/eg1 文件夹中的测试数据
"""
import sys
import numpy as np
import open3d as o3d
import os
sys.path.append('src')

from src.common.base import Fragment
from src.boundary.rim import extract_rim_curve
from src.boundary.geodesic_rim import extract_geodesic_rim_curve


def load_eg1_fragment(fragment_id=2):
    """
    加载eg1文件夹中的指定碎片
    :param fragment_id: 碎片编号 (2, 3, 4, 5, 6)
    :return: Fragment对象
    """
    file_name = f"{fragment_id}.obj"
    file_path = f"data/eg1/{file_name}"
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    # 创建Fragment对象
    fragment = Fragment(id=fragment_id, file_path=file_path, file_name=file_name)
    
    # 加载网格数据
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        if mesh is None or len(mesh.vertices) == 0:
            print(f"无法加载网格: {file_name}")
            return None
            
        fragment.mesh = mesh
        
        # 从网格采样点云
        if len(mesh.triangles) > 0:
            fragment.point_cloud = mesh.sample_points_uniformly(number_of_points=10000)
        else:
            # 如果没有三角面，直接使用顶点
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            fragment.point_cloud = pcd
            
        print(f"成功加载碎片 {fragment_id}: {file_name}")
        print(f"  网格顶点数: {len(mesh.vertices)}")
        print(f"  网格面数: {len(mesh.triangles)}")
        print(f"  点云点数: {len(fragment.point_cloud.points)}")
        
        return fragment
    except Exception as e:
        print(f"加载失败: {e}")
        return None


def extract_simple_patch(fragment, k_neighbors=30):
    """
    简单的patch提取方法（不需要预先检测边界）
    :param fragment: Fragment对象
    :param k_neighbors: 近邻数
    :return: patch点云
    """
    if fragment.point_cloud is None:
        return None
    
    points = np.asarray(fragment.point_cloud.points)
    n_points = len(points)
    
    # 基于高度提取高点作为种子点
    z_coords = points[:, 2]
    z_threshold = np.percentile(z_coords, 95)  # 取最高的5%
    seed_mask = z_coords > z_threshold
    seed_indices = np.where(seed_mask)[0]
    
    if len(seed_indices) == 0:
        # 如果没有高点，取随机点作为种子
        seed_indices = np.random.choice(n_points, min(50, n_points), replace=False)
    
    # 使用KD树查找近邻
    pcd_tree = o3d.geometry.KDTreeFlann(fragment.point_cloud)
    patch_indices = set(seed_indices)
    
    # 扩展patch区域
    for seed_idx in seed_indices[:20]:  # 限制种子点数量
        _, neighbor_idx, _ = pcd_tree.search_knn_vector_3d(points[seed_idx], k_neighbors)
        patch_indices.update(neighbor_idx)
    
    patch_indices = list(patch_indices)
    
    # 创建patch点云
    patch_pcd = o3d.geometry.PointCloud()
    patch_pcd.points = o3d.utility.Vector3dVector(points[patch_indices])
    patch_pcd.paint_uniform_color([0, 0, 1])  # 蓝色
    
    return patch_pcd


def compare_methods():
    print("=== Rim曲线提取方法对比测试 (使用eg1数据) ===")
    
    # 加载eg1测试数据
    fragment = load_eg1_fragment(2)  # 使用碎片2进行测试
    if fragment is None:
        print("无法加载测试数据")
        return

    # 1. 提取断面patch
    print("\n1. 提取断面patch...")
    patch_pcd = extract_simple_patch(fragment, k_neighbors=25)
    if patch_pcd is None:
        print("Patch提取失败")
        return
    
    print(f"   Patch点数: {len(patch_pcd.points)}")

    # 方法1: 传统PCA方法
    print("\n2. 传统PCA方法提取rim曲线...")
    try:
        # 为传统方法创建边界点
        if not hasattr(fragment, 'boundary_pts') or fragment.boundary_pts is None:
            print("   警告: 缺少边界点，使用patch边界点")
            patch_points = np.asarray(patch_pcd.points)
            # 提取patch的边界点
            patch_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
            normals = np.asarray(patch_pcd.normals)
            
            # 基于法向变化检测边界
            tree = o3d.geometry.KDTreeFlann(patch_pcd)
            curvature = np.zeros(len(patch_points))
            
            for i in range(len(patch_points)):
                _, idx, _ = tree.search_knn_vector_3d(patch_points[i], 20)
                nbr_normals = normals[idx]
                mean_normal = nbr_normals.mean(axis=0)
                curvature[i] = np.mean(np.linalg.norm(nbr_normals - mean_normal, axis=1))
            
            # 选择高曲率点作为边界
            boundary_indices = np.argsort(curvature)[-min(100, len(curvature)):]
            fragment.boundary_pts = patch_points[boundary_indices]
            
            boundary_pcd = o3d.geometry.PointCloud()
            boundary_pcd.points = o3d.utility.Vector3dVector(fragment.boundary_pts)
            boundary_pcd.paint_uniform_color([1, 0, 0])
            fragment.boundary_pcd = boundary_pcd
            print(f"   创建了 {len(fragment.boundary_pts)} 个边界点")
        
        rim_traditional, rim_pcd_traditional = extract_rim_curve(
            fragment, n_samples=100, visualize=False
        )
        if rim_traditional is not None:
            print(f"   ✓ 传统方法成功: {len(rim_traditional)} 个点")
        else:
            print("   ✗ 传统方法失败")
    except Exception as e:
        print(f"   ✗ 传统方法出错: {e}")

    # 方法2: 测地线方法
    print("\n3. 测地线方法提取rim曲线...")
    try:
        rim_geodesic, rim_pcd_geodesic = extract_geodesic_rim_curve(
            fragment, patch_pcd=patch_pcd, n_samples=100, visualize=True
        )
        if rim_geodesic is not None:
            print(f"   ✓ 测地线方法成功: {len(rim_geodesic)} 个点")
            if hasattr(fragment, 'geodesic_attributes'):
                print(f"   ✓ 几何属性维度: {fragment.geodesic_attributes.shape}")
        else:
            print("   ✗ 测地线方法失败")
    except Exception as e:
        print(f"   ✗ 测地线方法出错: {e}")
        import traceback
        traceback.print_exc()

    # 结果对比
    print("\n=== 结果对比 ===")
    if hasattr(fragment, 'rim_curve') and fragment.rim_curve is not None:
        print(f"传统方法rim点数: {len(fragment.rim_curve)}")
    if hasattr(fragment, 'geodesic_rim_curve') and fragment.geodesic_rim_curve is not None:
        print(f"测地线方法rim点数: {len(fragment.geodesic_rim_curve)}")
        if hasattr(fragment, 'geodesic_attributes'):
            print(f"几何属性: {fragment.geodesic_attributes.shape}")


if __name__ == "__main__":
    compare_methods()