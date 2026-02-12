"""
传统几何特征提取模块
实现FPFH等经典点云特征描述子
"""
import numpy as np
import open3d as o3d


def compute_patch_fpfh(fragment, radius=0.05, max_nn=100):
    """
    计算断面patch的FPFH特征
    :param fragment: Fragment对象，需包含section_patch
    :param radius: FPFH搜索半径
    :param max_nn: 最大近邻数
    :return: FPFH特征向量 | None
    """
    if not hasattr(fragment, "section_patch") or fragment.section_patch is None:
        return None
    
    patch = fragment.section_patch
    if len(patch.points) < 10:
        return None
    
    # 确保有点云有法向量
    if not patch.has_normals():
        patch.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        patch.orient_normals_consistent_tangent_plane(k=20)
    
    # 计算FPFH特征
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        patch,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    
    # 转换为numpy数组并返回统计特征
    fpfh_data = np.asarray(fpfh.data)
    if fpfh_data.size == 0:
        return None
    
    # 计算统计特征（均值、标准差等）
    features = {
        'mean': np.mean(fpfh_data, axis=1),
        'std': np.std(fpfh_data, axis=1),
        'max': np.max(fpfh_data, axis=1),
        'min': np.min(fpfh_data, axis=1)
    }
    
    # 拼接为单一特征向量
    feature_vector = np.concatenate([
        features['mean'], 
        features['std'], 
        features['max'], 
        features['min']
    ])
    
    return feature_vector


def compute_patch_shot(fragment, radius=0.05):
    """
    计算断面patch的SHOT特征（可选的替代方案）
    :param fragment: Fragment对象
    :param radius: SHOT描述子搜索半径
    :return: SHOT特征向量 | None
    """
    # 注意：Open3D的SHOT实现可能需要额外安装
    # 这里提供接口但可能需要额外配置
    try:
        if not hasattr(fragment, "section_patch") or fragment.section_patch is None:
            return None
        
        patch = fragment.section_patch
        if len(patch.points) < 10:
            return None
            
        # 确保有法向量
        if not patch.has_normals():
            patch.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
            )
        
        # SHOT计算（如果Open3D支持）
        # shot = o3d.compute_shot_feature(patch, radius)
        # return np.asarray(shot.data).flatten()
        
        # 临时返回None，需要具体实现
        return None
    except Exception as e:
        print(f"[SHOT特征] 计算失败: {e}")
        return None


def extract_statistical_features(fragment):
    """
    提取patch的基本统计特征
    :param fragment: Fragment对象
    :return: 统计特征字典
    """
    if not hasattr(fragment, "section_patch") or fragment.section_patch is None:
        return None
    
    patch = fragment.section_patch
    points = np.asarray(patch.points)
    
    if len(points) < 3:
        return None
    
    # 基本统计特征
    centroid = np.mean(points, axis=0)
    covariance = np.cov(points.T)
    
    # 特征值分解
    eigenvals, eigenvecs = np.linalg.eigh(covariance)
    eigenvals = np.sort(eigenvals)[::-1]  # 降序排列
    
    # 计算各种几何特征
    features = {
        'centroid': centroid,
        'volume': np.prod(eigenvals),
        'surface_area': len(points),
        'compactness': eigenvals[2] / eigenvals[0],  # 紧凑度
        'scatter': np.sum(eigenvals),  # 散布度
        'eigenvals': eigenvals,
        'anisotropy': (eigenvals[0] - eigenvals[2]) / eigenvals[0],  # 各向异性
        'linearity': (eigenvals[0] - eigenvals[1]) / eigenvals[0],  # 线性度
        'planarity': (eigenvals[1] - eigenvals[2]) / eigenvals[0],  # 平面性
    }
    
    return features