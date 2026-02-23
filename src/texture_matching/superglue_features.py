"""
SuperPoint关键点特征提取模块
从3D模型生成纹理图像并提取SuperPoint关键点描述子
"""
import numpy as np
import torch
import cv2
from pathlib import Path
import open3d as o3d
from typing import Optional, Tuple, Dict
import warnings

# SuperGlue相关导入
try:
    from models.matching import Matching
    from models.utils import frame2tensor
    SUPERGLUE_AVAILABLE = True
except ImportError:
    SUPERGLUE_AVAILABLE = False
    warnings.warn("SuperGlue not available. Please install SuperGluePretrainedNetwork")


def project_3d_to_2d_texture(fragment, resolution: Tuple[int, int] = (512, 512)) -> Optional[np.ndarray]:
    """
    将3D碎片投影到2D纹理图像
    :param fragment: Fragment对象
    :param resolution: 输出图像分辨率
    :return: 纹理图像 (H, W, 3) 或 None
    """
    if not hasattr(fragment, 'point_cloud') or fragment.point_cloud is None:
        return None
    
    # 获取点云数据
    points = np.asarray(fragment.point_cloud.points)
    
    # 如果有颜色信息，使用颜色；否则使用法向量生成伪颜色
    if fragment.point_cloud.has_colors():
        colors = np.asarray(fragment.point_cloud.colors)
    else:
        # 基于法向量生成颜色
        if fragment.point_cloud.has_normals():
            normals = np.asarray(fragment.point_cloud.normals)
            # 将法向量转换为颜色 (归一化到[0,1])
            colors = (normals + 1.0) / 2.0
        else:
            # 估计法向量
            fragment.point_cloud.estimate_normals()
            normals = np.asarray(fragment.point_cloud.normals)
            colors = (normals + 1.0) / 2.0
    
    # 简单的球面投影方法
    # 计算点云的包围盒中心
    center = np.mean(points, axis=0)
    points_centered = points - center
    
    # 计算到中心的距离
    distances = np.linalg.norm(points_centered, axis=1)
    
    # 归一化到[0,1]范围
    if np.max(distances) > np.min(distances):
        distances_normalized = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
    else:
        distances_normalized = np.zeros_like(distances)
    
    # 创建纹理图像
    h, w = resolution
    texture_image = np.zeros((h, w, 3), dtype=np.float32)
    
    # 简单的UV映射：使用球面坐标
    for i, point in enumerate(points_centered):
        # 转换为球面坐标
        x, y, z = point
        r = np.sqrt(x*x + y*y + z*z)
        if r == 0:
            continue
            
        # 计算球面角度
        theta = np.arccos(z / r)  # 极角 [0, π]
        phi = np.arctan2(y, x)    # 方位角 [-π, π]
        
        # 转换为图像坐标
        u = int((phi + np.pi) / (2 * np.pi) * (w - 1))
        v = int(theta / np.pi * (h - 1))
        
        # 边界检查
        if 0 <= u < w and 0 <= v < h:
            texture_image[v, u] = colors[i]
    
    # 转换为uint8格式
    texture_image = (texture_image * 255).astype(np.uint8)
    
    # 填充空洞（简单的邻近插值）
    mask = np.all(texture_image == 0, axis=2)
    if np.any(mask):
        # 找到非零像素的坐标
        nonzero_coords = np.argwhere(~mask)
        zero_coords = np.argwhere(mask)
        
        # 对每个零像素，找到最近的非零像素
        for coord in zero_coords:
            distances = np.linalg.norm(nonzero_coords - coord, axis=1)
            if len(distances) > 0:
                nearest_idx = np.argmin(distances)
                nearest_coord = nonzero_coords[nearest_idx]
                texture_image[coord[0], coord[1]] = texture_image[nearest_coord[0], nearest_coord[1]]
    
    return texture_image


def extract_superglue_features(image: np.ndarray, 
                              max_keypoints: int = 1024,
                              keypoint_threshold: float = 0.005) -> Optional[Dict]:
    """
    使用SuperPoint提取图像的关键点和描述子
    :param image: 输入图像 (H, W, 3) 或 (H, W)
    :param max_keypoints: 最大关键点数量
    :param keypoint_threshold: 关键点检测阈值
    :return: 包含关键点、描述子等信息的字典，或None
    """
    if not SUPERGLUE_AVAILABLE:
        print("[SuperGlue特征] SuperGlue不可用")
        return None
    
    try:
        # 确保图像是灰度图
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        # 调整图像大小以提高性能
        h, w = gray_image.shape
        if max(h, w) > 640:
            scale = 640.0 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            gray_image = cv2.resize(gray_image, (new_w, new_h))
        
        # 配置SuperPoint参数
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': 'indoor',  # 使用室内场景权重
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        
        # 初始化匹配器
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        matcher = Matching(config).eval().to(device)
        
        # 转换图像格式
        inp = frame2tensor(gray_image, device)
        
        # 提取特征
        pred = matcher.superpoint({'image': inp})
        
        # 提取关键点和描述子
        keypoints = pred['keypoints'][0].detach().cpu().numpy()  # (N, 2)
        descriptors = pred['descriptors'][0].detach().cpu().numpy().T  # (N, 256)
        scores = pred['scores'][0].detach().cpu().numpy()  # (N,)
        
        # 按分数排序，保留最高分的关键点
        if len(scores) > max_keypoints:
            top_k_indices = np.argsort(scores)[-max_keypoints:]
            keypoints = keypoints[top_k_indices]
            descriptors = descriptors[top_k_indices]
            scores = scores[top_k_indices]
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'scores': scores,
            'num_keypoints': len(keypoints),
            'image_shape': gray_image.shape
        }
        
    except Exception as e:
        print(f"[SuperGlue特征] 特征提取失败: {e}")
        return None


def generate_superglue_embedding(fragment, 
                                resolution: Tuple[int, int] = (512, 512),
                                max_keypoints: int = 1024) -> Optional[np.ndarray]:
    """
    为3D碎片生成SuperGlue特征嵌入向量
    :param fragment: Fragment对象
    :param resolution: 纹理图像分辨率
    :param max_keypoints: 最大关键点数
    :return: 256维特征向量，或None
    """
    try:
        # 1. 从3D投影到2D纹理
        texture_image = project_3d_to_2d_texture(fragment, resolution)
        if texture_image is None:
            print("[SuperGlue嵌入] 纹理图像生成失败")
            return None
        
        # 2. 提取SuperPoint特征
        features = extract_superglue_features(
            texture_image, 
            max_keypoints=max_keypoints,
            keypoint_threshold=0.005
        )
        
        if features is None or features['num_keypoints'] == 0:
            print("[SuperGlue嵌入] 关键点提取失败")
            return None
        
        # 3. 聚合描述子生成全局特征
        descriptors = features['descriptors']  # (N, 256)
        
        # 使用均值池化生成全局描述子
        global_descriptor = np.mean(descriptors, axis=0)
        
        # L2归一化
        norm = np.linalg.norm(global_descriptor)
        if norm > 1e-8:
            global_descriptor = global_descriptor / norm
        
        print(f"[SuperGlue嵌入] 成功生成 {len(descriptors)} 个关键点，输出256维特征")
        return global_descriptor.astype(np.float32)
        
    except Exception as e:
        print(f"[SuperGlue嵌入] 生成失败: {e}")
        return None


# 便捷函数
def extract_3d_superglue_features(fragment, 
                                 texture_resolution: Tuple[int, int] = (512, 512),
                                 max_keypoints: int = 1024) -> Dict:
    """
    提取3D碎片的完整SuperGlue特征信息
    :param fragment: Fragment对象
    :param texture_resolution: 纹理图像分辨率
    :param max_keypoints: 最大关键点数
    :return: 包含所有特征信息的字典
    """
    result = {
        'texture_image': None,
        'keypoints': None,
        'local_descriptors': None,
        'global_embedding': None,
        'num_keypoints': 0,
        'success': False
    }
    
    try:
        # 生成纹理图像
        texture_image = project_3d_to_2d_texture(fragment, texture_resolution)
        if texture_image is None:
            return result
        
        result['texture_image'] = texture_image
        
        # 提取SuperGlue特征
        features = extract_superglue_features(
            texture_image, 
            max_keypoints=max_keypoints
        )
        
        if features is None:
            return result
        
        result['keypoints'] = features['keypoints']
        result['local_descriptors'] = features['descriptors']
        result['num_keypoints'] = features['num_keypoints']
        
        # 生成全局嵌入
        if features['num_keypoints'] > 0:
            global_desc = np.mean(features['descriptors'], axis=0)
            norm = np.linalg.norm(global_desc)
            if norm > 1e-8:
                global_desc = global_desc / norm
            result['global_embedding'] = global_desc.astype(np.float32)
        
        result['success'] = True
        print(f"[3D SuperGlue] 成功提取 {features['num_keypoints']} 个关键点")
        
    except Exception as e:
        print(f"[3D SuperGlue] 提取失败: {e}")
    
    return result