# D:\ceramic_reconstruction\src\boundary_validation\config.py
"""
边界验证模块配置文件
定义各种阈值和参数设置
"""

import numpy as np

class BoundaryValidationConfig:
    """边界验证配置类"""
    
    # 1. 边界区域提取参数
    BOUNDARY_EXTRACTION = {
        'curvature_threshold': 0.1,           # 曲率阈值
        'roughness_threshold': 0.05,          # 表面粗糙度阈值
        'depth_discontinuity_threshold': 0.02, # 深度不连续阈值
        'min_boundary_points': 50,            # 最小边界点数
        'normal_angle_threshold': 30.0,       # 法向角度阈值（度）
        'clustering_eps': 0.03,               # DBSCAN聚类半径
        'min_cluster_size': 20,               # 最小聚类大小
    }
    
    # 2. 特征匹配验证参数
    FEATURE_MATCHING = {
        'predator_enabled': True,             # ✅ 启用 Predator（有预训练权重）
        'd3feat_enabled': True,               # ⚠️ 保留接口（当前使用随机权重，后续可加载预训练模型）
        'matchability_threshold': 0.6,        # 匹配可信度阈值（降低）
        'min_matches': 3,                     # 最少匹配点对数（进一步降低）
        'overlap_threshold': 0.15,            # 重叠度阈值（降低）
        'inlier_ratio_threshold': 0.2,        # 内点比率阈值（降低，放宽 RANSAC 约束）
        'feature_weights': {
            'overlap_score': 0.3,             # 重叠度权重
            'matchability_score': 0.4,        # 匹配度权重
            'inlier_ratio': 0.3,              # 内点比率权重
        }
    }
    
    # 3. 互补性检查参数
    COMPLEMENTARITY_CHECK = {
        'normal_complementarity': {
            'mirror_angle_tolerance': 20.0,   # 镜像法向角度容忍度（度）（放宽）
            'min_normal_similarity': 0.7,     # 最小法向相似度（降低）
            'reverse_normal_ratio': 0.6,      # 反向法向比例阈值（降低）
        },
        'shape_complementarity': {
            'patch_size': 50,                 # boundary patch 大小
            'pointnet_enabled': True,         # 是否使用 PointNet++
            'cnn_enabled': True,              # 是否使用 3D CNN ✓ 启用
            'similarity_threshold': 0.6,      # 形状相似度阈值（降低）
        }
    }
    
    # 4. 局部对齐参数
    LOCAL_ALIGNMENT = {
        'dcp_enabled': True,                  # 是否使用DCP
        'icp_enabled': True,                  # 是否使用局部ICP
        'dcp': {
            'transformer_layers': 6,          # Transformer层数
            'attention_heads': 4,             # 注意力头数
            'embedding_dim': 512,             # 嵌入维度
            'max_iterations': 50,             # 最大迭代次数
            'convergence_threshold': 1e-6,    # 收敛阈值
        },
        'icp': {
            'max_iterations': 30,             # ICP最大迭代次数
            'distance_threshold': 0.05,       # 距离阈值
            'fitness_threshold': 0.8,         # 适应度阈值
            'rmse_threshold': 0.01,           # RMSE阈值
        }
    }
    
    # 5. 碰撞检测参数
    COLLISION_DETECTION = {
        'sdf_enabled': True,                  # 是否使用SDF
        'voxel_enabled': False,               # 是否使用体素占用
        'resolution': 64,                     # 网格分辨率
        'padding_factor': 1.2,                # 边界填充因子
        'penetration_threshold': 0.01,        # 穿透阈值
        'collision_penalty': 2.0,             # 碰撞惩罚系数
    }
    
    # 6. 综合评分参数
    FINAL_SCORING = {
        'weights': {
            'feature_score': 0.25,            # 特征匹配得分权重
            'normal_score': 0.20,             # 法向互补性得分权重
            'shape_score': 0.25,              # 形状互补性得分权重
            'alignment_score': 0.20,          # 对齐精度得分权重
            'collision_penalty': 0.10,        # 碰撞惩罚权重
        },
        'thresholds': {
            'minimum_total_score': 0.6,       # 最低总分阈值
            'minimum_component_score': 0.4,   # 最低组件得分阈值
        }
    }
    
    # 7. 可视化参数
    VISUALIZATION = {
        'show_boundary_extraction': False,    # 显示边界提取过程
        'show_feature_matching': False,       # 显示特征匹配结果
        'show_complementarity': False,        # 显示互补性检查
        'show_alignment': False,              # 显示对齐过程
        'show_collision': False,              # 显示碰撞检测
        'point_size': 2.0,                    # 点大小
        'line_width': 3.0,                    # 线宽
    }

# 全局配置实例
CONFIG = BoundaryValidationConfig()

def get_config():
    """获取全局配置"""
    return CONFIG

def update_config(new_config):
    """更新配置"""
    global CONFIG
    CONFIG = new_config
    return CONFIG