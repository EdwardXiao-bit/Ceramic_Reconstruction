# src/boundary_validation/config.py
"""
边界验证模块配置（MVP修复版）
核心修改：大幅降低阈值适应无预训练权重的MVP阶段
"""


class BoundaryValidationConfig:
    BOUNDARY_EXTRACTION = {
        'curvature_threshold': 0.05,  # 降低（原0.1）
        'roughness_threshold': 0.02,  # 降低
        'depth_discontinuity_threshold': 0.01,
        'min_boundary_points': 20,  # 降低（原50）
        'normal_angle_threshold': 30.0,
        'clustering_eps': 0.05,  # 增大
        'min_cluster_size': 10,  # 降低（原20）
    }

    FEATURE_MATCHING = {
        'predator_enabled': True,
        'd3feat_enabled': False,  # 暂时禁用，只用FPFH
        'matchability_threshold': 0.3,  # 大幅降低（原0.6）
        'min_matches': 1,  # 降低到1（原3）
        'overlap_threshold': 0.05,  # 大幅降低（原0.15）
        'inlier_ratio_threshold': 0.05,  # 大幅降低（原0.2）
        'feature_weights': {
            'overlap_score': 0.3,
            'matchability_score': 0.4,
            'inlier_ratio': 0.3,
        }
    }

    COMPLEMENTARITY_CHECK = {
        'normal_complementarity': {
            'mirror_angle_tolerance': 45.0,  # 大幅放宽（原20度）
            'min_normal_similarity': 0.3,  # 大幅降低（原0.7）
            'reverse_normal_ratio': 0.3,  # 降低（原0.6）
        },
        'shape_complementarity': {
            'patch_size': 30,  # 降低
            'pointnet_enabled': False,  # MVP阶段禁用
            'cnn_enabled': False,  # MVP阶段禁用
            'similarity_threshold': 0.3,  # 降低（原0.6）
        }
    }

    LOCAL_ALIGNMENT = {
        'dcp_enabled': False,  # MVP阶段禁用，只用ICP
        'icp_enabled': True,
        'dcp': {
            'transformer_layers': 6,
            'attention_heads': 4,
            'embedding_dim': 512,
            'max_iterations': 50,
            'convergence_threshold': 1e-6,
        },
        'icp': {
            'max_iterations': 50,
            'distance_threshold': 0.1,  # 放宽（原0.05）
            'fitness_threshold': 0.3,  # 降低（原0.8）
            'rmse_threshold': 0.05,  # 放宽（原0.01）
        }
    }

    COLLISION_DETECTION = {
        'sdf_enabled': True,
        'voxel_enabled': False,
        'resolution': 32,  # 降低分辨率加速
        'padding_factor': 1.2,
        'penetration_threshold': 0.02,
        'collision_penalty': 2.0,
    }

    FINAL_SCORING = {
        'weights': {
            'feature_score': 0.40,  # 提高特征权重
            'normal_score': 0.15,
            'shape_score': 0.15,
            'alignment_score': 0.20,
            'collision_penalty': 0.10,
        },
        'thresholds': {
            'minimum_total_score': 0.10,  # 大幅降低（原0.6）！MVP关键修改
            'minimum_component_score': 0.05,  # 大幅降低（原0.4）
        }
    }

    VISUALIZATION = {
        'show_boundary_extraction': False,
        'show_feature_matching': False,
        'show_complementarity': False,
        'show_alignment': False,
        'show_collision': False,
        'point_size': 2.0,
        'line_width': 3.0,
    }


CONFIG = BoundaryValidationConfig()


def get_config():
    return CONFIG


def update_config(new_config):
    global CONFIG
    CONFIG = new_config
    return CONFIG