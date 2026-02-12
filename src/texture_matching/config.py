"""
SuperGlue配置文件
对应算法文档中的参数设置
"""
import yaml
from pathlib import Path


# 默认配置
DEFAULT_CONFIG = {
    # SuperPoint配置
    'superpoint': {
        'nms_radius': 4,           # 非极大值抑制半径
        'keypoint_threshold': 0.005,  # 关键点检测阈值
        'max_keypoints': 1024      # 最大关键点数量
    },
    
    # SuperGlue配置
    'superglue': {
        'weights': 'indoor',       # 权重类型: 'indoor' 或 'outdoor'
        'sinkhorn_iterations': 20, # Sinkhorn迭代次数
        'match_threshold': 0.2,    # 匹配阈值
    },
    
    # 纹理处理配置
    'texture_processing': {
        'image_resolution': [512, 512],  # 纹理图像分辨率
        'projection_method': 'orthographic',  # 投影方法
        'segmentation_threshold': 0.7,       # 分割阈值
    },
    
    # 匹配参数
    'matching': {
        'min_matches': 10,         # 最少匹配点数
        'similarity_threshold': 0.3,  # 相似度阈值
        'top_k_candidates': 20,    # 候选对数量
        'feature_weight': 0.7,     # SuperGlue特征权重
        'embedding_weight': 0.3,   # 全局embedding权重
    },
    
    # 可视化配置
    'visualization': {
        'show_texture_regions': True,   # 显示纹样区域
        'show_matches': True,          # 显示匹配结果
        'save_intermediate': True,     # 保存中间结果
    }
}


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """加载配置"""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                # 合并默认配置和加载的配置
                config = DEFAULT_CONFIG.copy()
                self._merge_configs(config, loaded_config)
                return config
            except Exception as e:
                print(f"配置文件加载失败: {e}，使用默认配置")
        
        return DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, base: dict, override: dict):
        """递归合并配置字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default=None):
        """获取配置项，支持点分隔的路径"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value):
        """设置配置项"""
        keys = key_path.split('.')
        config = self.config
        
        # 导航到父级
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置值
        config[keys[-1]] = value
    
    def save_config(self, save_path: str = None):
        """保存配置到文件"""
        save_path = save_path or self.config_path
        if not save_path:
            save_path = 'configs/superglue_config.yaml'
        
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         indent=2, allow_unicode=True)
            print(f"配置已保存至: {save_path}")
        except Exception as e:
            print(f"配置保存失败: {e}")


# 预定义的配置模板
CONFIG_TEMPLATES = {
    'high_precision': {
        'superpoint': {
            'keypoint_threshold': 0.01,
            'max_keypoints': 2048
        },
        'superglue': {
            'match_threshold': 0.3,
            'sinkhorn_iterations': 30
        },
        'matching': {
            'similarity_threshold': 0.5,
            'min_matches': 15
        }
    },
    
    'fast_matching': {
        'superpoint': {
            'keypoint_threshold': 0.001,
            'max_keypoints': 512
        },
        'superglue': {
            'match_threshold': 0.1,
            'sinkhorn_iterations': 10
        },
        'matching': {
            'similarity_threshold': 0.2,
            'min_matches': 5
        }
    },
    
    'balanced': {
        'superpoint': {
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'match_threshold': 0.2,
            'sinkhorn_iterations': 20
        },
        'matching': {
            'similarity_threshold': 0.3,
            'min_matches': 10
        }
    }
}


def get_template_config(template_name: str = 'balanced') -> dict:
    """获取预定义配置模板"""
    if template_name in CONFIG_TEMPLATES:
        # 合并默认配置和模板配置
        config = DEFAULT_CONFIG.copy()
        ConfigManager()._merge_configs(config, CONFIG_TEMPLATES[template_name])
        return config
    else:
        print(f"未知模板名称 '{template_name}'，使用默认配置")
        return DEFAULT_CONFIG.copy()


def create_default_config_file(config_path: str = 'configs/superglue_config.yaml'):
    """创建默认配置文件"""
    config_manager = ConfigManager()
    config_manager.save_config(config_path)
    print(f"默认配置文件已创建: {config_path}")


if __name__ == "__main__":
    # 创建默认配置文件
    create_default_config_file()