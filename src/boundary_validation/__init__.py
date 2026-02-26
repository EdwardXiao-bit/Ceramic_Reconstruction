# D:\ceramic_reconstruction\src\boundary_validation\__init__.py
"""
边界验证模块初始化文件
"""

from .config import CONFIG, get_config, update_config
from .boundary_extractor import BoundaryExtractor
from .feature_matcher import FeatureMatcher
from .complementarity_checker import ComplementarityChecker
from .local_aligner import LocalAligner
from .collision_detector import CollisionDetector
from .scoring_system import ScoringSystem
from .validator import BoundaryValidator

__all__ = [
    'CONFIG',
    'get_config', 
    'update_config',
    'BoundaryExtractor',
    'FeatureMatcher',
    'ComplementarityChecker',
    'LocalAligner',
    'CollisionDetector',
    'ScoringSystem',
    'BoundaryValidator'
]

__version__ = '1.0.0'