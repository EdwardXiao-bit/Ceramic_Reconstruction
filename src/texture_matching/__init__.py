"""
纹理匹配模块
实现SuperGlue纹样匹配功能
"""

from .superglue_integration import (
    TextureMatcher,
    PatternEncoder,
    integrate_texture_matching,
    check_superglue_availability,
    get_texture_matching_fallback
)

from .advanced_matching import (
    AdvancedTextureMatcher,
    TextureMatchingPipeline,
    run_texture_matching_pipeline
)

from .config import (
    ConfigManager,
    get_template_config,
    create_default_config_file
)

__all__ = [
    'TextureMatcher',
    'PatternEncoder', 
    'integrate_texture_matching',
    'check_superglue_availability',
    'get_texture_matching_fallback',
    'AdvancedTextureMatcher',
    'TextureMatchingPipeline',
    'run_texture_matching_pipeline',
    'ConfigManager',
    'get_template_config',
    'create_default_config_file'
]