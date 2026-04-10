"""
增强版纹理信息提取模块
支持OBJ文件的MTL材质和纹理信息解析
"""
import numpy as np
import torch
import cv2
from pathlib import Path
import open3d as o3d
from typing import List, Tuple, Dict, Optional
import warnings
import os

# SuperGlue相关导入
try:
    from models.matching import Matching
    from models.utils import frame2tensor
    SUPERGLUE_AVAILABLE = True
except ImportError:
    SUPERGLUE_AVAILABLE = False
    warnings.warn("SuperGlue not available. Please install SuperGluePretrainedNetwork")


class TextureExtractor:
    """增强版纹理提取器 - 支持MTL材质文件"""
    
    def __init__(self, data_dir: str = "data/demo"):
        self.data_dir = Path(data_dir)
        self.texture_cache = {}
    
    def extract_with_materials(self, obj_file_path: str) -> Optional[Dict]:
        """
        从OBJ文件提取完整的材质和纹理信息
        支持自动关联同名纹理图片（.jpg/.png）
        """
        obj_path = Path(obj_file_path)
        if not obj_path.exists():
            return None
        
        # 查找对应的MTL文件
        mtl_path = self._find_mtl_file(obj_path)
        
        # 如果没有MTL文件，尝试直接加载同名纹理图片
        if not mtl_path:
            print(f"[纹理提取] 未找到 {obj_path.name} 的MTL文件，尝试自动关联纹理图片")
            texture_file = self._find_texture_file(obj_path)
            if texture_file:
                print(f"[纹理提取] ✓ 成功关联纹理: {texture_file.name}")
                # 创建简化的材质信息
                materials = {
                    'auto_detected': {
                        'diffuse_color': [0.8, 0.8, 0.8],
                        'texture_map': str(texture_file),
                        'ambient_color': [0.2, 0.2, 0.2],
                        'specular_color': [1.0, 1.0, 1.0]
                    }
                }
                
                # 加载OBJ网格
                try:
                    mesh = o3d.io.read_triangle_mesh(str(obj_path))
                    if not mesh.has_triangle_normals():
                        mesh.compute_vertex_normals()
                except Exception as e:
                    print(f"[纹理提取] 网格加载失败: {e}")
                    return None
                
                result = {
                    'mesh': mesh,
                    'materials': materials,
                    'texture_coordinates': self._extract_uv_coordinates(mesh),
                    'vertex_colors': None
                }
                return result
            else:
                print(f"[纹理提取] ✗ 未找到 {obj_path.name} 的纹理图片")
                return None
        
        print(f"[纹理提取] 发现MTL文件: {mtl_path.name}")
        
        # 解析MTL文件
        materials = self._parse_mtl_file(mtl_path)
        if not materials:
            print(f"[纹理提取] MTL文件解析失败")
            return None
        
        # 如果MTL中没有纹理引用，尝试自动关联
        has_texture = any(mat.get('texture_map') for mat in materials.values())
        if not has_texture:
            texture_file = self._find_texture_file(obj_path)
            if texture_file:
                print(f"[纹理提取] MTL中无纹理引用，自动关联: {texture_file.name}")
                # 为第一个材质添加纹理引用
                first_mat_name = list(materials.keys())[0]
                materials[first_mat_name]['texture_map'] = str(texture_file)
        
        # 加载OBJ网格
        try:
            mesh = o3d.io.read_triangle_mesh(str(obj_path))
            if not mesh.has_triangle_normals():
                mesh.compute_vertex_normals()
        except Exception as e:
            print(f"[纹理提取] 网格加载失败: {e}")
            return None
        
        # 提取纹理坐标和材质信息
        result = {
            'mesh': mesh,
            'materials': materials,
            'texture_coordinates': self._extract_uv_coordinates(mesh),
            'vertex_colors': self._extract_vertex_colors_from_materials(mesh, materials, mtl_path.parent)
        }
        
        return result
    
    def _find_mtl_file(self, obj_path: Path) -> Optional[Path]:
        """查找对应的MTL文件"""
        # 同目录下的同名MTL文件
        mtl_candidates = [
            obj_path.with_suffix('.mtl'),
            obj_path.parent / f"{obj_path.stem}.mtl"
        ]
        
        for mtl_path in mtl_candidates:
            if mtl_path.exists():
                return mtl_path
        return None
    
    def _find_texture_file(self, obj_path: Path) -> Optional[Path]:
        """根据OBJ文件名自动查找纹理图片（方案B：自动关联）"""
        texture_candidates = [
            obj_path.with_suffix('.jpg'),
            obj_path.with_suffix('.jpeg'),
            obj_path.with_suffix('.png'),
            obj_path.parent / f"{obj_path.stem}.jpg",
            obj_path.parent / f"{obj_path.stem}.jpeg",
            obj_path.parent / f"{obj_path.stem}.png"
        ]
        
        for tex_path in texture_candidates:
            if tex_path.exists():
                print(f"[纹理提取] 自动发现纹理文件: {tex_path.name}")
                return tex_path
        return None
    
    def _parse_mtl_file(self, mtl_path: Path) -> Dict:
        """解析MTL材质文件"""
        materials = {}
        current_material = None
        
        try:
            with open(mtl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'newmtl':
                        current_material = parts[1]
                        materials[current_material] = {
                            'diffuse_color': [0.8, 0.8, 0.8],  # 默认漫反射颜色
                            'texture_map': None,
                            'ambient_color': [0.2, 0.2, 0.2],
                            'specular_color': [1.0, 1.0, 1.0]
                        }
                    elif current_material and parts[0] in ['Ka', 'Kd', 'Ks']:
                        # 颜色信息
                        color_key = {'Ka': 'ambient_color', 'Kd': 'diffuse_color', 'Ks': 'specular_color'}[parts[0]]
                        if len(parts) >= 4:
                            materials[current_material][color_key] = [float(x) for x in parts[1:4]]
                    elif current_material and parts[0] == 'map_Kd':
                        # 漫反射贴图
                        texture_file = ' '.join(parts[1:])
                        texture_path = mtl_path.parent / texture_file
                        if texture_path.exists():
                            materials[current_material]['texture_map'] = str(texture_path)
                            print(f"[纹理提取] 发现纹理贴图: {texture_file}")
        
        except Exception as e:
            print(f"[纹理提取] MTL文件解析错误: {e}")
            return {}
        
        return materials
    
    def _extract_uv_coordinates(self, mesh) -> Optional[np.ndarray]:
        """提取UV纹理坐标"""
        if hasattr(mesh, 'triangle_uvs') and len(mesh.triangle_uvs) > 0:
            return np.asarray(mesh.triangle_uvs)
        elif hasattr(mesh, 'vertex_uvs') and len(mesh.vertex_uvs) > 0:
            return np.asarray(mesh.vertex_uvs)
        else:
            print("[纹理提取] 网格不包含UV坐标信息")
            return None
    
    def _extract_vertex_colors_from_materials(self, mesh, materials: Dict, texture_dir: Path) -> Optional[np.ndarray]:
        """基于材质信息提取顶点颜色"""
        if not materials:
            return None
        
        vertices = np.asarray(mesh.vertices)
        vertex_colors = np.zeros((len(vertices), 3))
        
        # 简化处理：使用第一个材质的漫反射颜色
        first_material = list(materials.values())[0] if materials else None
        if first_material:
            # 如果有纹理贴图，尝试读取
            if first_material.get('texture_map'):
                try:
                    texture_img = cv2.imread(first_material['texture_map'])
                    if texture_img is not None:
                        # 简单的纹理采样（实际应用中需要UV映射）
                        h, w = texture_img.shape[:2]
                        sampled_colors = []
                        
                        # 随机采样一些点来估算平均颜色
                        for _ in range(min(100, len(vertices))):
                            x = np.random.randint(0, w)
                            y = np.random.randint(0, h)
                            color = texture_img[y, x]
                            sampled_colors.append(color[::-1])  # BGR to RGB
                        
                        if sampled_colors:
                            avg_color = np.mean(sampled_colors, axis=0) / 255.0
                            vertex_colors[:] = avg_color
                            print(f"[纹理提取] 从纹理贴图提取颜色: {avg_color}")
                except Exception as e:
                    print(f"[纹理提取] 纹理贴图处理失败: {e}")
            
            # 使用材质的漫反射颜色
            diffuse_color = first_material.get('diffuse_color', [0.8, 0.8, 0.8])
            if np.sum(vertex_colors) == 0:  # 如果还没有设置颜色
                vertex_colors[:] = diffuse_color
                print(f"[纹理提取] 使用材质漫反射颜色: {diffuse_color}")
        
        return vertex_colors


class EnhancedTextureMatcher:
    """结合材质信息的增强纹理匹配器"""
    
    def __init__(self):
        self.texture_extractor = TextureExtractor()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.matcher = None
        if SUPERGLUE_AVAILABLE:
            self._init_superglue()
    
    def _init_superglue(self):
        """初始化SuperGlue"""
        try:
            config = {
                'superpoint': {
                    'nms_radius': 4,
                    'keypoint_threshold': 0.005,
                    'max_keypoints': 1024
                },
                'superglue': {
                    'weights': 'indoor',
                    'sinkhorn_iterations': 20,
                    'match_threshold': 0.2,
                }
            }
            from models.matching import Matching
            self.matcher = Matching(config).eval().to(self.device)
            print(f"[SuperGlue] 初始化成功 (设备: {self.device})")
        except Exception as e:
            print(f"[SuperGlue] 初始化失败: {e}")
            self.matcher = None
    
    def extract_enhanced_features(self, fragment_path: str) -> Optional[Dict]:
        """
        提取增强的纹理特征（包含材质信息）
        """
        # 首先尝试从OBJ+MTL提取完整纹理信息
        texture_data = self.texture_extractor.extract_with_materials(fragment_path)
        
        if texture_data:
            print(f"[增强特征提取] 成功提取材质纹理信息")
            # 可以在这里添加更复杂的纹理分析
            return {
                'method': 'enhanced_texture',
                'texture_data': texture_data,
                'has_materials': True
            }
        else:
            print(f"[增强特征提取] 使用基础几何特征")
            # 回退到基础几何特征提取
            return None


# 便捷函数
def analyze_fragment_textures(data_dir: str = "data/demo"):
    """分析所有碎片的纹理信息"""
    extractor = TextureExtractor(data_dir)
    
    # 查找OBJ文件
    obj_files = list(Path(data_dir).glob("*.obj"))
    print(f"发现 {len(obj_files)} 个OBJ文件")
    
    for obj_file in obj_files:
        print(f"\n分析文件: {obj_file.name}")
        texture_info = extractor.extract_with_materials(str(obj_file))
        if texture_info:
            materials = texture_info['materials']
            print(f"  发现 {len(materials)} 个材质:")
            for mat_name, mat_props in materials.items():
                print(f"    - {mat_name}:")
                print(f"      漫反射颜色: {mat_props['diffuse_color']}")
                if mat_props['texture_map']:
                    print(f"      纹理贴图: {Path(mat_props['texture_map']).name}")
        else:
            print("  未找到有效的材质信息")

if __name__ == "__main__":
    # 运行纹理分析
    analyze_fragment_textures()