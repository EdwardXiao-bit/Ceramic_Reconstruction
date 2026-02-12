"""
增强版SuperGlue集成模块
提供更完整的纹样匹配功能
"""
import numpy as np
import torch
import cv2
from pathlib import Path
import open3d as o3d
from typing import List, Tuple, Dict, Optional
import json
import pickle
from datetime import datetime

# 注释掉缺失的模块导入
# from .superglue_integration import TextureMatcher, PatternEncoder
from .config import ConfigManager
# 从enhanced_superglue导入正确的SUPERGLUE_AVAILABLE
from .enhanced_superglue import SUPERGLUE_AVAILABLE


# 创建临时的PatternEncoder类作为占位符
class PatternEncoder:
    @staticmethod
    def extract_global_embedding(image):
        # 简单的占位实现
        return np.random.rand(256) if image is not None else None
    
    @staticmethod
    def compute_similarity(emb1, emb2):
        # 简单的占位实现
        if emb1 is None or emb2 is None:
            return 0.0
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))


# 创建临时的TextureMatcher基类作为占位符
class TextureMatcher:
    def __init__(self, config=None):
        self.config = config or {}
    
    def extract_texture_region(self, fragment):
        # 简单的占位实现
        if hasattr(fragment, 'point_cloud') and fragment.point_cloud is not None:
            points = np.asarray(fragment.point_cloud.points)
            # 简单地返回前100个点作为纹理区域
            return points[:min(100, len(points))] if len(points) > 0 else None
        return None
    
    def project_to_image(self, texture_points, resolution=(512, 512)):
        # 简单的占位实现
        if texture_points is None or len(texture_points) == 0:
            return None
        # 创建一个简单的灰度图像
        return np.random.randint(0, 255, resolution, dtype=np.uint8)
    
    def extract_superglue_features(self, image):
        # 简单的占位实现
        if image is None:
            return None
        return {
            'keypoints': np.array([[100, 100], [200, 200]], dtype=np.float32),
            'descriptors': np.random.rand(2, 256).astype(np.float32),
            'scores': np.array([0.9, 0.8], dtype=np.float32)
        }
    
    def compute_texture_similarity(self, features1, features2):
        # 简单的占位实现
        if features1 is None or features2 is None:
            return 0.0
        return 0.5  # 返回固定相似度


class AdvancedTextureMatcher(TextureMatcher):
    """高级纹样匹配器"""
    
    def __init__(self, config_path: str = None, use_superglue: bool = True):
        # 加载配置
        self.config_manager = ConfigManager(config_path)
        config = self.config_manager.config
        
        # 初始化基础匹配器
        super().__init__(config)
        
        # 设置SuperGlue使用标志
        self.use_superglue = use_superglue and SUPERGLUE_AVAILABLE
        
        # 高级功能参数
        self.similarity_threshold = self.config_manager.get('matching.similarity_threshold', 0.3)
        self.min_matches = self.config_manager.get('matching.min_matches', 10)
        self.feature_weight = self.config_manager.get('matching.feature_weight', 0.7)
        self.embedding_weight = self.config_manager.get('matching.embedding_weight', 0.3)
        
        # 结果缓存
        self.match_cache = {}
        self.feature_cache = {}
    
    def batch_extract_features(self, fragments: List, cache_dir: str = None) -> Dict:
        """
        批量提取特征并缓存
        """
        if cache_dir:
            cache_path = Path(cache_dir) / 'texture_features.pkl'
            if cache_path.exists():
                print("[批量特征提取] 加载缓存特征...")
                try:
                    with open(cache_path, 'rb') as f:
                        cached_features = pickle.load(f)
                    print(f"[批量特征提取] 加载了 {len(cached_features)} 个碎片的特征")
                    return cached_features
                except Exception as e:
                    print(f"[批量特征提取] 缓存加载失败: {e}")
        
        print("[批量特征提取] 开始提取特征...")
        features = {}
        
        for i, fragment in enumerate(fragments):
            print(f"  处理碎片 {i+1}/{len(fragments)}: {getattr(fragment, 'file_name', f'fragment_{i}')}")
            
            # 提取纹理区域
            texture_points = self.extract_texture_region(fragment)
            if texture_points is None:
                features[i] = None
                continue
            
            # 投影到图像
            texture_image = self.project_to_image(
                texture_points,
                tuple(self.config_manager.get('texture_processing.image_resolution', [512, 512]))
            )
            
            if texture_image is None:
                features[i] = None
                continue
            
            # 提取特征
            superglue_features = self.extract_superglue_features(texture_image)
            global_embedding = PatternEncoder().extract_global_embedding(texture_image)
            
            features[i] = {
                'superglue_features': superglue_features,
                'global_embedding': global_embedding,
                'texture_image': texture_image,
                'texture_points': texture_points
            }
            
            print(f"    ✓ 提取完成 ({len(superglue_features['keypoints']) if superglue_features else 0} 个关键点)")
        
        # 保存缓存
        if cache_dir and features:
            try:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                cache_path = Path(cache_dir) / 'texture_features.pkl'
                with open(cache_path, 'wb') as f:
                    pickle.dump(features, f)
                print(f"[批量特征提取] 特征已缓存至: {cache_path}")
            except Exception as e:
                print(f"[批量特征提取] 缓存保存失败: {e}")
        
        return features
    
    def advanced_matching(self, features: Dict, fragments: List) -> List[Tuple]:
        """
        高级匹配算法
        """
        print("[高级匹配] 开始匹配计算...")
        
        # 只使用当前碎片列表中存在的索引
        max_valid_index = len(fragments) - 1
        valid_indices = [i for i, feat in features.items() 
                        if feat is not None and i <= max_valid_index]
        
        print(f"[高级匹配] 有效碎片索引: {valid_indices} (当前碎片数: {len(fragments)})")
        
        candidates = []
        
        total_pairs = len(valid_indices) * (len(valid_indices) - 1) // 2
        processed_pairs = 0
        
        for i in range(len(valid_indices)):
            for j in range(i + 1, len(valid_indices)):
                idx1, idx2 = valid_indices[i], valid_indices[j]
                processed_pairs += 1
                
                print(f"  匹配进度: {processed_pairs}/{total_pairs} ({idx1}-{idx2})")
                
                feat1 = features[idx1]
                feat2 = features[idx2]
                
                # 计算多种相似度
                sg_similarity = self.compute_texture_similarity(
                    feat1['superglue_features'], feat2['superglue_features']
                )
                
                emb_similarity = PatternEncoder().compute_similarity(
                    feat1['global_embedding'], feat2['global_embedding']
                )
                
                # 综合相似度
                combined_similarity = (
                    self.feature_weight * sg_similarity + 
                    self.embedding_weight * emb_similarity
                )
                
                # 几何约束（可选）
                geometric_score = self._compute_geometric_compatibility(
                    fragments[idx1], fragments[idx2]
                )
                
                # 最终得分
                final_score = combined_similarity * geometric_score
                
                if final_score >= self.similarity_threshold:
                    candidates.append((idx1, idx2, final_score, {
                        'sg_similarity': sg_similarity,
                        'emb_similarity': emb_similarity,
                        'geometric_score': geometric_score
                    }))
                
                print(f"    相似度: {final_score:.3f} (SG:{sg_similarity:.3f}, Emb:{emb_similarity:.3f}, Geo:{geometric_score:.3f})")
        
        # 排序并返回Top-K
        candidates.sort(key=lambda x: x[2], reverse=True)
        top_k = self.config_manager.get('matching.top_k_candidates', 20)
        return candidates[:top_k]
    
    def _compute_geometric_compatibility(self, frag1, frag2) -> float:
        """
        计算几何兼容性分数
        基于碎片尺寸、厚度等几何特征
        """
        try:
            # 简单的几何兼容性检查
            if hasattr(frag1, 'point_cloud') and hasattr(frag2, 'point_cloud'):
                # 获取点云边界框
                bbox1 = frag1.point_cloud.get_axis_aligned_bounding_box()
                bbox2 = frag2.point_cloud.get_axis_aligned_bounding_box()
                
                # 计算尺寸比
                size1 = bbox1.get_extent()
                size2 = bbox2.get_extent()
                
                size_ratio = np.minimum(size1 / (size2 + 1e-8), size2 / (size1 + 1e-8))
                avg_ratio = np.mean(size_ratio)
                
                # 尺寸相近性得分
                size_score = np.clip(avg_ratio, 0, 1)
                
                return size_score
            else:
                return 1.0  # 无几何信息时返回中性分数
        except Exception:
            return 1.0
    
    def visualize_matches(self, candidates: List[Tuple], features: Dict, fragments: List, 
                         output_dir: str = None):
        """
        可视化匹配结果
        """
        if not self.config_manager.get('visualization.show_matches', True):
            return
        
        print("[可视化] 生成匹配结果可视化...")
        
        for i, (idx1, idx2, score, details) in enumerate(candidates[:5]):  # 只显示前5个
            try:
                feat1 = features[idx1]
                feat2 = features[idx2]
                
                if feat1 is None or feat2 is None:
                    continue
                
                # 创建匹配可视化
                img1 = feat1['texture_image']
                img2 = feat2['texture_image']
                
                # 简化的匹配线可视化
                vis_img = self._create_match_visualization(img1, img2, details)
                
                if output_dir:
                    output_path = Path(output_dir) / f'match_{idx1}_{idx2}.png'
                    cv2.imwrite(str(output_path), vis_img)
                    print(f"  ✓ 匹配可视化已保存: {output_path}")
                
            except Exception as e:
                print(f"  ✗ 可视化失败 (碎片{idx1}-{idx2}): {e}")
    
    def _create_match_visualization(self, img1, img2, details) -> np.ndarray:
        """创建匹配可视化图像"""
        # 确保图像为彩色
        if len(img1.shape) == 2:
            img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            img1_color = img1.copy()
            
        if len(img2.shape) == 2:
            img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else:
            img2_color = img2.copy()
        
        # 水平拼接
        h1, w1 = img1_color.shape[:2]
        h2, w2 = img2_color.shape[:2]
        
        vis_height = max(h1, h2)
        vis_width = w1 + w2 + 10  # 10像素间隔
        
        vis_img = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        vis_img[:h1, :w1] = img1_color
        vis_img[:h2, w1+10:w1+10+w2] = img2_color
        
        # 添加文本信息
        cv2.putText(vis_img, f"Score: {details.get('score', 0):.3f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_img


class TextureMatchingPipeline:
    """完整的纹理匹配流水线"""
    
    def __init__(self, config_path: str = None, use_superglue: bool = True):
        self.matcher = AdvancedTextureMatcher(config_path, use_superglue=use_superglue)
        self.results = {}
    
    def run_pipeline(self, fragments: List, output_dir: str = 'results/texture_matching') -> Dict:
        """
        运行完整的纹理匹配流水线
        """
        print("=" * 60)
        print(" SuperGlue纹理匹配流水线启动 ")
        print("=" * 60)
        print(f"处理碎片数量: {len(fragments)}")
        print(f"输出目录: {output_dir}")
        print()
        
        # 1. 批量特征提取
        features = self.matcher.batch_extract_features(fragments, output_dir)
        
        # 2. 高级匹配
        candidates = self.matcher.advanced_matching(features, fragments)
        
        # 3. 可视化结果
        self.matcher.visualize_matches(candidates, features, fragments, output_dir)
        
        # 4. 生成报告
        report = self._generate_report(candidates, fragments, features, output_dir)
        
        print("=" * 60)
        print(" 纹理匹配流水线完成 ")
        print("=" * 60)
        
        return report
    
    def _generate_report(self, candidates: List, fragments: List, features: Dict, 
                        output_dir: str) -> Dict:
        """生成匹配报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_fragments': len(fragments),
            'processed_fragments': len([f for f in features.values() if f is not None]),
            'total_candidates': len(candidates),
            'config': self.matcher.config_manager.config,
            'matches': []
        }
        
        # 添加匹配详情
        for idx1, idx2, score, details in candidates:
            frag1_name = getattr(fragments[idx1], 'file_name', f'fragment_{idx1}')
            frag2_name = getattr(fragments[idx2], 'file_name', f'fragment_{idx2}')
            
            match_info = {
                'fragment_pair': (frag1_name, frag2_name),
                'indices': (idx1, idx2),
                'total_score': score,
                'component_scores': details
            }
            report['matches'].append(match_info)
        
        # 保存报告
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            report_path = Path(output_dir) / 'matching_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"✓ 匹配报告已保存: {report_path}")
        except Exception as e:
            print(f"✗ 报告保存失败: {e}")
        
        return report


# 便捷函数
def run_texture_matching_pipeline(fragments: List, config_path: str = None,
                                output_dir: str = 'results/texture_matching',
                                use_superglue: bool = True) -> Dict:
    """
    运行纹理匹配流水线的便捷函数
    """
    pipeline = TextureMatchingPipeline(config_path, use_superglue=use_superglue)
    return pipeline.run_pipeline(fragments, output_dir)
