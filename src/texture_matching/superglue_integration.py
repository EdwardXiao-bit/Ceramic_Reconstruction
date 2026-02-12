"""
SuperGlue纹样匹配集成模块
实现文档中(二)纹样提取与特征编码部分
"""
import numpy as np
import torch
import cv2
from pathlib import Path
import open3d as o3d
from typing import List, Tuple, Dict, Optional
import warnings

# SuperGlue相关导入（延迟导入，避免依赖问题）
try:
    from superglue.models.matching import Matching
    from superglue.models.utils import frame2tensor
    SUPERGLUE_AVAILABLE = True
except ImportError:
    SUPERGLUE_AVAILABLE = False
    warnings.warn("SuperGlue not available. Please install SuperGluePretrainedNetwork")


class TextureMatcher:
    """纹样匹配器 - SuperGlue实现"""
    
    def __init__(self, config: Dict = None):
        """
        初始化纹样匹配器
        :param config: SuperGlue配置参数
        """
        self.config = config or self._default_config()
        self.matcher = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if SUPERGLUE_AVAILABLE:
            self._init_superglue()
        else:
            print("[纹理匹配] SuperGlue不可用，使用备用方案")
    
    def _default_config(self) -> Dict:
        """默认SuperGlue配置"""
        return {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',  # or 'outdoor'
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
    
    def _init_superglue(self):
        """初始化SuperGlue匹配器"""
        try:
            self.matcher = Matching(self.config).eval().to(self.device)
            print(f"[纹理匹配] SuperGlue初始化成功 (设备: {self.device})")
        except Exception as e:
            print(f"[纹理匹配] SuperGlue初始化失败: {e}")
            self.matcher = None
    
    def extract_texture_region(self, fragment, visualize: bool = False) -> Optional[np.ndarray]:
        """
        提取碎片表面纹样区域
        对应文档(二)1.纹样区域提取
        """
        if not hasattr(fragment, 'point_cloud') or fragment.point_cloud is None:
            print(f"[纹理提取] 碎片{fragment.id}无点云数据")
            return None
        
        # 简化的纹样区域提取（实际项目中可使用更复杂的分割方法）
        pcd = fragment.point_cloud
        points = np.asarray(pcd.points)
        
        # 检查是否有颜色信息（PLY文件可能包含）
        has_colors = hasattr(pcd, 'colors') and len(pcd.colors) > 0
        if has_colors:
            colors = np.asarray(pcd.colors)
            print(f"[纹理提取] 检测到颜色信息: {colors.shape}")
        
        # 基于Z坐标和曲率的简单分割
        z_coords = points[:, 2]
        z_threshold = np.percentile(z_coords, 70)  # 取较高区域
        
        # 简单的平面拟合来识别表面
        if len(points) > 3:
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                   ransac_n=3,
                                                   num_iterations=1000)
            texture_mask = np.array(inliers)
        else:
            texture_mask = z_coords > z_threshold
        
        if np.sum(texture_mask) == 0:
            print(f"[纹理提取] 碎片{fragment.id}未找到有效纹样区域")
            return None
        
        # 提取纹理区域点云
        texture_points = points[texture_mask]
        
        # 如果有颜色信息，也可以提取颜色特征
        if has_colors:
            texture_colors = colors[texture_mask]
            # 可以在这里添加基于颜色的纹理分析
            
        if visualize and len(texture_points) > 0:
            self._visualize_texture_region(fragment, texture_points)
        
        return texture_points
    
    def _visualize_texture_region(self, fragment, texture_points):
        """可视化纹样区域"""
        texture_pcd = o3d.geometry.PointCloud()
        texture_pcd.points = o3d.utility.Vector3dVector(texture_points)
        texture_pcd.paint_uniform_color([1, 0, 0])  # 红色标记纹样区域
        
        o3d.visualization.draw_geometries(
            [fragment.point_cloud, texture_pcd],
            window_name=f"碎片{fragment.id} - 纹样区域",
            width=800,
            height=600
        )
    
    def project_to_image(self, texture_points: np.ndarray, 
                        resolution: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        将3D纹样区域投影到2D图像
        对应文档(二)2.关键区域裁剪/规范化
        """
        if texture_points is None or len(texture_points) == 0:
            return None
        
        # 简单的正交投影
        # 实际项目中应该使用相机姿态进行精确投影
        points_2d = texture_points[:, :2]  # XY平面投影
        
        # 归一化到[0,1]范围
        min_coords = np.min(points_2d, axis=0)
        max_coords = np.max(points_2d, axis=0)
        normalized_points = (points_2d - min_coords) / (max_coords - min_coords + 1e-8)
        
        # 创建图像
        img = np.zeros(resolution, dtype=np.uint8)
        height, width = resolution
        
        # 将点绘制到图像上
        for point in normalized_points:
            x = int(point[0] * (width - 1))
            y = int(point[1] * (height - 1))
            if 0 <= x < width and 0 <= y < height:
                img[y, x] = 255
        
        # 膨胀操作使点连接
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        
        return img
    
    def extract_superglue_features(self, image: np.ndarray) -> Optional[Dict]:
        """
        使用SuperGlue提取图像特征
        对应文档(二)3.1 SuperPoint提取图像关键点
        """
        if image is None or self.matcher is None:
            return None
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        # 转换为tensor
        inp = frame2tensor(image_rgb, self.device)
        
        # 提取特征
        with torch.no_grad():
            pred = self.matcher({'image0': inp})
        
        # 提取关键点和描述子
        keypoints = pred['keypoints0'][0].cpu().numpy()
        scores = pred['scores0'][0].cpu().numpy()
        descriptors = pred['descriptors0'][0].cpu().numpy()
        
        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
            'image_shape': image.shape[:2]
        }
    
    def match_textures(self, features1: Dict, features2: Dict) -> Optional[Dict]:
        """
        纹样匹配
        对应文档(四)纹样相似度计算
        """
        if features1 is None or features2 is None:
            return None
        
        # 使用SuperGlue进行匹配
        data = {
            'image0': frame2tensor(np.zeros(features1['image_shape']), self.device),
            'image1': frame2tensor(np.zeros(features2['image_shape']), self.device)
        }
        data['keypoints0'] = torch.tensor(features1['keypoints']).unsqueeze(0).to(self.device)
        data['keypoints1'] = torch.tensor(features2['keypoints']).unsqueeze(0).to(self.device)
        data['descriptors0'] = torch.tensor(features1['descriptors']).unsqueeze(0).to(self.device)
        data['descriptors1'] = torch.tensor(features2['descriptors']).unsqueeze(0).to(self.device)
        data['scores0'] = torch.tensor(features1['scores']).unsqueeze(0).to(self.device)
        data['scores1'] = torch.tensor(features2['scores']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred = self.matcher(data)
        
        # 提取匹配结果
        matches0 = pred['matches0'][0].cpu().numpy()
        matching_scores = pred['matching_scores0'][0].cpu().numpy()
        
        # 过滤有效匹配
        valid_matches = matches0 > -1
        if np.sum(valid_matches) == 0:
            return None
        
        # 构建匹配对
        mkpts0 = features1['keypoints'][valid_matches]
        mkpts1 = features2['keypoints'][matches0[valid_matches]]
        match_scores = matching_scores[valid_matches]
        
        return {
            'matches': list(zip(mkpts0, mkpts1)),
            'scores': match_scores,
            'num_matches': len(mkpts0),
            'confidence': np.mean(match_scores) if len(match_scores) > 0 else 0
        }
    
    def compute_texture_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        计算纹样相似度
        对应文档(四)4.2 纹样相似度S_tex
        """
        match_result = self.match_textures(features1, features2)
        if match_result is None:
            return 0.0
        
        # 基于匹配数量和置信度计算相似度
        num_matches = match_result['num_matches']
        confidence = match_result['confidence']
        
        # 简单的相似度计算（可进一步优化）
        similarity = min(num_matches / 50.0, 1.0) * confidence
        return similarity


class PatternEncoder:
    """纹样全局特征编码器"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def extract_global_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取纹样全局embedding
        对应文档(二)3.2 纹样patch全局表示
        """
        if image is None:
            return None
        
        # 简化的CNN特征提取（实际项目中可使用预训练的ResNet/ViT）
        # 这里使用手工特征作为示例
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 提取基本统计特征
        features = []
        
        # 1. 直方图特征
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = hist.flatten() / np.sum(hist)
        features.extend(hist)
        
        # 2. 纹理特征（LBP简化版）
        lbp_features = self._extract_lbp_features(gray)
        features.extend(lbp_features)
        
        # 3. 几何特征
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            areas = [cv2.contourArea(cnt) for cnt in contours]
            features.append(np.mean(areas) / (gray.shape[0] * gray.shape[1]))
            features.append(len(contours) / 100.0)  # 归一化轮廓数量
        else:
            features.extend([0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_lbp_features(self, image: np.ndarray, num_points: int = 8) -> List[float]:
        """简化版LBP特征提取"""
        # 简化的LBP实现
        lbp = np.zeros_like(image)
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                center = image[i, j]
                code = 0
                # 检查8个邻居
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                lbp[i, j] = code
        
        # 计算直方图
        hist, _ = np.histogram(lbp.flatten(), bins=32, range=(0, 256))
        hist = hist.astype(np.float32) / np.sum(hist)
        return hist.tolist()
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算全局纹样相似度"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        # 余弦相似度
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)  # 确保非负


def integrate_texture_matching(fragments: List, top_k: int = 10) -> List[Tuple]:
    """
    整合纹样匹配流程
    对应文档(四)碎片匹配初筛中的纹样匹配部分
    """
    print("===== 开始纹样匹配流程 =====")
    
    # 初始化匹配器
    texture_matcher = TextureMatcher()
    pattern_encoder = PatternEncoder()
    
    # 提取纹样特征
    fragment_features = {}
    fragment_embeddings = {}
    
    print("1. 提取各碎片纹样特征...")
    for i, fragment in enumerate(fragments):
        print(f"   处理碎片{i}: {getattr(fragment, 'file_name', f'fragment_{i}')}")
        
        # 提取纹理区域
        texture_points = texture_matcher.extract_texture_region(fragment)
        if texture_points is not None:
            # 投影到图像
            texture_image = texture_matcher.project_to_image(texture_points)
            if texture_image is not None:
                # 提取SuperGlue特征
                superglue_features = texture_matcher.extract_superglue_features(texture_image)
                # 提取全局embedding
                global_embedding = pattern_encoder.extract_global_embedding(texture_image)
                
                fragment_features[i] = superglue_features
                fragment_embeddings[i] = global_embedding
                print(f"      ✓ 提取完成 ({len(superglue_features['keypoints']) if superglue_features else 0} 个关键点)")
            else:
                print(f"      ✗ 图像投影失败")
        else:
            print(f"      ✗ 纹样区域提取失败")
    
    # 计算纹样相似度矩阵
    print("2. 计算纹样相似度...")
    similarities = {}
    valid_indices = list(fragment_features.keys())
    
    for i in range(len(valid_indices)):
        for j in range(i+1, len(valid_indices)):
            idx1, idx2 = valid_indices[i], valid_indices[j]
            
            # SuperGlue匹配相似度
            sg_sim = texture_matcher.compute_texture_similarity(
                fragment_features[idx1], fragment_features[idx2]
            )
            
            # 全局embedding相似度
            emb_sim = pattern_encoder.compute_similarity(
                fragment_embeddings[idx1], fragment_embeddings[idx2]
            )
            
            # 综合相似度（可调整权重）
            combined_sim = 0.7 * sg_sim + 0.3 * emb_sim
            
            similarities[(idx1, idx2)] = combined_sim
            print(f"      碎片{idx1}-碎片{idx2}: {combined_sim:.3f} (SG:{sg_sim:.3f}, Emb:{emb_sim:.3f})")
    
    # 生成候选匹配对
    print("3. 生成候选匹配对...")
    candidates = []
    for (idx1, idx2), similarity in similarities.items():
        if similarity > 0.1:  # 相似度阈值
            candidates.append((idx1, idx2, similarity))
    
    # 按相似度排序并取top-k
    candidates.sort(key=lambda x: x[2], reverse=True)
    top_candidates = candidates[:top_k]
    
    print(f"4. 完成！找到 {len(top_candidates)} 个高相似度候选对")
    for idx1, idx2, sim in top_candidates:
        print(f"      碎片{idx1}-碎片{idx2}: 相似度 {sim:.3f}")
    
    return top_candidates


# 兼容性检查和备用方案
def check_superglue_availability() -> bool:
    """检查SuperGlue可用性"""
    return SUPERGLUE_AVAILABLE

def get_texture_matching_fallback():
    """获取纹样匹配的备用方案"""
    class FallbackTextureMatcher:
        def __init__(self):
            print("[纹理匹配] 使用传统特征匹配作为备用方案")
        
        def extract_features(self, image):
            # 使用传统的SIFT/ORB特征
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # 使用ORB特征
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            return {'keypoints': keypoints, 'descriptors': descriptors}
        
        def match_features(self, feat1, feat2):
            if feat1['descriptors'] is None or feat2['descriptors'] is None:
                return 0.0
            
            # BFMatcher匹配
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(feat1['descriptors'], feat2['descriptors'])
            return len(matches) / max(len(feat1['descriptors']), len(feat2['descriptors']))
    
    return FallbackTextureMatcher()