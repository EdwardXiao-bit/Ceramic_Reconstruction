"""
增强版SuperGlue集成模块
支持真正的SuperPoint/SuperGlue特征提取
"""
import numpy as np
import torch
import cv2
from pathlib import Path
import open3d as o3d
from typing import List, Tuple, Dict, Optional
import warnings

# SuperGlue相关导入（增强版）
SUPERGLUE_AVAILABLE = False
SUPPRESS_WARNINGS = True

try:
    # 尝试导入SuperGlue
    from models.matching import Matching
    from models.utils import frame2tensor
    SUPERGLUE_AVAILABLE = True
    if not SUPPRESS_WARNINGS:
        print("[SuperGlue] SuperGlue模型可用")
except ImportError as e:
    if not SUPPRESS_WARNINGS:
        print(f"[SuperGlue] SuperGlue导入失败: {e}")
    SUPERGLUE_AVAILABLE = False

# 传统特征作为备用
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    if not SUPPRESS_WARNINGS:
        print("[纹理匹配] OpenCV不可用")


class EnhancedTextureMatcher:
    """增强版纹理匹配器 - 支持SuperGlue"""
    
    def __init__(self, use_superglue: bool = True, config: Dict = None):
        """
        初始化增强版纹理匹配器
        :param use_superglue: 是否优先使用SuperGlue
        :param config: 配置参数
        """
        self.use_superglue = use_superglue and SUPERGLUE_AVAILABLE
        self.config = config or self._default_config()
        self.matcher = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.use_superglue:
            self._init_superglue()
        else:
            print(f"[纹理匹配] 使用传统特征匹配 (SuperGlue: {'不可用' if not SUPERGLUE_AVAILABLE else '已禁用'})")
    
    def _default_config(self) -> Dict:
        """默认配置"""
        base_config = {
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
        
        if self.use_superglue:
            return base_config
        else:
            # 传统特征配置
            return {
                'orb': {
                    'nfeatures': 1000,
                    'scaleFactor': 1.2,
                    'nlevels': 8,
                    'edgeThreshold': 31,
                    'firstLevel': 0,
                    'WTA_K': 2,
                    'scoreType': cv2.ORB_HARRIS_SCORE,
                    'patchSize': 31,
                    'fastThreshold': 20
                }
            }
    
    def _init_superglue(self):
        """初始化SuperGlue匹配器"""
        try:
            self.matcher = Matching(self.config).eval().to(self.device)
            print(f"[SuperGlue] 初始化成功 (设备: {self.device})")
        except Exception as e:
            print(f"[SuperGlue] 初始化失败: {e}")
            self.use_superglue = False
            self.matcher = None
    
    def extract_features(self, image: np.ndarray) -> Optional[Dict]:
        """
        提取图像特征（自动选择SuperGlue或传统方法）
        """
        if image is None:
            return None
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        if self.use_superglue and self.matcher is not None:
            return self._extract_superglue_features(image_rgb)
        else:
            return self._extract_traditional_features(image)
    
    def _extract_superglue_features(self, image_rgb: np.ndarray) -> Dict:
        """使用SuperGlue提取特征"""
        try:
            # 确保输入是正确的numpy数组格式
            if len(image_rgb.shape) == 3:
                # RGB图像 [H, W, 3]
                if image_rgb.shape[2] == 3:
                    # 正确的RGB格式
                    pass
                elif image_rgb.shape[2] > 3:
                    # 多于3个通道，取前3个
                    image_rgb = image_rgb[:, :, :3]
                else:
                    # 少于3个通道，转换为RGB
                    if image_rgb.shape[2] == 1:
                        image_rgb = cv2.cvtColor(image_rgb.squeeze(), cv2.COLOR_GRAY2RGB)
                    else:
                        raise ValueError(f"Unsupported channel count: {image_rgb.shape[2]}")
            elif len(image_rgb.shape) == 2:
                # 灰度图像 [H, W]，转换为RGB
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
            else:
                raise ValueError(f"Unsupported image shape: {image_rgb.shape}")
            
            # 手动创建正确的tensor格式 [1, 3, H, W]
            # 转换为float32并归一化到[0,1]
            image_float = image_rgb.astype(np.float32) / 255.0
            
            # 转换维度顺序 HWC -> CHW
            image_chw = np.transpose(image_float, (2, 0, 1))
            
            # 添加批次维度
            image_tensor = torch.from_numpy(image_chw).unsqueeze(0)
            
            # 确保在正确的设备上
            inp = image_tensor.to(self.device)
            
            # 验证最终形状
            if inp.dim() != 4 or inp.shape[1] != 3:
                raise ValueError(f"Final tensor shape {inp.shape} is invalid, expected [1, 3, H, W]")
            
            # 提取特征
            with torch.no_grad():
                pred = self.matcher({'image0': inp})
            
            # 提取关键点和描述子
            keypoints = pred['keypoints0'][0].cpu().numpy()
            scores = pred['scores0'][0].cpu().numpy()
            descriptors = pred['descriptors0'][0].cpu().numpy()
            
            return {
                'method': 'superglue',
                'keypoints': keypoints,
                'scores': scores,
                'descriptors': descriptors,
                'image_shape': image_rgb.shape[:2]
            }
        except Exception as e:
            print(f"[SuperGlue] 特征提取失败: {e}")
            # 降级到传统方法
            gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if len(image_rgb.shape) == 3 else image_rgb
            return self._extract_traditional_features(gray_image)
    
    def _extract_traditional_features(self, image: np.ndarray) -> Dict:
        """使用传统方法提取特征"""
        if not OPENCV_AVAILABLE:
            return None
            
        # 使用ORB特征
        orb = cv2.ORB_create(**self.config.get('orb', {}))
        keypoints, descriptors = orb.detectAndCompute(image, None)
        
        if keypoints is None or descriptors is None:
            return None
        
        # 转换为numpy数组
        keypoints_np = np.array([kp.pt for kp in keypoints])
        scores = np.array([kp.response for kp in keypoints])
        
        return {
            'method': 'orb',
            'keypoints': keypoints_np,
            'scores': scores,
            'descriptors': descriptors,
            'image_shape': image.shape[:2]
        }
    
    def match_features(self, features1: Dict, features2: Dict) -> Optional[Dict]:
        """
        匹配特征点
        """
        if features1 is None or features2 is None:
            return None
        
        if features1['method'] == 'superglue' and features2['method'] == 'superglue':
            return self._match_superglue_features(features1, features2)
        else:
            return self._match_traditional_features(features1, features2)
    
    def _match_superglue_features(self, feat1: Dict, feat2: Dict) -> Dict:
        """SuperGlue特征匹配"""
        try:
            # 准备匹配数据
            data = {
                'image0': frame2tensor(np.zeros(feat1['image_shape']), self.device),
                'image1': frame2tensor(np.zeros(feat2['image_shape']), self.device)
            }
            
            data['keypoints0'] = torch.tensor(feat1['keypoints']).unsqueeze(0).to(self.device)
            data['keypoints1'] = torch.tensor(feat2['keypoints']).unsqueeze(0).to(self.device)
            data['descriptors0'] = torch.tensor(feat1['descriptors']).unsqueeze(0).to(self.device)
            data['descriptors1'] = torch.tensor(feat2['descriptors']).unsqueeze(0).to(self.device)
            data['scores0'] = torch.tensor(feat1['scores']).unsqueeze(0).to(self.device)
            data['scores1'] = torch.tensor(feat2['scores']).unsqueeze(0).to(self.device)
            
            # 执行匹配
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
            mkpts0 = feat1['keypoints'][valid_matches]
            mkpts1 = feat2['keypoints'][matches0[valid_matches]]
            match_scores = matching_scores[valid_matches]
            
            return {
                'method': 'superglue',
                'matches': list(zip(mkpts0, mkpts1)),
                'scores': match_scores,
                'num_matches': len(mkpts0),
                'confidence': np.mean(match_scores) if len(match_scores) > 0 else 0
            }
            
        except Exception as e:
            print(f"[SuperGlue] 匹配失败: {e}")
            # 降级到传统匹配
            return self._match_traditional_features(feat1, feat2)
    
    def _match_traditional_features(self, feat1: Dict, feat2: Dict) -> Dict:
        """传统特征匹配"""
        if not OPENCV_AVAILABLE:
            return None
            
        # 使用BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(feat1['descriptors'], feat2['descriptors'])
        
        if len(matches) == 0:
            return None
        
        # 按距离排序
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 提取匹配点
        mkpts0 = np.float32([feat1['keypoints'][m.queryIdx] for m in matches])
        mkpts1 = np.float32([feat2['keypoints'][m.trainIdx] for m in matches])
        scores = np.array([1.0 / (1.0 + m.distance) for m in matches])  # 距离越小分数越高
        
        return {
            'method': 'traditional',
            'matches': list(zip(mkpts0, mkpts1)),
            'scores': scores,
            'num_matches': len(matches),
            'confidence': np.mean(scores) if len(scores) > 0 else 0
        }


# 便捷函数
def create_enhanced_texture_matcher(use_superglue: bool = True) -> EnhancedTextureMatcher:
    """创建增强版纹理匹配器"""
    return EnhancedTextureMatcher(use_superglue=use_superglue)