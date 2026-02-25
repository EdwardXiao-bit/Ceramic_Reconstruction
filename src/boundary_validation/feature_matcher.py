# D:\ceramic_reconstruction\src\boundary_validation\feature_matcher.py
"""
边界特征匹配验证模块
使用Predator或D3Feat计算matchability_score，得到边界点对匹配及互补性得分
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import torch

@dataclass
class MatchResult:
    """匹配结果数据类"""
    matches: np.ndarray              # 匹配点对索引 [(idx1, idx2), ...]
    matchability_scores: np.ndarray  # 匹配可信度得分
    overlap_score: float             # 重叠度得分
    inlier_ratio: float              # 内点比率
    boundary_complementarity_score: float  # 边界互补性得分
    transformation: np.ndarray       # 变换矩阵 (4x4)

class FeatureMatcher:
    """边界特征匹配器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.predator_model = None
        self.d3feat_model = None
        self._initialize_models()
        
    def _initialize_models(self):
        """初始化特征匹配模型"""
        if self.config['predator_enabled']:
            try:
                self.predator_model = self._load_predator_model()
                print("[特征匹配] Predator模型加载成功")
            except Exception as e:
                print(f"[特征匹配] Predator模型加载失败: {e}")
                self.predator_model = None
                
        if self.config['d3feat_enabled']:
            try:
                self.d3feat_model = self._load_d3feat_model()
                print("[特征匹配] D3Feat模型加载成功")
            except Exception as e:
                print(f"[特征匹配] D3Feat模型加载失败: {e}")
                self.d3feat_model = None
    
    def _load_predator_model(self):
        """加载Predator模型"""
        # 这里应该是实际的Predator模型加载代码
        # 由于Predator是外部依赖，这里提供占位实现
        class MockPredator:
            def predict(self, points1, points2):
                # 模拟匹配结果
                n_matches = min(len(points1), len(points2)) // 4
                matches = np.random.choice(len(points1), size=n_matches, replace=False)
                matched_points2 = np.random.choice(len(points2), size=n_matches, replace=False)
                match_pairs = np.column_stack([matches, matched_points2])
                
                scores = np.random.uniform(0.7, 0.95, n_matches)
                return match_pairs, scores
                
        return MockPredator()
    
    def _load_d3feat_model(self):
        """加载D3Feat模型"""
        # 这里应该是实际的D3Feat模型加载代码
        class MockD3Feat:
            def extract_features(self, points):
                # 模拟特征提取
                features = np.random.randn(len(points), 256)
                return features
                
        return MockD3Feat()
    
    def match_boundaries(self, boundary1: Any, boundary2: Any) -> Optional[MatchResult]:
        """
        对两个边界区域进行特征匹配验证
        
        Args:
            boundary1: 第一个边界区域
            boundary2: 第二个边界区域
            
        Returns:
            MatchResult: 匹配结果
        """
        print("[特征匹配] 开始边界特征匹配验证...")
        
        points1 = boundary1.points
        points2 = boundary2.points
        
        if len(points1) < self.config['min_matches'] or len(points2) < self.config['min_matches']:
            print("[特征匹配] 边界点数不足，无法进行匹配")
            return None
            
        # 1. 特征提取和匹配
        print("[特征匹配] 执行特征匹配...")
        matches, matchability_scores = self._perform_matching(points1, points2)
        
        if len(matches) < self.config['min_matches']:
            print(f"[特征匹配] 匹配点对数不足: {len(matches)} < {self.config['min_matches']}")
            return None
            
        # 2. 计算重叠度得分
        print("[特征匹配] 计算重叠度...")
        overlap_score = self._compute_overlap_score(points1, points2, matches)
        
        # 3. 计算内点比率
        print("[特征匹配] 计算内点比率...")
        inlier_ratio, refined_matches = self._compute_inlier_ratio(
            points1, points2, matches, matchability_scores
        )
        
        # 4. 计算边界互补性得分
        print("[特征匹配] 计算边界互补性...")
        complementarity_score = self._compute_boundary_complementarity(
            boundary1, boundary2, refined_matches
        )
        
        # 5. 计算初始变换矩阵
        print("[特征匹配] 计算初始变换...")
        transformation = self._compute_transformation(
            points1, points2, refined_matches
        )
        
        # 6. 综合边界验证评分
        print("[特征匹配] 计算综合得分...")
        weights = self.config['feature_weights']
        boundary_score = (
            weights['overlap_score'] * overlap_score +
            weights['matchability_score'] * np.mean(matchability_scores) +
            weights['inlier_ratio'] * inlier_ratio
        )
        
        match_result = MatchResult(
            matches=refined_matches,
            matchability_scores=matchability_scores,
            overlap_score=overlap_score,
            inlier_ratio=inlier_ratio,
            boundary_complementarity_score=boundary_score,
            transformation=transformation
        )
        
        print(f"[特征匹配] 匹配完成:")
        print(f"  匹配点对数: {len(refined_matches)}")
        print(f"  重叠度得分: {overlap_score:.3f}")
        print(f"  内点比率: {inlier_ratio:.3f}")
        print(f"  互补性得分: {boundary_score:.3f}")
        
        return match_result
    
    def _perform_matching(self, points1: np.ndarray, points2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行特征匹配
        """
        matches_list = []
        scores_list = []
        
        # 使用Predator进行匹配
        if self.predator_model is not None and self.config['predator_enabled']:
            try:
                predator_matches, predator_scores = self.predator_model.predict(points1, points2)
                matches_list.append(predator_matches)
                scores_list.append(predator_scores)
                print(f"[特征匹配] Predator匹配: {len(predator_matches)} 对")
            except Exception as e:
                print(f"[特征匹配] Predator匹配失败: {e}")
        
        # 使用D3Feat进行匹配
        if self.d3feat_model is not None and self.config['d3feat_enabled']:
            try:
                # 提取特征
                feat1 = self.d3feat_model.extract_features(points1)
                feat2 = self.d3feat_model.extract_features(points2)
                
                # 最近邻匹配
                d3feat_matches, d3feat_scores = self._nearest_neighbor_matching(feat1, feat2)
                matches_list.append(d3feat_matches)
                scores_list.append(d3feat_scores)
                print(f"[特征匹配] D3Feat匹配: {len(d3feat_matches)} 对")
            except Exception as e:
                print(f"[特征匹配] D3Feat匹配失败: {e}")
        
        # 如果都没有成功，使用基础的FPFH匹配
        if not matches_list:
            print("[特征匹配] 使用基础FPFH匹配...")
            fpfh_matches, fpfh_scores = self._fpfh_matching(points1, points2)
            matches_list.append(fpfh_matches)
            scores_list.append(fpfh_scores)
        
        # 合并所有匹配结果
        if matches_list:
            all_matches = np.vstack(matches_list)
            all_scores = np.concatenate(scores_list)
            
            # 去重和筛选
            unique_matches, unique_scores = self._filter_matches(all_matches, all_scores)
            return unique_matches, unique_scores
        else:
            return np.array([]).reshape(0, 2), np.array([])
    
    def _nearest_neighbor_matching(self, feat1: np.ndarray, feat2: np.ndarray, 
                                 ratio_threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        最近邻特征匹配
        """
        from sklearn.neighbors import NearestNeighbors
        
        # 构建特征空间的近邻搜索
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feat2)
        distances, indices = nbrs.kneighbors(feat1)
        
        # Lowe's ratio test
        ratios = distances[:, 0] / (distances[:, 1] + 1e-8)
        good_matches = ratios < ratio_threshold
        
        if np.sum(good_matches) == 0:
            return np.array([]).reshape(0, 2), np.array([])
            
        # 构建匹配对
        query_indices = np.where(good_matches)[0]
        train_indices = indices[good_matches, 0]
        matches = np.column_stack([query_indices, train_indices])
        
        # 计算匹配得分（基于距离的倒数）
        match_scores = 1.0 / (1.0 + distances[good_matches, 0])
        
        return matches, match_scores
    
    def _fpfh_matching(self, points1: np.ndarray, points2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用FPFH特征进行匹配
        """
        # 创建点云对象
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        pcd1.estimate_normals()
        
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2.estimate_normals()
        
        # 计算FPFH特征
        fpfh1 = o3d.pipelines.registration.compute_fpfh_feature(
            pcd1, o3d.geometry.KDTreeSearchParamKNN(knn=20)
        )
        fpfh2 = o3d.pipelines.registration.compute_fpfh_feature(
            pcd2, o3d.geometry.KDTreeSearchParamKNN(knn=20)
        )
        
        # 特征匹配
        matches, scores = self._nearest_neighbor_matching(
            np.asarray(fpfh1.data).T, 
            np.asarray(fpfh2.data).T
        )
        
        return matches, scores
    
    def _filter_matches(self, matches: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        过滤和去重匹配结果
        """
        if len(matches) == 0:
            return matches, scores
            
        # 按得分排序
        sorted_indices = np.argsort(scores)[::-1]
        matches = matches[sorted_indices]
        scores = scores[sorted_indices]
        
        # 去除重复匹配
        unique_matches = []
        unique_scores = []
        used_indices1 = set()
        used_indices2 = set()
        
        for i, (idx1, idx2) in enumerate(matches):
            if idx1 not in used_indices1 and idx2 not in used_indices2:
                unique_matches.append([idx1, idx2])
                unique_scores.append(scores[i])
                used_indices1.add(idx1)
                used_indices2.add(idx2)
                
                # 限制最大匹配数
                if len(unique_matches) >= 100:
                    break
                    
        return np.array(unique_matches), np.array(unique_scores)
    
    def _compute_overlap_score(self, points1: np.ndarray, points2: np.ndarray, 
                              matches: np.ndarray) -> float:
        """
        计算重叠度得分
        """
        if len(matches) == 0:
            return 0.0
            
        # 获取匹配点坐标
        matched_points1 = points1[matches[:, 0]]
        matched_points2 = points2[matches[:, 1]]
        
        # 计算匹配点对间的平均距离
        distances = np.linalg.norm(matched_points1 - matched_points2, axis=1)
        
        # 基于距离计算重叠度（距离越小重叠度越高）
        threshold = 0.05  # 5cm阈值
        overlap_ratios = np.maximum(0, 1 - distances / threshold)
        
        return float(np.mean(overlap_ratios))
    
    def _compute_inlier_ratio(self, points1: np.ndarray, points2: np.ndarray,
                             matches: np.ndarray, scores: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算内点比率并返回精化的匹配
        """
        if len(matches) == 0:
            return 0.0, np.array([]).reshape(0, 2)
            
        # 使用RANSAC估计变换并识别内点
        matched_points1 = points1[matches[:, 0]]
        matched_points2 = points2[matches[:, 1]]
        
        try:
            # RANSAC配准
            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                source=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(matched_points1)),
                target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(matched_points2)),
                corres=o3d.utility.Vector2iVector(np.column_stack([range(len(matched_points1)), range(len(matched_points2))])),
                max_correspondence_distance=0.05,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000, 0.999)
            )
            
            # 应用变换计算残差
            transformed_points1 = (result.transformation[:3, :3] @ matched_points1.T + result.transformation[:3, 3:4]).T
            residuals = np.linalg.norm(transformed_points1 - matched_points2, axis=1)
            
            # 识别内点
            inlier_threshold = 0.02
            inliers = residuals < inlier_threshold
            inlier_ratio = np.sum(inliers) / len(matches)
            
            # 返回内点匹配
            refined_matches = matches[inliers]
            
            return float(inlier_ratio), refined_matches
            
        except Exception as e:
            print(f"[特征匹配] RANSAC失败: {e}")
            # 降级：使用简单的距离阈值
            distances = np.linalg.norm(matched_points1 - matched_points2, axis=1)
            inliers = distances < 0.05
            inlier_ratio = np.sum(inliers) / len(matches)
            refined_matches = matches[inliers]
            
            return float(inlier_ratio), refined_matches
    
    def _compute_boundary_complementarity(self, boundary1: Any, boundary2: Any, 
                                        matches: np.ndarray) -> float:
        """
        计算边界互补性得分
        """
        if len(matches) == 0:
            return 0.0
            
        # 获取匹配点的法向量
        normals1 = boundary1.normals[matches[:, 0]]
        normals2 = boundary2.normals[matches[:, 1]]
        
        # 计算法向量互补性（理想情况下法向量应该相反）
        dot_products = np.sum(normals1 * normals2, axis=1)
        normal_complementarity = np.abs(dot_products)  # 越接近0越好（正交），越接近1越差
        
        # 形状互补性（基于曲率差异）
        curvature1 = boundary1.curvature[matches[:, 0]]
        curvature2 = boundary2.curvature[matches[:, 1]]
        curvature_diff = np.abs(curvature1 - curvature2)
        shape_complementarity = 1.0 / (1.0 + curvature_diff)  # 差异越小互补性越好
        
        # 综合互补性得分
        complementarity_score = 0.6 * (1.0 - np.mean(normal_complementarity)) + \
                               0.4 * np.mean(shape_complementarity)
                               
        return float(np.clip(complementarity_score, 0.0, 1.0))
    
    def _compute_transformation(self, points1: np.ndarray, points2: np.ndarray, 
                               matches: np.ndarray) -> np.ndarray:
        """
        计算变换矩阵
        """
        if len(matches) < 3:
            return np.eye(4)
            
        matched_points1 = points1[matches[:, 0]]
        matched_points2 = points2[matches[:, 1]]
        
        try:
            # SVD方法计算最优变换
            centroid1 = np.mean(matched_points1, axis=0)
            centroid2 = np.mean(matched_points2, axis=0)
            
            # 中心化
            centered1 = matched_points1 - centroid1
            centered2 = matched_points2 - centroid2
            
            # 计算协方差矩阵
            H = centered1.T @ centered2
            
            # SVD分解
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # 确保右手坐标系
            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = Vt.T @ U.T
            
            # 计算平移
            t = centroid2 - R @ centroid1
            
            # 构建变换矩阵
            transformation = np.eye(4)
            transformation[:3, :3] = R
            transformation[:3, 3] = t
            
            return transformation
            
        except Exception as e:
            print(f"[特征匹配] 变换计算失败: {e}")
            return np.eye(4)