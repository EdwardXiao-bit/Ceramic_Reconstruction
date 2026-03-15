# D:\ceramic_reconstruction\src\boundary_validation\validator.py
"""
边界验证主控制器
整合所有边界验证功能模块，提供统一的验证接口
"""

import numpy as np
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import asdict

from .config import get_config
from .boundary_extractor import BoundaryExtractor, BoundaryRegion
from .feature_matcher import FeatureMatcher, MatchResult
from .complementarity_checker import ComplementarityChecker, ComplementarityResult
from .local_aligner import LocalAligner, AlignmentResult
from .collision_detector import CollisionDetector, CollisionResult
from .scoring_system import ScoringSystem, ValidationScores

class BoundaryValidator:
    """边界验证主控制器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化边界验证器
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or get_config()
        self.boundary_extractor = BoundaryExtractor(self.config.BOUNDARY_EXTRACTION)
        self.feature_matcher = FeatureMatcher(self.config.FEATURE_MATCHING)
        self.complementarity_checker = ComplementarityChecker(self.config.COMPLEMENTARITY_CHECK)
        self.local_aligner = LocalAligner(self.config.LOCAL_ALIGNMENT)
        self.collision_detector = CollisionDetector(self.config.COLLISION_DETECTION)
        self.scoring_system = ScoringSystem(self.config.FINAL_SCORING)
        
        print("[边界验证器] 初始化完成")
    
    def validate_fragment_pair(self, fragment1: Any, fragment2: Any,
                              initial_transformation: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        验证一对碎片的边界匹配
        
        Args:
            fragment1: 第一个碎片对象
            fragment2: 第二个碎片对象
            initial_transformation: 初始变换矩阵（4x4），如果为None则使用单位矩阵
            
        Returns:
            Dict: 包含完整验证结果的字典
        """
        print("=" * 60)
        print(f"[边界验证] 开始验证碎片对: {getattr(fragment1, 'id', 'frag1')} - {getattr(fragment2, 'id', 'frag2')}")
        print("=" * 60)
        
        start_time = time.time()
        
        # 设置初始变换
        if initial_transformation is None:
            initial_transformation = np.eye(4)
        
        validation_result = {
            'fragment_ids': [getattr(fragment1, 'id', 'frag1'), getattr(fragment2, 'id', 'frag2')],
            'processing_time': 0.0,
            'success': False,
            'error_message': None,
            'boundary_regions': {},
            'intermediate_results': {},
            'final_scores': None,
            'detailed_report': None
        }
        
        try:
            # 1. 边界区域提取
            print("\n[步骤1/6] 边界区域提取...")
            boundary1, boundary2 = self.boundary_extractor.extract_boundary_regions(fragment1, fragment2)
            
            if boundary1 is None or boundary2 is None:
                validation_result['error_message'] = "边界区域提取失败"
                print("[边界验证] 边界提取失败，终止验证")
                return validation_result
            
            validation_result['boundary_regions'] = {
                'fragment1_points': len(boundary1.points),
                'fragment2_points': len(boundary2.points),
                'boundary1_confidence': boundary1.confidence,
                'boundary2_confidence': boundary2.confidence
            }
            
            # 2. 边界特征匹配验证
            print("\n[步骤2/6] 边界特征匹配验证...")
            match_result = self.feature_matcher.match_boundaries(boundary1, boundary2)
            
            if match_result is None:
                validation_result['error_message'] = "特征匹配失败"
                print("[边界验证] 特征匹配失败，终止验证")
                return validation_result
            
            validation_result['intermediate_results']['feature_matching'] = {
                'matches_count': len(match_result.matches),
                'overlap_score': match_result.overlap_score,
                'inlier_ratio': match_result.inlier_ratio,
                'boundary_complementarity_score': match_result.boundary_complementarity_score,
                'initial_transformation': match_result.transformation  # 保存初始变换矩阵
            }
            
            # 3. 局部集合互补性检查
            print("\n[步骤3/6] 局部集合互补性检查...")
            complementarity_result = self.complementarity_checker.check_complementarity(
                boundary1, boundary2, match_result
            )
            
            validation_result['intermediate_results']['complementarity_check'] = {
                'normal_complementarity_score': complementarity_result.normal_complementarity_score,
                'shape_complementarity_score': complementarity_result.shape_complementarity_score,
                'average_normal_angle': complementarity_result.average_normal_angle,
                'reverse_normal_ratio': complementarity_result.reverse_normal_ratio,
                'normal_similarity_stats': complementarity_result.normal_similarity_stats,  # 法向相似度统计
                'shape_similarity_stats': complementarity_result.shape_similarity_stats  # 形状相似度统计
            }
            
            # 4. 局部对齐精化
            print("\n[步骤4/6] 局部对齐精化...")
            alignment_result = self.local_aligner.refine_alignment(
                fragment1, fragment2, match_result.transformation
            )
            
            validation_result['intermediate_results']['local_alignment'] = {
                'fitness_score': alignment_result.fitness_score,
                'rmse': alignment_result.rmse,
                'iterations_used': alignment_result.iterations_used,
                'convergence_status': alignment_result.convergence_status,
                'refined_transformation': alignment_result.refined_transformation,  # 保存精化后的变换矩阵
                'alignment_error': alignment_result.alignment_error  # 对齐误差
            }
            
            # 5. 碰撞与穿透检查
            print("\n[步骤5/6] 碰撞与穿透检查...")
            collision_result = self.collision_detector.check_collision(
                fragment1, fragment2, alignment_result.refined_transformation
            )
            
            validation_result['intermediate_results']['collision_detection'] = {
                'collision_score': collision_result.collision_score,
                'penetration_depth': collision_result.penetration_depth,
                'collision_points_count': len(collision_result.collision_points),
                'safety_margin': collision_result.safety_margin,
                'collision_volumes': collision_result.collision_volumes,  # 碰撞体积信息
                'detailed_analysis': collision_result.detailed_analysis  # 详细分析结果
            }
            
            # 6. 综合边界验证评分
            print("\n[步骤6/6] 综合边界验证评分...")
            final_scores = self.scoring_system.calculate_comprehensive_score(
                match_result, complementarity_result, alignment_result, collision_result
            )
            
            validation_result['final_scores'] = asdict(final_scores)
            validation_result['success'] = True
            
            # 生成详细报告
            detailed_report = self.scoring_system.generate_detailed_report(final_scores)
            validation_result['detailed_report'] = detailed_report
            
            # 计算处理时间
            processing_time = time.time() - start_time
            validation_result['processing_time'] = processing_time
            
            print("\n" + "=" * 60)
            print("[边界验证] 验证完成!")
            print(f"  处理时间: {processing_time:.2f}秒")
            print(f"  综合得分: {final_scores.total_score:.3f}")
            print(f"  验证状态: {final_scores.validation_status}")
            print("=" * 60)
            
            return validation_result
            
        except Exception as e:
            error_msg = f"验证过程中发生错误: {str(e)}"
            validation_result['error_message'] = error_msg
            validation_result['success'] = False
            print(f"[边界验证] 错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return validation_result
    
    def validate_multiple_pairs(self, fragment_pairs: List[Tuple[Any, Any]], 
                               transformations: Optional[List[np.ndarray]] = None) -> List[Dict[str, Any]]:
        """
        验证多对碎片
        
        Args:
            fragment_pairs: 碎片对列表 [(frag1, frag2), ...]
            transformations: 对应的初始变换矩阵列表
            
        Returns:
            List[Dict]: 验证结果列表
        """
        print(f"[批量验证] 开始验证 {len(fragment_pairs)} 对碎片...")
        
        results = []
        
        for i, (frag1, frag2) in enumerate(fragment_pairs):
            print(f"\n[批量验证] 处理第 {i+1}/{len(fragment_pairs)} 对...")
            
            # 获取对应的变换矩阵
            transform = transformations[i] if transformations and i < len(transformations) else None
            
            # 执行验证
            result = self.validate_fragment_pair(frag1, frag2, transform)
            results.append(result)
        
        # 生成批量比较报告
        successful_results = [r for r in results if r['success']]
        if successful_results:
            final_scores = [ValidationScores(**r['final_scores']) for r in successful_results]
            comparison = self.scoring_system.compare_multiple_matches(final_scores)
            print(f"\n[批量验证] 最佳匹配得分: {comparison['best_match_score']:.3f}")
            print(f"[批量验证] 平均得分: {comparison['score_statistics']['mean']:.3f}")
        
        print(f"[批量验证] 完成，成功验证 {len(successful_results)}/{len(results)} 对")
        return results
    
    def save_validation_result(self, result: Dict[str, Any], output_path: str) -> bool:
        """
        保存验证结果到文件
        
        Args:
            result: 验证结果字典
            output_path: 输出文件路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换numpy数组为列表以便JSON序列化
            serializable_result = self._make_serializable(result)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            
            print(f"[结果保存] 验证结果已保存到: {output_path}")
            return True
            
        except Exception as e:
            print(f"[结果保存] 保存失败: {e}")
            return False
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可序列化的格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def load_validation_result(self, input_path: str) -> Optional[Dict[str, Any]]:
        """
        从文件加载验证结果
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            Dict: 验证结果字典，如果加载失败返回None
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            print(f"[结果加载] 成功加载验证结果: {input_path}")
            return result
        except Exception as e:
            print(f"[结果加载] 加载失败: {e}")
            return None
    
    def visualize_validation_process(self, fragment1: Any, fragment2: Any,
                                   result: Dict[str, Any]) -> None:
        """
        可视化验证过程（如果启用）
        """
        if not self.config['VISUALIZATION'].get('enabled', False):
            return
            
        print("[可视化] 显示验证过程...")
        # 这里可以集成可视化代码
        # 例如使用Open3D显示边界提取、匹配结果等
    
    def get_validation_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取验证统计信息
        
        Args:
            results: 验证结果列表
            
        Returns:
            Dict: 统计信息
        """
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {'message': '没有成功的验证结果'}
        
        # 提取分数
        total_scores = [r['final_scores']['total_score'] for r in successful_results]
        processing_times = [r['processing_time'] for r in successful_results]
        
        statistics = {
            'total_validations': len(results),
            'successful_validations': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'score_statistics': {
                'mean': float(np.mean(total_scores)),
                'std': float(np.std(total_scores)),
                'min': float(np.min(total_scores)),
                'max': float(np.max(total_scores)),
                'median': float(np.median(total_scores))
            },
            'time_statistics': {
                'mean': float(np.mean(processing_times)),
                'std': float(np.std(processing_times)),
                'min': float(np.min(processing_times)),
                'max': float(np.max(processing_times))
            },
            'validation_status_distribution': self._get_status_distribution(successful_results)
        }
        
        return statistics
    
    def _get_status_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """获取验证状态分布"""
        status_counts = {}
        for result in results:
            status = result['final_scores']['validation_status']
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts
    
    def update_configuration(self, new_config: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            new_config: 新的配置字典
        """
        self.config.update(new_config)
        
        # 重新初始化各模块
        self.boundary_extractor = BoundaryExtractor(self.config['BOUNDARY_EXTRACTION'])
        self.feature_matcher = FeatureMatcher(self.config['FEATURE_MATCHING'])
        self.complementarity_checker = ComplementarityChecker(self.config['COMPLEMENTARITY_CHECK'])
        self.local_aligner = LocalAligner(self.config['LOCAL_ALIGNMENT'])
        self.collision_detector = CollisionDetector(self.config['COLLISION_DETECTION'])
        self.scoring_system = ScoringSystem(self.config['FINAL_SCORING'])
        
        print("[配置更新] 配置已更新并重新初始化所有模块")

# 便捷函数
def create_validator(config: Optional[Dict[str, Any]] = None) -> BoundaryValidator:
    """
    创建边界验证器实例的便捷函数
    
    Args:
        config: 配置字典
        
    Returns:
        BoundaryValidator: 验证器实例
    """
    return BoundaryValidator(config)

def quick_validate_pair(fragment1: Any, fragment2: Any) -> Dict[str, Any]:
    """
    快速验证一对碎片的便捷函数
    
    Args:
        fragment1: 第一个碎片
        fragment2: 第二个碎片
        
    Returns:
        Dict: 验证结果
    """
    validator = BoundaryValidator()
    return validator.validate_fragment_pair(fragment1, fragment2)