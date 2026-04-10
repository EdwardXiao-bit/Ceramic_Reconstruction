# D:\ceramic_reconstruction\src\boundary_validation\scoring_system.py
"""
综合边界验证评分系统
融合多个子分数得到最终边界验证评分
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ValidationScores:
    """验证评分数据类"""
    feature_score: float              # 特征匹配得分
    normal_score: float               # 法向互补性得分
    shape_score: float                # 形状互补性得分
    alignment_score: float            # 对齐精度得分
    collision_penalty: float          # 碰撞惩罚得分
    total_score: float                # 综合总分
    component_scores: Dict[str, float]  # 各组件得分详情
    weighted_contributions: Dict[str, float]  # 加权贡献度
    validation_status: str            # 验证状态
    confidence_level: float           # 置信度水平

class ScoringSystem:
    """综合评分系统"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights = config['weights']
        self.thresholds = config['thresholds']

    def calculate_comprehensive_score(self,
                                    feature_result: Any,
                                    complementarity_result: Any,
                                    alignment_result: Any,
                                    collision_result: Any) -> ValidationScores:
        """
        计算综合边界验证评分

        Args:
            feature_result: 特征匹配结果
            complementarity_result: 互补性检查结果
            alignment_result: 对齐精化结果
            collision_result: 碰撞检测结果

        Returns:
            ValidationScores: 综合评分结果
        """
        print("[评分系统] 开始计算综合边界验证评分...")

        # 1. 提取各组件得分
        component_scores = self._extract_component_scores(
            feature_result, complementarity_result, alignment_result, collision_result
        )

        # 2. 应用权重计算加权得分
        weighted_scores = self._calculate_weighted_scores(component_scores)

        # 3. 计算综合总分
        total_score = self._calculate_total_score(weighted_scores)

        # 4. 计算各组件贡献度
        contributions = self._calculate_contributions(weighted_scores, total_score)

        # 5. 确定验证状态
        validation_status = self._determine_validation_status(total_score, component_scores)

        # 6. 计算置信度水平
        confidence_level = self._calculate_confidence_level(component_scores, weighted_scores)

        # 7. 构建最终结果
        validation_scores = ValidationScores(
            feature_score=component_scores['feature_score'],
            normal_score=component_scores['normal_score'],
            shape_score=component_scores['shape_score'],
            alignment_score=component_scores['alignment_score'],
            collision_penalty=component_scores['collision_penalty'],
            total_score=total_score,
            component_scores=component_scores,
            weighted_contributions=contributions,
            validation_status=validation_status,
            confidence_level=confidence_level
        )

        print(f"[评分系统] 评分完成:")
        print(f"  综合得分: {validation_scores.total_score:.3f}")
        print(f"  验证状态: {validation_scores.validation_status}")
        print(f"  置信度: {validation_scores.confidence_level:.3f}")

        return validation_scores

    def _extract_component_scores(self, feature_result, complementarity_result,
                                  alignment_result, collision_result):
        """
        提取各组件分数（修复版）
        修复：collision_penalty语义修正
              - collision_score=0 → 无碰撞 → 好 → 分数应该高
              - 原代码: collision_penalty = 1.0 - collision_score（正确）
              - 但权重是0.10，collision_penalty=1.0时贡献0.10分（正确）
        主要修复：alignment_score计算（原来fitness=0时给0，现在给partial credit）
        """
        scores = {}

        # 1. 特征匹配得分
        if feature_result is not None:
            scores['feature_score'] = float(np.clip(
                feature_result.boundary_complementarity_score, 0.0, 1.0))
        else:
            scores['feature_score'] = 0.0

        # 2. 法向互补性得分
        if complementarity_result is not None:
            scores['normal_score'] = float(np.clip(
                complementarity_result.normal_complementarity_score, 0.0, 1.0))
        else:
            scores['normal_score'] = 0.0

        # 3. 形状互补性得分
        if complementarity_result is not None:
            scores['shape_score'] = float(np.clip(
                complementarity_result.shape_complementarity_score, 0.0, 1.0))
        else:
            scores['shape_score'] = 0.0

        # 4. 对齐精度得分（修复：fitness=0但变换有效时给partial credit）
        if alignment_result is not None:
            fitness = alignment_result.fitness_score
            rmse = alignment_result.rmse

            if fitness > 0:
                # 有实际对齐
                fitness_score = fitness
                rmse_score = 1.0 / (1.0 + rmse * 10)
                scores['alignment_score'] = 0.7 * fitness_score + 0.3 * rmse_score
            elif alignment_result.convergence_status in ('partial', 'converged'):
                # 有运行但fitness低：给基础分
                scores['alignment_score'] = 0.1
            else:
                scores['alignment_score'] = 0.0
        else:
            scores['alignment_score'] = 0.0

        # 5. 碰撞惩罚：1.0=无碰撞（好），0.0=严重碰撞（差）
        if collision_result is not None:
            collision_penalty = 1.0 - float(np.clip(collision_result.collision_score, 0.0, 1.0))
            scores['collision_penalty'] = collision_penalty
        else:
            scores['collision_penalty'] = 0.5  # 未检测时给中性分

        return scores

    def _calculate_weighted_scores(self, component_scores: Dict[str, float]) -> Dict[str, float]:
        """
        计算加权得分
        """
        weighted_scores = {}

        # 应用权重
        weighted_scores['weighted_feature'] = component_scores['feature_score'] * self.weights['feature_score']
        weighted_scores['weighted_normal'] = component_scores['normal_score'] * self.weights['normal_score']
        weighted_scores['weighted_shape'] = component_scores['shape_score'] * self.weights['shape_score']
        weighted_scores['weighted_alignment'] = component_scores['alignment_score'] * self.weights['alignment_score']
        weighted_scores['weighted_collision'] = component_scores['collision_penalty'] * self.weights['collision_penalty']

        return weighted_scores

    def _calculate_total_score(self, weighted_scores: Dict[str, float]) -> float:
        """
        计算综合总分
        """
        # 简单加权求和
        total_score = sum(weighted_scores.values())

        # 确保在[0,1]范围内
        return float(np.clip(total_score, 0.0, 1.0))

    def _calculate_contributions(self, weighted_scores: Dict[str, float],
                               total_score: float) -> Dict[str, float]:
        """
        计算各组件对总分的贡献度
        """
        if total_score <= 0:
            return {key.replace('weighted_', ''): 0.0 for key in weighted_scores.keys()}

        contributions = {}
        for key, weighted_score in weighted_scores.items():
            component_name = key.replace('weighted_', '')
            contributions[component_name] = float(weighted_score / total_score)

        return contributions

    def _determine_validation_status(self, total_score, component_scores):
        """
        验证状态判定（修复版：降低MVP阶段阈值）
        """
        min_total = self.thresholds.get('minimum_total_score', 0.10)

        if total_score >= 0.7:
            return 'excellent_match'
        elif total_score >= 0.5:
            return 'good_match'
        elif total_score >= 0.3:
            return 'acceptable_match'
        elif total_score >= min_total:
            return 'marginal_match'
        elif total_score >= 0.05:
            return 'poor_match'
        else:
            return 'invalid_match'

    def _calculate_confidence_level(self, component_scores: Dict[str, float],
                                  weighted_scores: Dict[str, float]) -> float:
        """
        计算验证置信度水平
        """
        # 基于各组件得分的一致性和稳定性计算置信度

        # 1. 得分一致性（标准差越小置信度越高）
        score_values = list(component_scores.values())
        score_std = np.std(score_values)
        consistency_score = 1.0 / (1.0 + score_std * 2)  # 放大标准差影响

        # 2. 关键组件表现
        critical_components = ['feature_score', 'normal_score', 'alignment_score']
        critical_scores = [component_scores.get(comp, 0.0) for comp in critical_components]
        critical_performance = np.mean(critical_scores)

        # 3. 加权得分分布
        weighted_values = list(weighted_scores.values())
        weight_distribution = 1.0 - np.std(weighted_values) / (np.mean(weighted_values) + 1e-8)

        # 4. 综合置信度
        confidence = 0.4 * consistency_score + 0.4 * critical_performance + 0.2 * weight_distribution

        return float(np.clip(confidence, 0.0, 1.0))

    def generate_detailed_report(self, validation_scores: ValidationScores) -> Dict[str, Any]:
        """
        生成详细的验证报告
        """
        report = {
            'summary': {
                'total_score': validation_scores.total_score,
                'validation_status': validation_scores.validation_status,
                'confidence_level': validation_scores.confidence_level
            },
            'component_scores': validation_scores.component_scores,
            'weighted_contributions': validation_scores.weighted_contributions,
            'detailed_analysis': self._generate_detailed_analysis(validation_scores),
            'recommendations': self._generate_recommendations(validation_scores),
            'quality_assessment': self._assess_overall_quality(validation_scores)
        }

        return report

    def _generate_detailed_analysis(self, validation_scores: ValidationScores) -> Dict[str, Any]:
        """
        生成详细分析
        """
        analysis = {}

        # 各组件详细分析
        component_details = {}
        scores = validation_scores.component_scores

        # 特征匹配分析
        if scores['feature_score'] >= 0.8:
            feature_quality = 'excellent'
        elif scores['feature_score'] >= 0.6:
            feature_quality = 'good'
        elif scores['feature_score'] >= 0.4:
            feature_quality = 'fair'
        else:
            feature_quality = 'poor'

        component_details['feature_matching'] = {
            'quality': feature_quality,
            'strengths': self._identify_strengths(scores['feature_score'], 'feature'),
            'weaknesses': self._identify_weaknesses(scores['feature_score'], 'feature')
        }

        # 法向互补性分析
        normal_quality = 'excellent' if scores['normal_score'] >= 0.8 else \
                        'good' if scores['normal_score'] >= 0.6 else \
                        'fair' if scores['normal_score'] >= 0.4 else 'poor'

        component_details['normal_complementarity'] = {
            'quality': normal_quality,
            'strengths': self._identify_strengths(scores['normal_score'], 'normal'),
            'weaknesses': self._identify_weaknesses(scores['normal_score'], 'normal')
        }

        # 形状互补性分析
        shape_quality = 'excellent' if scores['shape_score'] >= 0.8 else \
                       'good' if scores['shape_score'] >= 0.6 else \
                       'fair' if scores['shape_score'] >= 0.4 else 'poor'

        component_details['shape_complementarity'] = {
            'quality': shape_quality,
            'strengths': self._identify_strengths(scores['shape_score'], 'shape'),
            'weaknesses': self._identify_weaknesses(scores['shape_score'], 'shape')
        }

        # 对齐精度分析
        alignment_quality = 'excellent' if scores['alignment_score'] >= 0.8 else \
                           'good' if scores['alignment_score'] >= 0.6 else \
                           'fair' if scores['alignment_score'] >= 0.4 else 'poor'

        component_details['alignment_accuracy'] = {
            'quality': alignment_quality,
            'strengths': self._identify_strengths(scores['alignment_score'], 'alignment'),
            'weaknesses': self._identify_weaknesses(scores['alignment_score'], 'alignment')
        }

        # 碰撞惩罚分析
        collision_quality = 'excellent' if scores['collision_penalty'] >= 0.8 else \
                           'good' if scores['collision_penalty'] >= 0.6 else \
                           'fair' if scores['collision_penalty'] >= 0.4 else 'poor'

        component_details['collision_avoidance'] = {
            'quality': collision_quality,
            'strengths': self._identify_strengths(1.0 - scores['collision_penalty'], 'collision'),  # 反向
            'weaknesses': self._identify_weaknesses(1.0 - scores['collision_penalty'], 'collision')
        }

        analysis['component_details'] = component_details

        # 整体平衡性分析
        score_values = list(scores.values())
        analysis['balance_analysis'] = {
            'score_variance': float(np.var(score_values)),
            'score_range': float(np.max(score_values) - np.min(score_values)),
            'most_critical_component': min(scores.items(), key=lambda x: x[1])[0] if score_values else None
        }

        return analysis

    def _identify_strengths(self, score: float, component_type: str) -> list:
        """识别组件优势"""
        strengths = []

        if score >= 0.8:
            strengths.append("表现优秀")
        elif score >= 0.6:
            strengths.append("表现良好")

        if component_type == 'feature':
            if score >= 0.7:
                strengths.append("特征匹配度高")
        elif component_type == 'normal':
            if score >= 0.7:
                strengths.append("法向互补性强")
        elif component_type == 'shape':
            if score >= 0.7:
                strengths.append("形状匹配度好")
        elif component_type == 'alignment':
            if score >= 0.7:
                strengths.append("对齐精度高")
        elif component_type == 'collision':
            if score >= 0.7:
                strengths.append("碰撞风险低")

        return strengths

    def _identify_weaknesses(self, score: float, component_type: str) -> list:
        """识别组件劣势"""
        weaknesses = []

        if score < 0.4:
            weaknesses.append("表现较差")
        elif score < 0.6:
            weaknesses.append("有待改进")

        if component_type == 'feature':
            if score < 0.5:
                weaknesses.append("特征匹配不足")
        elif component_type == 'normal':
            if score < 0.5:
                weaknesses.append("法向互补性弱")
        elif component_type == 'shape':
            if score < 0.5:
                weaknesses.append("形状匹配度低")
        elif component_type == 'alignment':
            if score < 0.5:
                weaknesses.append("对齐精度不足")
        elif component_type == 'collision':
            if score < 0.5:
                weaknesses.append("碰撞风险较高")

        return weaknesses

    def _generate_recommendations(self, validation_scores: ValidationScores) -> list:
        """
        生成优化建议
        """
        recommendations = []
        scores = validation_scores.component_scores

        # 基于总分的建议
        if validation_scores.total_score < 0.5:
            recommendations.append("整体匹配质量较低，建议重新考虑碎片配对")
        elif validation_scores.total_score < 0.7:
            recommendations.append("匹配质量一般，可通过局部调整改善")
        else:
            recommendations.append("匹配质量较好，可进行精细调整")

        # 基于各组件的具体建议
        if scores['feature_score'] < 0.5:
            recommendations.append("特征匹配度不足，建议检查边界特征提取参数")

        if scores['normal_score'] < 0.5:
            recommendations.append("法向互补性较差，建议调整碎片相对姿态")

        if scores['shape_score'] < 0.5:
            recommendations.append("形状匹配度低，可能需要更精确的几何处理")

        if scores['alignment_score'] < 0.5:
            recommendations.append("对齐精度不足，建议使用更高精度的对齐算法")

        if scores['collision_penalty'] < 0.5:  # 注意这里是惩罚项
            recommendations.append("存在碰撞风险，建议调整碎片相对位置")

        # 基于置信度的建议
        if validation_scores.confidence_level < 0.6:
            recommendations.append("验证结果不确定性较高，建议多次验证或人工确认")

        return recommendations

    def _assess_overall_quality(self, validation_scores: ValidationScores) -> Dict[str, Any]:
        """
        评估整体质量
        """
        scores = validation_scores.component_scores
        total_score = validation_scores.total_score

        # 质量等级划分
        if total_score >= 0.8:
            quality_level = 'excellent'
            quality_description = '优秀匹配'
        elif total_score >= 0.65:
            quality_level = 'good'
            quality_description = '良好匹配'
        elif total_score >= 0.5:
            quality_level = 'fair'
            quality_description = '一般匹配'
        else:
            quality_level = 'poor'
            quality_description = '较差匹配'

        # 稳定性评估
        score_variance = np.var(list(scores.values()))
        if score_variance < 0.02:
            stability = 'very_stable'
            stability_desc = '非常稳定'
        elif score_variance < 0.05:
            stability = 'stable'
            stability_desc = '稳定'
        elif score_variance < 0.1:
            stability = 'moderately_stable'
            stability_desc = '中等稳定'
        else:
            stability = 'unstable'
            stability_desc = '不稳定'

        return {
            'quality_level': quality_level,
            'quality_description': quality_description,
            'stability': stability,
            'stability_description': stability_desc,
            'reliability': 'high' if validation_scores.confidence_level > 0.8 else \
                          'medium' if validation_scores.confidence_level > 0.6 else 'low'
        }

    def compare_multiple_matches(self, validation_results: list) -> Dict[str, Any]:
        """
        比较多组匹配结果
        """
        if not validation_results:
            return {}

        # 提取总分
        total_scores = [result.total_score for result in validation_results]

        # 找到最佳匹配
        best_index = np.argmax(total_scores)
        best_result = validation_results[best_index]

        # 统计分析
        comparison = {
            'best_match_index': int(best_index),
            'best_match_score': float(best_result.total_score),
            'score_statistics': {
                'mean': float(np.mean(total_scores)),
                'std': float(np.std(total_scores)),
                'min': float(np.min(total_scores)),
                'max': float(np.max(total_scores))
            },
            'rankings': self._generate_rankings(validation_results),
            'comparison_summary': self._generate_comparison_summary(validation_results)
        }

        return comparison

    def _generate_rankings(self, validation_results: list) -> list:
        """生成排名"""
        # 按总分排序
        indexed_results = [(i, result.total_score) for i, result in enumerate(validation_results)]
        sorted_results = sorted(indexed_results, key=lambda x: x[1], reverse=True)

        rankings = []
        for rank, (index, score) in enumerate(sorted_results, 1):
            rankings.append({
                'rank': rank,
                'match_index': index,
                'total_score': float(score),
                'validation_status': validation_results[index].validation_status
            })

        return rankings

    def _generate_comparison_summary(self, validation_results: list) -> str:
        """生成比较摘要"""
        scores = [result.total_score for result in validation_results]
        statuses = [result.validation_status for result in validation_results]

        excellent_count = statuses.count('excellent_match')
        good_count = statuses.count('good_match')
        acceptable_count = statuses.count('acceptable_match')

        if excellent_count > 0:
            return f"发现{excellent_count}个优秀匹配，{good_count}个良好匹配"
        elif good_count > 0:
            return f"发现{good_count}个良好匹配，{acceptable_count}个可接受匹配"
        else:
            return "所有匹配质量一般，需要进一步优化"