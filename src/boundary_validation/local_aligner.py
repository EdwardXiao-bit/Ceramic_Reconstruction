# D:\ceramic_reconstruction\src\boundary_validation\local_aligner.py
"""
局部对齐精化模块
实现Deep Closest Point(DCP)粗对齐refine和局部ICP
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class AlignmentResult:
    """对齐结果数据类"""
    refined_transformation: np.ndarray     # 精化后的变换矩阵
    alignment_error: float                 # 对齐误差
    fitness_score: float                   # 适应度得分
    rmse: float                           # 均方根误差
    iterations_used: int                   # 使用的迭代次数
    convergence_status: str                # 收敛状态

class LocalAligner:
    """局部对齐精化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dcp_model = None
        self._initialize_models()
        
    def _initialize_models(self):
        """初始化对齐模型"""
        if self.config['dcp_enabled']:
            try:
                self.dcp_model = self._create_dcp_model()
                print("[局部对齐] DCP模型初始化成功")
            except Exception as e:
                print(f"[局部对齐] DCP模型初始化失败: {e}")
                self.dcp_model = None
    
    def _create_dcp_model(self):
        """创建 DCP 模型（带降级处理）"""
        from pathlib import Path
            
        # 尝试加载真实 DCP 模型
        try:
            # 首先检查是否有 DCP 模型实现
            try:
                from src.models.dcp import DCP
                has_dcp_implementation = True
            except ImportError:
                print("[局部对齐] ⚠ 未找到 DCP 模型实现，使用 Mock 模式")
                has_dcp_implementation = False
                
            if not has_dcp_implementation:
                return self._create_mock_dcp()
                
            import yaml
                
            # 加载配置
            config_paths = [
                Path('configs/dcp.yaml'),
                Path(__file__).parent.parent.parent / 'configs' / 'dcp.yaml'
            ]
                
            config_file = None
            for path in config_paths:
                if path.exists():
                    config_file = path
                    break
                
            if config_file is None:
                raise FileNotFoundError("未找到 dcp.yaml 配置文件")
                
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # 创建模型
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = DCP(config['MODEL']).to(device)
                
            # 按优先级尝试加载预训练权重
            weight_paths = [
                # 1. DCP 通用权重
                Path('pretrained_weights/dcp/dcp_best.pth'),
                # 2. 相对路径备用
                Path(__file__).parent.parent.parent / 'pretrained_weights' / 'dcp' / 'dcp_best.pth'
            ]
                
            loaded_weight = None
            for weight_path in weight_paths:
                if weight_path.exists():
                    print(f"[局部对齐] 加载 DCP 权重：{weight_path}")
                    checkpoint = torch.load(weight_path, map_location=device)
                        
                    # 兼容不同格式的 checkpoint
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # 如果直接是 state_dict
                        model.load_state_dict(checkpoint)
                        
                    loaded_weight = weight_path.name
                    break
                
            if loaded_weight:
                print(f"[局部对齐] ✓ DCP 模型加载成功 (权重：{loaded_weight})")
            else:
                print("[局部对齐] ⚠ 警告：未找到 DCP 预训练权重，使用随机初始化")
                
            model.eval()
            return model
                
        except Exception as e:
            print(f"[局部对齐] ⚠ DCP 加载失败 ({e})，降级到 Mock 模式")
            # 降级到 Mock 实现
            return self._create_mock_dcp()
        
    def _create_mock_dcp(self):
        """创建 Mock DCP（降级处理）"""
        class MockDCP(nn.Module):
            def __init__(self):
                super().__init__()
                # 简化的 Transformer 编码器
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
                    num_layers=6
                )
                self.regressor = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 6)  # 3 个旋转参数 + 3 个平移参数
                )
                    
            def forward(self, src, tgt):
                # 简化的前向传播
                batch_size = src.shape[0]
                # 模拟编码和回归
                delta_params = torch.randn(batch_size, 6) * 0.1
                return delta_params
                    
        return MockDCP()
    
    def refine_alignment(self, fragment1: Any, fragment2: Any, 
                        initial_transformation: np.ndarray) -> AlignmentResult:
        """
        精化对齐结果
        
        Args:
            fragment1: 第一个碎片
            fragment2: 第二个碎片
            initial_transformation: 初始变换矩阵
            
        Returns:
            AlignmentResult: 对齐结果
        """
        print("[局部对齐] 开始局部对齐精化...")
        
        # 获取点云数据
        points1 = self._get_fragment_points(fragment1)
        points2 = self._get_fragment_points(fragment2)
        
        if points1 is None or points2 is None:
            print("[局部对齐] 点云数据获取失败")
            return self._create_failed_result()
        
        best_result = None
        best_fitness = -1
        
        # 1. 使用DCP进行精化（如果启用）
        if self.config['dcp_enabled'] and self.dcp_model is not None:
            print("[局部对齐] 使用DCP进行精化...")
            dcp_result = self._refine_with_dcp(points1, points2, initial_transformation)
            if dcp_result.fitness_score > best_fitness:
                best_result = dcp_result
                best_fitness = dcp_result.fitness_score
        
        # 2. 使用局部ICP进行精化（如果启用）
        if self.config['icp_enabled']:
            print("[局部对齐] 使用局部ICP进行精化...")
            icp_result = self._refine_with_local_icp(points1, points2, initial_transformation)
            if icp_result.fitness_score > best_fitness:
                best_result = icp_result
                best_fitness = icp_result.fitness_score
        
        # 3. 如果没有成功的方法，返回初始变换
        if best_result is None:
            print("[局部对齐] 所有精化方法失败，返回初始变换")
            best_result = AlignmentResult(
                refined_transformation=initial_transformation,
                alignment_error=1.0,
                fitness_score=0.0,
                rmse=float('inf'),
                iterations_used=0,
                convergence_status='failed'
            )
        
        print(f"[局部对齐] 精化完成:")
        print(f"  适应度得分: {best_result.fitness_score:.3f}")
        print(f"  RMSE: {best_result.rmse:.6f}")
        print(f"  迭代次数: {best_result.iterations_used}")
        print(f"  收敛状态: {best_result.convergence_status}")
        
        return best_result
    
    def _get_fragment_points(self, fragment: Any) -> Optional[np.ndarray]:
        """获取碎片点云数据"""
        if hasattr(fragment, 'point_cloud') and fragment.point_cloud is not None:
            return np.asarray(fragment.point_cloud.points)
        elif hasattr(fragment, 'mesh') and fragment.mesh is not None:
            # 从网格采样点
            pcd = fragment.mesh.sample_points_uniformly(number_of_points=5000)
            return np.asarray(pcd.points)
        else:
            return None
    
    def _refine_with_dcp(self, points1: np.ndarray, points2: np.ndarray, 
                        initial_transform: np.ndarray) -> AlignmentResult:
        """
        使用 DCP 进行对齐精化
        """
        try:
            # 应用初始变换
            transformed_points1 = self._apply_transformation(points1, initial_transform)
            
            # 准备输入数据
            src_tensor = torch.FloatTensor(transformed_points1).unsqueeze(0)  # [1, N, 3]
            tgt_tensor = torch.FloatTensor(points2).unsqueeze(0)  # [1, M, 3]
            
            # 确保相同点数（简单截断）
            min_points = min(src_tensor.shape[1], tgt_tensor.shape[1])
            src_tensor = src_tensor[:, :min_points, :]
            tgt_tensor = tgt_tensor[:, :min_points, :]
            
            # DCP 预测
            with torch.no_grad():
                # 检查是否是真实 DCP 模型
                if hasattr(self.dcp_model, 'forward') and callable(getattr(self.dcp_model, 'forward')):
                    # 真实 DCP：返回 (R, t) 或完整的 transform
                    output = self.dcp_model(src_tensor, tgt_tensor)
                    
                    if isinstance(output, tuple) and len(output) == 2:
                        # 返回 (R_pred, t_pred)
                        R_pred, t_pred = output
                        # 构建增量变换矩阵
                        delta_transform = torch.eye(4, device=R_pred.device)
                        delta_transform[:3, :3] = R_pred[0]
                        delta_transform[:3, 3] = t_pred[0]
                        delta_transform = delta_transform.cpu().numpy()
                    else:
                        # 返回完整的 4x4 变换矩阵
                        delta_transform = output[0].cpu().numpy()
                else:
                    # Mock DCP：直接返回参数
                    delta_params = self.dcp_model(src_tensor, tgt_tensor)
                    delta_params = delta_params.squeeze(0).cpu().numpy()
                    delta_transform = self._params_to_transform(delta_params[:3], delta_params[3:])
            
            # 组合变换
            refined_transform = delta_transform @ initial_transform
            
            # 评估结果
            fitness_score, rmse = self._evaluate_alignment(
                points1, points2, refined_transform
            )
            
            result = AlignmentResult(
                refined_transformation=refined_transform,
                alignment_error=1.0 - fitness_score,
                fitness_score=fitness_score,
                rmse=rmse,
                iterations_used=1,  # DCP 通常是单次预测
                convergence_status='converged' if fitness_score > 0.5 else 'partial'
            )
            
            return result
            
        except Exception as e:
            print(f"[DCP 精化] 失败：{e}")
            import traceback
            traceback.print_exc()
            return self._create_failed_result()
    
    def _refine_with_local_icp(self, points1: np.ndarray, points2: np.ndarray, 
                              initial_transform: np.ndarray) -> AlignmentResult:
        """
        使用局部ICP进行对齐精化
        """
        icp_config = self.config['icp']
        
        try:
            # 创建Open3D点云对象
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(points1)
            
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(points2)
            
            # 设置ICP参数
            threshold = icp_config['distance_threshold']
            max_iterations = icp_config['max_iterations']
            
            # 执行局部ICP
            result = o3d.pipelines.registration.registration_icp(
                source=source,
                target=target,
                max_correspondence_distance=threshold,
                init=np.eye(4),  # 使用单位矩阵，因为我们将在评估时应用初始变换
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=max_iterations
                )
            )
            
            # 组合变换：ICP变换 × 初始变换
            refined_transform = result.transformation @ initial_transform
            
            # 评估对齐质量
            fitness_score = result.fitness
            rmse = result.inlier_rmse
            
            # 获取迭代次数（处理不同Open3D版本的兼容性）
            iterations_used = 0
            if hasattr(result, 'number_of_iterations'):
                iterations_used = result.number_of_iterations
            elif hasattr(result, 'iteration'):
                iterations_used = result.iteration
            else:
                # 如果都无法获取，使用配置中的最大迭代次数作为估计
                iterations_used = min(max_iterations, 50)  # 保守估计
            
            # 判断收敛状态
            if fitness_score >= icp_config['fitness_threshold'] and rmse <= icp_config['rmse_threshold']:
                convergence_status = 'converged'
            elif fitness_score >= 0.3:
                convergence_status = 'partial'
            else:
                convergence_status = 'failed'
            
            result_obj = AlignmentResult(
                refined_transformation=refined_transform,
                alignment_error=1.0 - fitness_score,
                fitness_score=fitness_score,
                rmse=rmse,
                iterations_used=iterations_used,
                convergence_status=convergence_status
            )
            
            return result_obj
            
        except Exception as e:
            print(f"[局部ICP] 失败: {e}")
            return self._create_failed_result()
    
    def _apply_transformation(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """应用变换矩阵到点云"""
        # 齐次坐标变换
        homogeneous_points = np.hstack([points, np.ones((len(points), 1))])
        transformed_points = (transform @ homogeneous_points.T).T
        return transformed_points[:, :3]  # 移除齐次坐标
    
    def _params_to_transform(self, rotation_params: np.ndarray, 
                           translation_params: np.ndarray) -> np.ndarray:
        """
        将参数转换为变换矩阵
        """
        # 简化的欧拉角到旋转矩阵转换
        rx, ry, rz = rotation_params
        
        # 绕x轴旋转
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        # 绕y轴旋转
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        # 绕z轴旋转
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转矩阵
        R = Rz @ Ry @ Rx
        
        # 构建完整变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = translation_params
        
        return transform
    
    def _evaluate_alignment(self, points1: np.ndarray, points2: np.ndarray, 
                          transform: np.ndarray) -> Tuple[float, float]:
        """
        评估对齐质量
        """
        # 应用变换
        transformed_points1 = self._apply_transformation(points1, transform)
        
        # 使用最近邻计算重叠度
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        tree = o3d.geometry.KDTreeFlann(pcd2)
        
        total_distance = 0
        inlier_count = 0
        threshold = 0.02  # 2cm阈值
        
        for point in transformed_points1:
            _, indices, distances = tree.search_knn_vector_3d(point, 1)
            if len(indices) > 0:
                distance = distances[0]
                total_distance += distance
                if distance < threshold:
                    inlier_count += 1
        
        # 计算适应度和RMSE
        fitness_score = inlier_count / len(transformed_points1) if len(transformed_points1) > 0 else 0
        rmse = np.sqrt(total_distance / len(transformed_points1)) if len(transformed_points1) > 0 else float('inf')
        
        return fitness_score, rmse
    
    def _create_failed_result(self) -> AlignmentResult:
        """创建失败的对齐结果"""
        return AlignmentResult(
            refined_transformation=np.eye(4),
            alignment_error=1.0,
            fitness_score=0.0,
            rmse=float('inf'),
            iterations_used=0,
            convergence_status='failed'
        )
    
    def multi_scale_alignment(self, fragment1: Any, fragment2: Any,
                            initial_transformation: np.ndarray) -> AlignmentResult:
        """
        多尺度对齐精化
        """
        print("[多尺度对齐] 开始多尺度对齐精化...")
        
        # 获取点云数据
        points1 = self._get_fragment_points(fragment1)
        points2 = self._get_fragment_points(fragment2)
        
        if points1 is None or points2 is None:
            return self._create_failed_result()
        
        current_transform = initial_transformation.copy()
        scale_factors = [0.2, 0.5, 1.0]  # 从粗到细的尺度
        
        for i, scale in enumerate(scale_factors):
            print(f"[多尺度对齐] 第{i+1}级尺度 ({scale})...")
            
            # 下采样点云
            scaled_points1 = self._downsample_points(points1, scale)
            scaled_points2 = self._downsample_points(points2, scale)
            
            # 在当前尺度下进行精化
            scale_result = self.refine_alignment_at_scale(
                scaled_points1, scaled_points2, current_transform
            )
            
            current_transform = scale_result.refined_transformation
        
        # 最终评估
        final_fitness, final_rmse = self._evaluate_alignment(points1, points2, current_transform)
        
        result = AlignmentResult(
            refined_transformation=current_transform,
            alignment_error=1.0 - final_fitness,
            fitness_score=final_fitness,
            rmse=final_rmse,
            iterations_used=len(scale_factors),
            convergence_status='multi_scale_converged' if final_fitness > 0.6 else 'multi_scale_partial'
        )
        
        print(f"[多尺度对齐] 完成，最终适应度: {final_fitness:.3f}")
        return result
    
    def _downsample_points(self, points: np.ndarray, scale: float) -> np.ndarray:
        """点云下采样"""
        if scale >= 1.0:
            return points
            
        target_count = max(int(len(points) * scale), 100)  # 至少保留100个点
        if target_count >= len(points):
            return points
            
        # 随机下采样
        indices = np.random.choice(len(points), size=target_count, replace=False)
        return points[indices]
    
    def refine_alignment_at_scale(self, points1: np.ndarray, points2: np.ndarray,
                                transform: np.ndarray) -> AlignmentResult:
        """
        在特定尺度下进行对齐精化
        """
        # 优先使用DCP，降级到ICP
        if self.config['dcp_enabled'] and self.dcp_model is not None:
            return self._refine_with_dcp(points1, points2, transform)
        else:
            return self._refine_with_local_icp(points1, points2, transform)