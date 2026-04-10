"""
Pose Graph 优化模块
使用 g2o 或 Ceres Solver 进行全局位姿优化
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys


@dataclass
class PoseGraphNode:
    """Pose Graph 节点"""
    id: int
    rotation: np.ndarray  # 3x3 旋转矩阵
    translation: np.ndarray  # 3x1 平移向量
    is_fixed: bool = False  # 是否为固定参考节点
    
    def get_transformation(self) -> np.ndarray:
        """获取齐次变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T


@dataclass
class PoseGraphConstraint:
    """Pose Graph 约束"""
    node1_id: int
    node2_id: int
    relative_transform: np.ndarray  # 4x4 相对变换
    information_matrix: np.ndarray  # 6x6 信息矩阵（权重）


class PoseGraphOptimizer:
    """Pose Graph 优化器"""
    
    def __init__(self, use_g2o: bool = False, use_ceres: bool = False):
        """
        初始化优化器
        
        Args:
            use_g2o: 是否使用 g2o（如果可用）
            use_ceres: 是否使用 Ceres（如果可用）
        """
        self.use_g2o = use_g2o and self._check_g2o_available()
        self.use_ceres = use_ceres and self._check_ceres_available()
        
        if not self.use_g2o and not self.use_ceres:
            print("[Pose Graph 优化] 使用自定义简化优化器")
            
    def _check_g2o_available(self) -> bool:
        """检查 g2o 是否可用"""
        try:
            import g2o
            return True
        except ImportError:
            return False
            
    def _check_ceres_available(self) -> bool:
        """检查 Ceres 是否可用"""
        try:
            import ceres
            return True
        except ImportError:
            return False
    
    def optimize(self, nodes: Dict[int, PoseGraphNode], 
                constraints: List[PoseGraphConstraint],
                max_iterations: int = 100) -> Dict[int, PoseGraphNode]:
        """
        执行 Pose Graph 优化
        
        Args:
            nodes: 位姿节点字典
            constraints: 约束列表
            max_iterations: 最大迭代次数
            
        Returns:
            Dict[int, PoseGraphNode]: 优化后的位姿
        """
        if self.use_g2o:
            return self._optimize_with_g2o(nodes, constraints, max_iterations)
        elif self.use_ceres:
            return self._optimize_with_ceres(nodes, constraints, max_iterations)
        else:
            return self._optimize_simple(nodes, constraints, max_iterations)
    
    def _optimize_with_g2o(self, nodes: Dict[int, PoseGraphNode],
                          constraints: List[PoseGraphConstraint],
                          max_iterations: int) -> Dict[int, PoseGraphNode]:
        """使用 g2o 进行优化"""
        try:
            import g2o
            
            # 创建优化器
            optimizer = g2o.SparseOptimizer()
            solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
            solver = g2o.OptimizationAlgorithmLevenberg(solver)
            optimizer.set_algorithm(solver)
            
            # 添加节点
            for node_id, node in nodes.items():
                v_se3 = g2o.VertexSE3()
                v_se3.setId(node_id)
                v_se3.setEstimate(node.get_transformation())
                optimizer.add_vertex(v_se3)
                
            # 添加边
            for constraint in constraints:
                edge = g2o.EdgeSE3()
                edge.set_vertex(0, optimizer.vertex(constraint.node1_id))
                edge.set_vertex(1, optimizer.vertex(constraint.node2_id))
                edge.set_measurement(constraint.relative_transform)
                edge.set_information(constraint.information_matrix)
                optimizer.add_edge(edge)
            
            # 固定第一个节点作为参考
            if nodes:
                first_node = optimizer.vertex(min(nodes.keys()))
                first_node.set_fixed(True)
            
            # 优化
            optimizer.initialize_optimization()
            optimizer.optimize(max_iterations)
            
            # 提取优化结果
            optimized_poses = {}
            for node_id, node in nodes.items():
                v_se3 = optimizer.vertex(node_id)
                if v_se3 is not None:
                    T = v_se3.estimate()
                    optimized_poses[node_id] = PoseGraphNode(
                        id=node_id,
                        rotation=T[:3, :3],
                        translation=T[:3, 3]
                    )
            
            print(f"[g2o 优化] 完成，迭代次数：{optimizer.current_iteration()}")
            return optimized_poses
            
        except Exception as e:
            print(f"[g2o 优化] 失败：{e}, 回退到简化优化器")
            return self._optimize_simple(nodes, constraints, max_iterations)
    
    def _optimize_with_ceres(self, nodes: Dict[int, PoseGraphNode],
                            constraints: List[PoseGraphConstraint],
                            max_iterations: int) -> Dict[int, PoseGraphNode]:
        """使用 Ceres 进行优化"""
        try:
            import ceres
            
            # 构建问题
            problem = ceres.Problem()
            
            # 为每个节点添加参数块
            pose_params = {}
            for node_id, node in nodes.items():
                # 使用旋转向量 + 平移向量表示位姿（7 维）
                params = np.zeros(7)
                params[:3] = node.translation
                params[3:] = self._rotation_to_vector(node.rotation)
                pose_params[node_id] = params
                
                # 如果是第一个节点，设为常量
                if node_id == min(nodes.keys()):
                    problem.AddParameterBlock(params.ctypes.data, 7)
                    problem.SetParameterBlockConstant(params.ctypes.data)
            
            # 添加残差块
            for constraint in constraints:
                param1 = pose_params[constraint.node1_id]
                param2 = pose_params[constraint.node2_id]
                
                # 创建代价函数
                cost_function = self._create_pose_graph_cost(
                    constraint.relative_transform,
                    constraint.information_matrix
                )
                
                problem.AddResidualBlock(cost_function, None,
                                       param1.ctypes.data,
                                       param2.ctypes.data)
            
            # 设置求解选项
            options = ceres.Solver.Options()
            options.linear_solver_type = ceres.SPARSE_SCHUR
            options.minimizer_progress_to_stdout = False
            options.max_num_iterations = max_iterations
            
            # 求解
            summary = ceres.Solver.Summary()
            ceres.Solve(options, problem, summary)
            
            print(f"[Ceres 优化] 完成，初始代价：{summary.initial_cost}, "
                  f"最终代价：{summary.final_cost}")
            
            # 提取优化结果
            optimized_poses = {}
            for node_id, params in pose_params.items():
                translation = params[:3]
                rotation = self._vector_to_rotation(params[3:])
                optimized_poses[node_id] = PoseGraphNode(
                    id=node_id,
                    rotation=rotation,
                    translation=translation
                )
            
            return optimized_poses
            
        except Exception as e:
            print(f"[Ceres 优化] 失败：{e}, 回退到简化优化器")
            return self._optimize_simple(nodes, constraints, max_iterations)
    
    def _optimize_simple(self, nodes: Dict[int, PoseGraphNode],
                        constraints: List[PoseGraphConstraint],
                        max_iterations: int) -> Dict[int, PoseGraphNode]:
        """
        简化的优化器（基于梯度下降的近似）
        当 g2o 和 Ceres 都不可用时使用
        """
        print(f"[简化优化器] 开始优化，{len(nodes)}个节点，{len(constraints)}个约束")
        
        # 复制初始位姿
        optimized_poses = {}
        for node_id, node in nodes.items():
            optimized_poses[node_id] = PoseGraphNode(
                id=node_id,
                rotation=node.rotation.copy(),
                translation=node.translation.copy()
            )
        
        # 固定第一个节点
        if nodes:
            first_id = min(nodes.keys())
            optimized_poses[first_id].is_fixed = True
        
        # 简单的迭代优化
        learning_rate = 0.1
        
        for iteration in range(max_iterations):
            total_error = 0.0
            
            # 对每个约束计算误差并更新
            for constraint in constraints:
                pose1 = optimized_poses[constraint.node1_id]
                pose2 = optimized_poses[constraint.node2_id]
                
                # 计算当前相对变换
                T1 = pose1.get_transformation()
                T2 = pose2.get_transformation()
                current_relative = np.linalg.inv(T1) @ T2
                
                # 计算误差
                error_transform = current_relative @ np.linalg.inv(constraint.relative_transform)
                error_angle = np.arccos(np.clip((np.trace(error_transform[:3, :3]) - 1) / 2, -1, 1))
                error_translation = np.linalg.norm(error_transform[:3, 3])
                
                total_error += error_angle + error_translation
                
                # 简单的位置更新（仅平移部分）
                if not pose1.is_fixed and not pose2.is_fixed:
                    delta_t = error_transform[:3, 3] * learning_rate * constraint.information_matrix[0, 0]
                    pose2.translation -= delta_t * 0.5
                    pose1.translation += delta_t * 0.5
                elif not pose2.is_fixed:
                    delta_t = error_transform[:3, 3] * learning_rate
                    pose2.translation -= delta_t
                elif not pose1.is_fixed:
                    delta_t = error_transform[:3, 3] * learning_rate
                    pose1.translation += delta_t
            
            if iteration % 10 == 0:
                print(f"[简化优化器] 迭代 {iteration}/{max_iterations}, "
                      f"总误差：{total_error:.6f}")
            
            # 收敛检查
            if total_error < 1e-6:
                print(f"[简化优化器] 在迭代 {iteration} 收敛")
                break
        
        return optimized_poses
    
    def _rotation_to_vector(self, R: np.ndarray) -> np.ndarray:
        """旋转矩阵转旋转向量"""
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle < 1e-6:
            return np.zeros(3)
        
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(angle))
        
        return axis * angle
    
    def _vector_to_rotation(self, r_vec: np.ndarray) -> np.ndarray:
        """旋转向量转旋转矩阵"""
        angle = np.linalg.norm(r_vec)
        if angle < 1e-6:
            return np.eye(3)
        
        axis = r_vec / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    def _create_pose_graph_cost(self, relative_transform: np.ndarray,
                               information_matrix: np.ndarray):
        """创建 Pose Graph 代价函数（用于 Ceres）"""
        # 这里需要实现 Ceres 的自动微分代价函数
        # 由于复杂度较高，简化版本中暂不实现
        pass


def optimize_global_poses(graph: 'FragmentMatchingGraph',
                         max_iterations: int = 100,
                         use_g2o: bool = False,
                         use_ceres: bool = False) -> Dict[int, any]:
    """
    优化全局位姿
    
    Args:
        graph: 碎片匹配图
        max_iterations: 最大迭代次数
        use_g2o: 是否使用 g2o
        use_ceres: 是否使用 Ceres
        
    Returns:
        Dict[int, FragmentPose]: 优化后的位姿
    """
    from .global_assembly import FragmentPose, FragmentMatchingGraph
    
    # 构建 Pose Graph
    pose_graph = graph.build_pose_graph()
    
    # 转换为优化器格式
    nodes = {}
    for node_id, (R, t) in pose_graph.nodes.items():
        nodes[node_id] = PoseGraphNode(id=node_id, rotation=R, translation=t)
    
    constraints = []
    for constraint_dict in pose_graph.constraints:
        constraints.append(PoseGraphConstraint(
            node1_id=constraint_dict['node1'],
            node2_id=constraint_dict['node2'],
            relative_transform=constraint_dict['relative_transform'],
            information_matrix=constraint_dict['information_matrix']
        ))
    
    # 执行优化
    optimizer = PoseGraphOptimizer(use_g2o=use_g2o, use_ceres=use_ceres)
    optimized_nodes = optimizer.optimize(nodes, constraints, max_iterations)
    
    # 转换回 FragmentPose 格式
    optimized_poses = {}
    for node_id, opt_node in optimized_nodes.items():
        frag_pose = graph.poses[node_id]
        # 直接返回 numpy 矩阵而不是 FragmentPose
        T = np.eye(4)
        T[:3, :3] = opt_node.rotation
        T[:3, 3] = opt_node.translation
        optimized_poses[node_id] = T
    
    return optimized_poses
