"""
全局拼接模块 - 碎片匹配图与位姿优化
实现完整的陶瓷碎片全局拼接流程
"""

import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import networkx as nx


@dataclass
class MatchEdge:
    """匹配边数据结构"""
    fragment1_id: int
    fragment2_id: int
    weight: float  # 综合评分 S_total 或 Predator overlap/confidence
    transformation: np.ndarray  # 局部对齐后的变换矩阵 R_refined, t_refined
    confidence: float  # 置信度
    dcp_residual: float = 0.0  # DCP 残差
    texture_similarity: float = 0.0  # 纹样相似度（可选）
    
    def __post_init__(self):
        if isinstance(self.transformation, list):
            self.transformation = np.array(self.transformation)


@dataclass 
class FragmentPose:
    """碎片位姿数据结构"""
    fragment_id: int
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    is_fixed: bool = False  # 是否为固定参考碎片
    propagation_path: List[int] = field(default_factory=list)  # 位姿传播路径
    candidate_poses: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)  # 多路径候选位姿
    
    def get_transformation(self) -> np.ndarray:
        """获取 4x4 变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T


class FragmentMatchingGraph:
    """碎片匹配图"""
    
    def __init__(self):
        """初始化匹配图"""
        self.graph = nx.Graph()
        self.edges_dict: Dict[Tuple[int, int], MatchEdge] = {}
        self.poses: Dict[int, FragmentPose] = {}
        self.fragments_data: Dict[int, any] = {}  # 存储碎片原始数据
        
    def add_fragment(self, fragment_id: int, fragment_data: any = None):
        """添加碎片节点"""
        if fragment_id not in self.graph.nodes:
            self.graph.add_node(fragment_id)
            self.fragments_data[fragment_id] = fragment_data
            self.poses[fragment_id] = FragmentPose(fragment_id=fragment_id)
            
    def add_match_edge(self, edge: MatchEdge):
        """添加匹配边"""
        f1, f2 = edge.fragment1_id, edge.fragment2_id
        
        # 确保节点存在
        self.add_fragment(f1)
        self.add_fragment(f2)
        
        # 添加边到图
        if not self.graph.has_edge(f1, f2):
            self.graph.add_edge(f1, f2, weight=edge.weight)
            self.edges_dict[(f1, f2)] = edge
            self.edges_dict[(f2, f1)] = edge  # 无向图
            
    def get_connected_components(self) -> List[Set[int]]:
        """获取所有连通子图"""
        return list(nx.connected_components(self.graph))
    
    def get_minimum_spanning_tree(self) -> nx.Graph:
        """获取最小生成树（用于位姿传播）"""
        # 使用负权重，因为 MST 找最小权重，而我们需要最大权重路径
        modified_graph = nx.Graph()
        for u, v, data in self.graph.edges(data=True):
            modified_graph.add_edge(u, v, weight=-data['weight'])
        
        mst = nx.minimum_spanning_tree(modified_graph)
        # 恢复正权重
        for u, v in mst.edges():
            mst[u][v]['weight'] = -mst[u][v]['weight']
            
        return mst
    
    def get_confidence_threshold_edges(self, threshold: float = 0.5) -> List[MatchEdge]:
        """获取高置信度边"""
        return [edge for edge in self.edges_dict.values() 
                if edge.confidence >= threshold and edge.fragment1_id < edge.fragment2_id]
    
    def build_pose_graph(self) -> 'PoseGraph':
        """构建 Pose Graph 用于优化"""
        pose_graph = PoseGraph()
        
        # 添加位姿节点
        for frag_id, pose in self.poses.items():
            pose_graph.add_pose_node(frag_id, pose.rotation, pose.translation)
        
        # 添加约束边
        for (f1, f2), edge in self.edges_dict.items():
            if f1 < f2:  # 避免重复
                pose_graph.add_constraint(f1, f2, edge.transformation, edge.weight)
                
        return pose_graph


class PoseGraph:
    """Pose Graph 数据结构"""
    
    def __init__(self):
        """初始化 Pose Graph"""
        self.nodes: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # id -> (R, t)
        self.constraints: List[Dict] = []  # 约束列表
        
    def add_pose_node(self, node_id: int, rotation: np.ndarray, translation: np.ndarray):
        """添加位姿节点"""
        self.nodes[node_id] = (rotation.copy(), translation.copy())
        
    def add_constraint(self, node1_id: int, node2_id: int, 
                      relative_transform: np.ndarray, weight: float = 1.0):
        """添加约束边"""
        self.constraints.append({
            'node1': node1_id,
            'node2': node2_id,
            'relative_transform': relative_transform.copy(),
            'weight': weight,
            'information_matrix': np.eye(6) * weight  # 信息矩阵
        })
        
    def get_optimization_problem_size(self) -> Tuple[int, int]:
        """获取优化问题规模"""
        return len(self.nodes), len(self.constraints)


def build_matching_graph_from_validation_results(
    fragments: List[any],
    validation_results: List[Dict]
) -> FragmentMatchingGraph:
    """
    从边界验证结果构建匹配图
    
    Args:
        fragments: 碎片列表
        validation_results: 边界验证结果列表
        
    Returns:
        FragmentMatchingGraph: 构建好的匹配图
    """
    graph = FragmentMatchingGraph()
    
    # 添加所有碎片节点
    for frag in fragments:
        graph.add_fragment(frag.id, frag)
    
    # 从验证结果添加边
    for result in validation_results:
        if not result.get('success', False):
            continue
            
        pair = result['pair']
        f1_id, f2_id = pair[0], pair[1]
        
        # 提取验证得分和变换
        final_scores = result['result']['final_scores']
        total_score = final_scores['total_score']
        
        # 获取局部对齐后的变换
        local_align = result['result']['intermediate_results']['local_alignment']
        transformation = np.eye(4)  # 这里应该从实际的对齐结果中获取
        
        # 计算置信度（可以使用综合得分或其他指标）
        confidence = total_score
        
        # 创建匹配边
        edge = MatchEdge(
            fragment1_id=f1_id,
            fragment2_id=f2_id,
            weight=total_score,
            transformation=transformation,
            confidence=confidence,
            dcp_residual=local_align.get('rmse', 0.0)
        )
        
        graph.add_match_edge(edge)
    
    return graph


def propagate_poses(graph: FragmentMatchingGraph, 
                   reference_fragment_id: Optional[int] = None) -> Dict[int, np.ndarray]:
    """
    位姿传播：从高置信度碎片对开始，沿图传播全局位姿
    
    Args:
        graph: 碎片匹配图
        reference_fragment_id: 参考碎片 ID（默认为度数最大的节点）
        
    Returns:
        Dict[int, np.ndarray]: 4x4 变换矩阵字典
    """
    if len(graph.graph.nodes) == 0:
        return {}
    
    # 选择参考碎片（默认选择连接度最高的节点）
    if reference_fragment_id is None:
        reference_fragment_id = max(graph.graph.nodes(), 
                                   key=lambda n: graph.graph.degree[n])
    
    print(f"[位姿传播] 选择参考碎片：{reference_fragment_id}")
    
    # 设置参考碎片位姿为单位变换
    ref_pose = graph.poses[reference_fragment_id]
    ref_pose.is_fixed = True
    ref_pose.rotation = np.eye(3)
    ref_pose.translation = np.zeros(3)
    
    # 使用 BFS 进行位姿传播
    visited = {reference_fragment_id}
    queue = [reference_fragment_id]
    
    while queue:
        current_id = queue.pop(0)
        current_pose = graph.poses[current_id]
        
        # 获取邻居节点
        neighbors = list(graph.graph.neighbors(current_id))
        
        for neighbor_id in neighbors:
            if neighbor_id in visited:
                continue
                
            # 获取边的变换
            edge = graph.edges_dict.get((current_id, neighbor_id))
            if edge is None:
                edge = graph.edges_dict.get((neighbor_id, current_id))
            
            if edge is None:
                continue
            
            # 应用变换传播位姿
            T_current = np.eye(4)
            T_current[:3, :3] = current_pose.rotation
            T_current[:3, 3] = current_pose.translation
            
            # 确定变换方向
            if edge.fragment1_id == current_id and edge.fragment2_id == neighbor_id:
                T_edge = edge.transformation
            else:
                T_edge = np.linalg.inv(edge.transformation)
            
            # 计算新位姿
            T_neighbor = T_current @ T_edge
            
            # 设置邻居位姿（直接存储 numpy 矩阵而不是 FragmentPose）
            neighbor_pose = graph.poses[neighbor_id]
            neighbor_pose.rotation = T_neighbor[:3, :3]
            neighbor_pose.translation = T_neighbor[:3, 3]
            neighbor_pose.propagation_path = current_pose.propagation_path + [current_id]
            
            visited.add(neighbor_id)
            queue.append(neighbor_id)
            
            print(f"[位姿传播] {current_id} -> {neighbor_id}, "
                  f"路径长度：{len(neighbor_pose.propagation_path)}")
    
    # 转换为 numpy 矩阵字典
    poses_numpy = {}
    for frag_id, frag_pose in graph.poses.items():
        T = np.eye(4)
        T[:3, :3] = frag_pose.rotation
        T[:3, 3] = frag_pose.translation
        poses_numpy[frag_id] = T
    
    return poses_numpy


def visualize_assembly_result(fragments: List[any], 
                             poses: Dict[int, any]) -> o3d.geometry.PointCloud:
    """
    可视化全局拼接结果
    
    Args:
        fragments: 碎片列表
        poses: 碎片位姿字典（可以是 FragmentPose 或 4x4 numpy 矩阵）
        
    Returns:
        o3d.geometry.PointCloud: 拼接后的点云
    """
    combined_pcd = o3d.geometry.PointCloud()
    
    for fragment in fragments:
        frag_id = fragment.id
        if frag_id not in poses:
            continue
            
        pose = poses[frag_id]
        
        # 获取碎片点云
        if hasattr(fragment, 'point_cloud') and fragment.point_cloud is not None:
            pcd = fragment.point_cloud
        elif hasattr(fragment, 'mesh') and fragment.mesh is not None:
            # 从网格采样
            pcd = fragment.mesh.sample_points_uniformly(number_of_points=5000)
        else:
            continue
        
        # 应用全局位姿变换（支持 FragmentPose 和 numpy 矩阵）
        if hasattr(pose, 'get_transformation'):
            # FragmentPose 对象
            T = pose.get_transformation()
        elif isinstance(pose, np.ndarray):
            # numpy 矩阵
            T = pose
        else:
            # 其他类型，尝试转换
            T = np.array(pose) if isinstance(pose, list) else pose
        
        transformed_pcd = pcd.transform(T)
        combined_pcd += transformed_pcd
    
    # 可视化
    o3d.visualization.draw_geometries([combined_pcd])
    
    return combined_pcd
