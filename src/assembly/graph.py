import open3d as o3d
import numpy as np


def assemble(fragments, matches):
    """
    装配碎片：合并匹配碎片的点云（MVP级仅可视化）
    :param fragments: list[Fragment] 碎片实例列表
    :param matches: list[tuple] 匹配对 (frag1_id, frag2_id, 相似度)
    :return: o3d.geometry.PointCloud 装配后的点云
    """
    if not matches or len(fragments) < 2:
        print("无有效匹配对，返回第一个碎片的点云")
        if fragments:
            return fragments[0].point_cloud
        else:
            # 创建一个空点云作为fallback
            empty_pcd = o3d.geometry.PointCloud()
            return empty_pcd

    # 取相似度最高的匹配对进行合并
    best_match = matches[0]
    f1_id, f2_id, _ = best_match

    # 找到对应的碎片
    f1 = next(f for f in fragments if f.id == f1_id)
    f2 = next(f for f in fragments if f.id == f2_id)

    # 合并点云（MVP级：简单拼接，未做位姿对齐）
    combined_pcd = o3d.geometry.PointCloud()
    pts1 = np.asarray(f1.point_cloud.points)
    pts2 = np.asarray(f2.point_cloud.points)
    # 简单平移第二个碎片，避免重叠（仅可视化）
    pts2 += np.array([0.5, 0, 0])
    combined_pts = np.vstack([pts1, pts2])

    combined_pcd.points = o3d.utility.Vector3dVector(combined_pts)
    # 可视化装配结果
    o3d.visualization.draw_geometries([combined_pcd])

    return combined_pcd