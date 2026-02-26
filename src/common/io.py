# D:\ceramic_reconstruction\src\common\io.py
import os
import numpy as np
import open3d as o3d

# 从同目录base.py导入Fragment类（路径正确，无导入错误）
from src.common.base import Fragment

# 支持的碎片文件格式（小写，兼容文件扩展名大小写）
SUPPORTED_FORMATS = ['.ply', '.pcd', '.obj']


def load_fragments(data_dir, num_points=10000):
    """
    加载碎片文件，自动区分「三角网格文件」和「纯点云文件」，彻底解决无三角面报错
    适配：带三角面的ply/obj、纯点云ply/pcd，混合存放也能正常加载
    :param data_dir: 碎片文件目录（如data/demo）
    :param num_points: 网格采样点云的目标点数，默认10000
    :return: 有效Fragment对象列表 [Fragment, ...]
    """
    fragments = []
    # 容错：目录不存在直接返回空
    if not os.path.exists(data_dir):
        print(f"[文件加载] 错误：目录 {data_dir} 不存在，返回空碎片列表")
        return fragments

    # 筛选目录下所有支持格式的文件
    file_list = [
        f for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_FORMATS
    ]
    if len(file_list) == 0:
        print(f"[文件加载] 提示：目录 {data_dir} 下无支持的文件（仅支持{SUPPORTED_FORMATS}），返回空碎片列表")
        return fragments

    # 遍历加载每个文件
    for idx, file_name in enumerate(file_list):
        file_path = os.path.join(data_dir, file_name)
        file_ext = os.path.splitext(file_name)[1].lower()
        # 初始化Fragment对象
        frag = Fragment(id=idx, file_path=file_path, file_name=file_name)
        print(f"\n[文件加载] 开始加载碎片{idx}：{file_name}")

        try:
            # 分支1：纯点云文件（.pcd 或 无三角面的.ply）→ 直接加载点云，不采样
            if file_ext == '.pcd' or (file_ext == '.ply' and not is_mesh_with_triangles(file_path)):
                frag.point_cloud = o3d.io.read_point_cloud(file_path)
                # 验证点云有效性
                if frag.point_cloud is None or len(frag.point_cloud.points) == 0:
                    print(f"[文件加载] 跳过碎片{idx}：{file_name} 是无效纯点云（无点数据）")
                    continue
                print(f"[文件加载] 碎片{idx}：识别为纯点云文件，直接加载 {len(frag.point_cloud.points)} 个原始点")

            # 分支2：三角网格文件（带三角面的.ply / .obj）→ 加载网格并采样点云
            else:
                frag.mesh = o3d.io.read_triangle_mesh(file_path)
                # 容错1：网格无顶点直接跳过
                if frag.mesh is None or len(frag.mesh.vertices) == 0:
                    print(f"[文件加载] 跳过碎片{idx}：{file_name} 是无效网格（无顶点数据）")
                    continue
                # 容错2：网格无三角面 → 顶点直接转点云，不调用采样方法（解决核心报错）
                if len(frag.mesh.triangles) == 0:
                    frag.point_cloud = o3d.geometry.PointCloud()
                    frag.point_cloud.points = frag.mesh.vertices
                    print(f"[文件加载] 碎片{idx}：网格无三角面，将顶点转为 {len(frag.mesh.vertices)} 个点云点")
                # 正常网格 → 均匀采样生成点云
                else:
                    frag.point_cloud = frag.mesh.sample_points_uniformly(number_of_points=num_points)
                    print(f"[文件加载] 碎片{idx}：识别为三角网格文件，采样生成 {num_points} 个点云点")

            # 点云去重（可选，避免重复点影响后续几何提取，voxel_size可根据模型大小微调）
            frag.point_cloud = frag.point_cloud.voxel_down_sample(voxel_size=0.005)
            # 添加到有效碎片列表
            fragments.append(frag)
            print(f"[文件加载] 碎片{idx}：{file_name} 加载完成")

        except Exception as e:
            # 单个文件加载失败不中断整体流程，仅打印错误并跳过
            print(f"[文件加载] 跳过碎片{idx}：{file_name} 加载失败，错误信息：{str(e)[:100]}")
            continue

    print(f"\n[文件加载] 加载完成！共加载 {len(fragments)} 个有效碎片（总文件数：{len(file_list)}）")
    return fragments


def is_mesh_with_triangles(file_path):
    """
    预检查.ply文件是否包含三角面，用于区分「纯点云ply」和「网格ply」
    :param file_path: ply文件路径
    :return: True（带三角面，是网格）/ False（无三角面，是纯点云）
    """
    try:
        # 临时加载网格，仅检查三角面数量，不做其他处理
        temp_mesh = o3d.io.read_triangle_mesh(file_path)
        return temp_mesh is not None and len(temp_mesh.triangles) > 0
    except Exception:
        # 加载失败直接判定为纯点云
        return False


# 测试代码（单独运行io.py时验证加载功能，无需修改）
if __name__ == "__main__":
    # 直接运行该文件，测试data/demo目录的加载效果
    test_fragments = load_fragments("data/demo")
    print(f"\n【测试结果】共加载 {len(test_fragments)} 个有效碎片")
    for f in test_fragments:
        print(f"→ 碎片{f.id}：{f.file_name}，点云点数：{len(f.point_cloud.points)}")