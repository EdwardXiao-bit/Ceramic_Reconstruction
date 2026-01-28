import os
import open3d as o3d
from src.common.geometry import Fragment  # 导入自定义的Fragment类


def load_fragments(data_dir):
    """
    从指定目录加载陶瓷碎片文件（支持.ply/.pcd/.obj格式）
    :param data_dir: 碎片目录路径
    :return: Fragment对象列表
    """
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"目录不存在：{data_dir}")
        return []

    # 支持的文件格式
    supported_formats = ['.ply', '.pcd', '.obj']
    fragment_files = []

    # 遍历目录下的文件，筛选支持的格式
    for file in os.listdir(data_dir):
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in supported_formats:
            fragment_files.append(os.path.join(data_dir, file))

    # 无有效文件时提示
    if len(fragment_files) == 0:
        print(f"未加载到任何碎片，请检查{data_dir}目录")
        print(f"支持的文件格式：{supported_formats}")
        return []

    # 读取文件并生成Fragment对象
    fragments = []
    for idx, file_path in enumerate(fragment_files):
        frag = Fragment(id=idx)  # 用文件索引作为碎片ID
        file_ext = os.path.splitext(file_path)[1].lower()

        # 读取网格/点云文件
        if file_ext == '.ply' or file_ext == '.obj':
            frag.mesh = o3d.io.read_triangle_mesh(file_path)
            # 网格转点云（方便后续处理）
            frag.point_cloud = frag.mesh.sample_points_uniformly(number_of_points=10000)
        elif file_ext == '.pcd':
            frag.point_cloud = o3d.io.read_point_cloud(file_path)

        # 验证文件是否读取成功
        if frag.point_cloud is None or len(frag.point_cloud.points) == 0:
            print(f"文件读取失败：{file_path}")
            continue

        fragments.append(frag)

    print(f"成功加载 {len(fragments)} 个碎片")
    return fragments