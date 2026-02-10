# D:\ceramic_reconstruction\src\common\base.py
class Fragment:
    """
    陶瓷碎片核心对象类，承载碎片的所有属性（文件信息、几何数据、特征向量）
    作为整个pipeline的核心数据载体，所有模块均基于该类传递数据
    """

    def __init__(self, id, file_path, file_name):
        # 基础文件信息
        self.id = id  # 碎片唯一标识ID（int，从0开始）
        self.file_path = file_path  # 碎片文件绝对路径（str）
        self.file_name = file_name  # 碎片文件名（str）

        # 几何基础数据
        self.mesh = None  # 三角网格对象（open3d.geometry.TriangleMesh | None）
        self.point_cloud = None  # 点云对象（open3d.geometry.PointCloud | None）

        # 归一化相关信息
        self.normalized = False  # 是否已归一化（bool）
        self.scale = 1.0  # 归一化缩放因子（float）
        self.centroid = None  # 归一化前的质心（np.ndarray | None）
        self.original_points = None  # 原始点云坐标（np.ndarray | None）

        # 边界检测相关属性
        self.boundary_pts = None  # 边界点坐标数组 (N,3)（np.ndarray | None）
        self.boundary_pcd = None  # 边界点云对象（open3d.geometry.PointCloud | None）

        # 断面提取相关属性
        self.section_patch = None  # 断面patch点云对象（open3d.geometry.PointCloud | None）

        # Rim曲线提取相关属性
        self.main_axis = None  # 器型主轴线（PCA提取，np.ndarray | None）
        self.rim_curve = None  # Rim曲线坐标数组 (M,3)（np.ndarray | None）
        self.rim_pcd = None  # Rim曲线点云对象（open3d.geometry.PointCloud | None）
        self.rim_lines = None  # Rim曲线闭合线集（open3d.geometry.LineSet | None）

        # 轮廓与特征编码相关属性
        self.profile_curve = None  # 整体轮廓曲线（np.ndarray | None）
        self.geo_embedding = None  # 128维几何特征嵌入向量（np.ndarray | None）
        self.profile_feature = None  # 轮廓特征向量（np.ndarray | None）