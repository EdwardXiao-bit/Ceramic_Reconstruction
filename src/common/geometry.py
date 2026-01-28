class Fragment:
    def __init__(self, id):
        self.id = id

        # 原始数据
        self.mesh = None
        self.point_cloud = None

        # 预处理
        self.normalized = False
        self.scale = 1.0

        # 边界相关
        self.boundary_edges = None
        self.boundary_patch = None
        self.rim_curve = None

        # profile
        self.main_axis = None
        self.profile_curve = None

        # 特征
        self.profile_feature = None
        self.geometry_feature = None

        # 位姿（全局）
        self.pose = None
