import torch
import numpy as np
from .pointnet import PointNetEncoder


class PatchEncoder:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = PointNetEncoder(output_dim=128).to(device)
        self.model.eval()  # MVP 阶段只做 inference

    def encode(self, section_patch, n_points=1024):
        """
        section_patch: o3d.PointCloud
        return: (128,) numpy array
        """
        pts = np.asarray(section_patch.points)

        # 1. 随机采样 / 重采样
        if len(pts) > n_points:
            idx = np.random.choice(len(pts), n_points, replace=False)
            pts = pts[idx]
        elif len(pts) < n_points:
            idx = np.random.choice(len(pts), n_points, replace=True)
            pts = pts[idx]

        # 2. 归一化（中心化 + 单位尺度）
        pts = pts - pts.mean(axis=0)
        scale = np.linalg.norm(pts, axis=1).max()
        pts = pts / (scale + 1e-6)

        # 3. 转 Tensor
        x = torch.from_numpy(pts).float().unsqueeze(0).to(self.device)

        # 4. Forward
        with torch.no_grad():
            emb = self.model(x)

        return emb.cpu().numpy().squeeze()
