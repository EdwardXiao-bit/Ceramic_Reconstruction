import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder for fracture patch
    Input: (B, N, 3)
    Output: (B, D)
    """
    def __init__(self, output_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        """
        x: (B, N, 3)
        """
        B, N, _ = x.shape
        x = self.mlp(x)          # (B, N, 256)
        x = torch.max(x, dim=1)[0]  # Global max pooling → (B, 256)
        x = self.fc(x)           # (B, D)
        x = F.normalize(x, dim=1)
        return x
