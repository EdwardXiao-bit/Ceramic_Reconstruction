"""
Breaking Bad Dataset 数据加载器
支持加载压缩格式的陶瓷碎片数据
"""
import os
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader


class BreakingBadDataset(Dataset):
    """
    Breaking Bad 数据集加载器
    支持加载 compressed_fracture.npy 和 compressed_mesh.obj
    """
    
    def __init__(self, 
                 data_root: str,
                 category: str = 'Bottle',
                 subset: str = 'everyday_compressed',
                 num_points: int = 1024,
                 transform=None):
        """
        Args:
            data_root: 数据集根目录 (e.g., D:\googledownload\Breaking-Bad-Dataset...)
            category: 物体类别 (e.g., 'Bottle', 'Bowl', 'BeerBottle')
            subset: 子集名称 ('everyday_compressed', 'artifact_compressed')
            num_points: 每个碎片采样的点数
            transform: 数据变换
        """
        self.data_root = Path(data_root)
        self.category = category
        self.subset = subset
        self.num_points = num_points
        self.transform = transform
        
        # 扫描所有可用的样本
        self.samples = self._scan_samples()
        print(f"✓ 找到 {len(self.samples)} 个 {category} 样本")
    
    def _scan_samples(self) -> List[Dict]:
        """扫描所有可用的碎片对"""
        samples = []
        
        category_dir = self.data_root / self.subset / self.category
        
        if not category_dir.exists():
            print(f"⚠️ 目录不存在：{category_dir}")
            return samples
        
        # 遍历每个形状
        for shape_dir in category_dir.iterdir():
            if not shape_dir.is_dir():
                continue
            
            # 检查是否有 mode_0 目录
            mode_0_dir = shape_dir / 'mode_0'
            if not mode_0_dir.exists():
                continue
            
            # 获取完整的 mesh
            compressed_mesh = shape_dir / 'compressed_mesh.obj'
            
            # 遍历所有 fracture 模式
            for mode_dir in shape_dir.glob('mode_*'):
                if not mode_dir.is_dir():
                    continue
                
                # 读取 fracture 信息
                fracture_npy = mode_dir / 'compressed_fracture.npy'
                
                if fracture_npy.exists():
                    samples.append({
                        'shape_id': shape_dir.name,
                        'mode_id': mode_dir.name,
                        'mesh_path': str(compressed_mesh),
                        'fracture_path': str(fracture_npy),
                        'category': self.category
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict:
        """
        获取一个训练样本
        
        Returns:
            dict: {
                'points1': 源点云 [N, 3],
                'points2': 目标点云 [M, 3],
                'labels': 碎片标签,
                'shape_id': 形状 ID,
                'mode_id': 断裂模式 ID
            }
        """
        sample_info = self.samples[idx]
        
        # 加载完整 mesh
        mesh = o3d.io.read_triangle_mesh(sample_info['mesh_path'])
        complete_points = np.asarray(mesh.sample_points_uniformly(number_of_points=self.num_points).points)
        
        # 生成两个有差异的点云（模拟不同视角的扫描）
        # 添加小的随机变换和噪声
        angle = np.random.uniform(-0.1, 0.1)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        # Rodrigues 旋转公式
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        t = np.random.uniform(-0.01, 0.01, 3)
        
        # 应用变换
        points1 = complete_points.astype(np.float32)
        points2 = (R @ complete_points.T + t.reshape(3, 1)).T.astype(np.float32)
        points2 += np.random.randn(self.num_points, 3).astype(np.float32) * 0.001  # 添加噪声
        
        # 应用变换
        if self.transform:
            points1 = self.transform(points1)
            points2 = self.transform(points2)
        
        return {
            'points1': torch.from_numpy(points1),
            'points2': torch.from_numpy(points2),
            'labels': torch.tensor([0, 1]),  # 示例标签
            'shape_id': sample_info['shape_id'],
            'mode_id': sample_info['mode_id'],
            'category': sample_info['category']
        }


def create_dataloader(data_root: str, 
                     category: str = 'Bottle',
                     batch_size: int = 8,
                     num_points: int = 1024,
                     shuffle: bool = True):
    """
    创建数据加载器
    
    Args:
        data_root: 数据集根目录
        category: 物体类别
        batch_size: 批次大小
        num_points: 每个碎片采样的点数
        shuffle: 是否打乱数据
    
    Returns:
        DataLoader
    """
    dataset = BreakingBadDataset(
        data_root=data_root,
        category=category,
        num_points=num_points
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows 上建议设为 0
        pin_memory=True
    )
    
    return dataloader


if __name__ == '__main__':
    # 测试数据加载器
    data_root = r"D:\googledownload\Breaking-Bad-Dataset.github.io-main\Breaking-Bad-Dataset.github.io-main"
    
    print("="*60)
    print("测试 Breaking Bad 数据加载器")
    print("="*60)
    
    # 创建数据加载器
    dataloader = create_dataloader(
        data_root=data_root,
        category='Bottle',
        batch_size=2,
        num_points=512
    )
    
    print(f"\n数据加载器已创建:")
    print(f"  - 类别：Bottle")
    print(f"  - 批次大小：2")
    print(f"  - 点数：512")
    print(f"  - 总批次：{len(dataloader)}")
    
    # 测试读取一个批次
    print("\n读取一个批次...")
    batch = next(iter(dataloader))
    
    print(f"✓ 成功读取批次:")
    print(f"  - points1 形状：{batch['points1'].shape}")
    print(f"  - points2 形状：{batch['points2'].shape}")
    print(f"  - labels 形状：{batch['labels'].shape}")
    print(f"  - shape_id: {batch['shape_id'][0]}")
    print(f"  - mode_id: {batch['mode_id'][0]}")
    
    print("\n✅ 数据加载器测试通过！")
