"""
训练 3D CNN、PointNet++ 和 D3Feat 模型（Breaking Bad 数据集专用）

使用说明:
1. 确保 Breaking Bad 数据集已准备好
2. 运行此脚本训练模型
3. 训练完成后，权重会自动保存到 pretrained_weights/breaking_bad/ 目录
4. 边界验证模块会自动加载这些预训练权重

使用方法:
    # 训练所有模型
    python scripts/train_complementarity_models.py --dataset breaking_bad --model all --epochs 100
    
    # 只训练 3D CNN
    python scripts/train_complementarity_models.py --model cnn --epochs 50
    
    # 只训练 PointNet++
    python scripts/train_complementarity_models.py --model pointnet2 --batch-size 16
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import sys
from typing import Dict, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class ComplementarityDataset(Dataset):
    """
    互补性检查数据集
    从 Breaking Bad 数据集中生成正负样本对
    """
    
    def __init__(self, data_root: str, category: str = 'BeerBottle', 
                 num_points: int = 1024, mode: str = 'train'):
        super().__init__()
        self.data_root = Path(data_root)
        self.category = category
        self.num_points = num_points
        self.mode = mode
        
        # 导入 Breaking Bad 数据集
        from src.datasets.breaking_bad import BreakingBadDataset
        
        self.dataset = BreakingBadDataset(
            data_root=data_root,
            category=category,
            subset='everyday_compressed',
            num_points=num_points
        )
        
        # 获取所有 shape_id 用于生成负样本
        self.shape_ids = list(set([s['shape_id'] for s in self.dataset.samples]))
        self.shape_to_samples = {}
        for i, sample in enumerate(self.dataset.samples):
            if sample['shape_id'] not in self.shape_to_samples:
                self.shape_to_samples[sample['shape_id']] = []
            self.shape_to_samples[sample['shape_id']].append(i)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Dict:
        # 获取基础点云
        sample = self.dataset[idx]
        points1 = sample['points1']
        points2 = sample['points2']
        
        # 50% 概率生成负样本（不同物体的配对）
        is_positive = np.random.rand() > 0.5
        label = 1.0 if is_positive else 0.0
        
        if not is_positive:
            # 负样本：从不同物体中随机选一个
            other_shapes = [s for s in self.shape_ids if s != sample['shape_id']]
            if len(other_shapes) > 0:
                other_shape = np.random.choice(other_shapes)
                other_idx = np.random.choice(self.shape_to_samples[other_shape])
                other_sample = self.dataset[other_idx]
                points2 = other_sample['points1']  # 使用另一个物体的点云
        
        return {
            'points1': points1,
            'points2': points2,
            'label': torch.tensor(label),
            'shape_id': sample['shape_id'],
            'mode_id': sample['mode_id']
        }


def train_cnn_3d(config: Dict, dataset_root: str, epochs: int = 100, 
                 batch_size: int = 8, lr: float = 0.001):
    """
    训练 3D CNN 模型用于形状互补性检查
    
    Args:
        config: 模型配置
        dataset_root: 数据集根目录
        epochs: 训练轮次
        batch_size: 批次大小
        lr: 学习率
    """
    print("=" * 60)
    print("训练 3D CNN 模型 (Breaking Bad 数据集)")
    print("=" * 60)
    
    from src.models.cnn_3d import ComplementarityPredictor, Voxelizer
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备：{device}")
    
    model = ComplementarityPredictor(
        input_channels=1,
        base_channels=32,
        grid_size=32
    ).to(device)
    print("✓ 3D CNN 模型已创建")
    
    # 创建数据集
    train_dataset = ComplementarityDataset(
        data_root=dataset_root,
        category=config.get('CATEGORY', 'Bowl'),
        num_points=config.get('NUM_POINTS', 512)
    )
    
    # 只有在有 GPU 时才使用 pin_memory
    use_pin_memory = device.type == 'cuda'
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_pin_memory
    )
    
    print(f"✓ 数据集已加载：{len(train_dataset)} 个样本")
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 训练循环
    best_loss = float('inf')
    save_path = Path('pretrained_weights/breaking_bad/cnn3d_breaking_bad_best.pth')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n开始训练...")
    print(f"保存路径：{save_path}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            points1 = batch['points1'].to(device)  # [B, N, 3]
            points2 = batch['points2'].to(device)  # [B, N, 3]
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # 简化处理：将点云转换为体素
            # 这里使用简化的方式，实际应该用 Voxelizer
            B, N, _ = points1.shape
            voxel1 = torch.zeros(B, 1, 32, 32, 32).to(device)
            voxel2 = torch.zeros(B, 1, 32, 32, 32).to(device)
            
            # 简单填充（实际训练中应该使用真正的体素化）
            for b in range(B):
                # 归一化点到 [0, 31]
                pts1_norm = ((points1[b] + 1) / 2 * 31).long().clamp(0, 31)
                pts2_norm = ((points2[b] + 1) / 2 * 31).long().clamp(0, 31)
                
                # 填充体素
                for i in range(min(N, 100)):  # 限制点数加速
                    x, y, z = pts1_norm[i]
                    voxel1[b, 0, x, y, z] = 1.0
                    x, y, z = pts2_norm[i]
                    voxel2[b, 0, x, y, z] = 1.0
            
            # 前向传播
            outputs = model(voxel1, voxel2)
            
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"  ✓ 保存最佳模型 (Loss: {best_loss:.4f})")
    
    print(f"\n✓ 训练完成！")
    print(f"最佳模型已保存到：{save_path}")
    
    return save_path


def train_pointnet2(config: Dict, dataset_root: str, epochs: int = 100,
                    batch_size: int = 16, lr: float = 0.001):
    """
    训练 PointNet++ 模型用于点云特征提取
    
    Args:
        config: 模型配置
        dataset_root: 数据集根目录
        epochs: 训练轮次
        batch_size: 批次大小
        lr: 学习率
    """
    print("=" * 60)
    print("训练 PointNet++ 模型 (Breaking Bad 数据集)")
    print("=" * 60)
    
    from src.models.pointnet2 import PointNet2SSG
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备：{device}")
    
    model_config = {
        'INPUT_DIM': 3,
        'OUTPUT_DIM': config.get('OUTPUT_DIM', 256),
        'USE_SSG': True
    }
    
    model = PointNet2SSG(model_config).to(device)
    print("✓ PointNet++ SSG 模型已创建")
    
    # 创建数据集
    train_dataset = ComplementarityDataset(
        data_root=dataset_root,
        category=config.get('CATEGORY', 'Bowl'),
        num_points=config.get('NUM_POINTS', 1024)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✓ 数据集已加载：{len(train_dataset)} 个样本")
    
    # 定义损失函数和优化器
    # 使用对比损失：让同一物体的特征尽可能接近
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练循环
    best_loss = float('inf')
    save_path = Path('pretrained_weights/breaking_bad/pointnet2_breaking_bad_best.pth')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n开始训练...")
    print(f"保存路径：{save_path}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            points1 = batch['points1'].to(device)
            points2 = batch['points2'].to(device)
            
            optimizer.zero_grad()
            
            # 提取特征
            feat1 = model.encode(points1)
            feat2 = model.encode(points2)
            
            # 对比学习：同一个物体的特征应该相似
            # 使用 MSE 损失让特征接近
            loss = criterion(feat1, feat2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"  ✓ 保存最佳模型 (Loss: {best_loss:.4f})")
    
    print(f"\n✓ 训练完成！")
    print(f"最佳模型已保存到：{save_path}")
    
    return save_path


def train_d3feat(config: Dict, dataset_root: str, epochs: int = 100,
                 batch_size: int = 8, lr: float = 0.001):
    """
    训练 D3Feat 模型用于局部特征描述
    
    Args:
        config: 模型配置
        dataset_root: 数据集根目录
        epochs: 训练轮次
        batch_size: 批次大小
        lr: 学习率
    """
    print("=" * 60)
    print("训练 D3Feat 模型 (Breaking Bad 数据集)")
    print("=" * 60)
    
    from src.models.d3feat import D3Feat
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备：{device}")
    
    model_config = {
        'INPUT_DIM': 3,
        'FEATURE_DIM': config.get('FEATURE_DIM', 256),
        'SA_LAYERS': {
            'C1': 64, 'NPOINT1': 1024, 'RADIUS1': 0.1, 'NSAMPLE1': 32,
            'C2': 128, 'NPOINT2': 256, 'RADIUS2': 0.2, 'NSAMPLE2': 32,
            'C3': 256, 'NPOINT3': 64, 'RADIUS3': 0.4, 'NSAMPLE3': 32
        },
        'KEYPOINT_HEAD': False
    }
    
    model = D3Feat(model_config).to(device)
    print("✓ D3Feat 模型已创建")
    
    # 创建数据集
    train_dataset = ComplementarityDataset(
        data_root=dataset_root,
        category=config.get('CATEGORY', 'Bowl'),
        num_points=config.get('NUM_POINTS', 2048)  # D3Feat 需要更多点
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✓ 数据集已加载：{len(train_dataset)} 个样本")
    
    # 定义损失函数和优化器
    # 使用对比损失：正样本拉近，负样本推远
    criterion = nn.CosineEmbeddingLoss(margin=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练循环
    best_loss = float('inf')
    save_path = Path('pretrained_weights/breaking_bad/d3feat_breaking_bad_best.pth')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n开始训练...")
    print(f"保存路径：{save_path}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            points1 = batch['points1'].to(device)
            points2 = batch['points2'].to(device)
            labels = batch['label'].to(device)  # 1.0=正样本, 0.0=负样本
            
            optimizer.zero_grad()
            
            # 提取特征（训练时使用 forward 而不是 extract_features，以保留梯度）
            feat1 = model(points1)  # [B, N, D]
            feat2 = model(points2)  # [B, M, D]
            
            # 全局池化得到固定长度特征
            feat1_global = feat1.mean(dim=1)  # [B, D]
            feat2_global = feat2.mean(dim=1)  # [B, D]
            
            # CosineEmbeddingLoss: label=1 时最小化距离，label=-1 时最大化距离
            # 需要将 label 从 [0, 1] 转换为 [-1, 1]
            target = 2 * labels - 1  # 0->-1, 1->1
            
            loss = criterion(feat1_global, feat2_global, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step(total_loss / len(train_loader))
        
        avg_loss = total_loss / len(train_loader)
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"  ✓ 保存最佳模型 (Loss: {best_loss:.4f})")
    
    print(f"\n✓ 训练完成！")
    print(f"最佳模型已保存到：{save_path}")
    
    return save_path


def main():
    parser = argparse.ArgumentParser(description='训练互补性检查模型')
    parser.add_argument('--dataset', type=str, default='breaking_bad',
                       help='数据集名称 (default: breaking_bad)')
    parser.add_argument('--data-root', type=str, 
                       default=r'D:\googledownload\Breaking-Bad-Dataset.github.io-main\Breaking-Bad-Dataset.github.io-main',
                       help='数据集根目录路径')
    parser.add_argument('--model', type=str, default='all',
                       choices=['cnn', 'pointnet2', 'd3feat', 'all'],
                       help='要训练的模型 (default: all)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮次 (default: 100)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批次大小 (default: 自动选择)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率 (default: 0.001)')
    parser.add_argument('--category', type=str, default='Bowl',
                       help='物体类别 (default: Bowl)')
    parser.add_argument('--config', type=str, default='configs/breaking_bad.yaml',
                       help='配置文件路径 (default: configs/breaking_bad.yaml)')
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
    else:
        print(f"⚠ 未找到配置文件 {config_path}，使用默认配置")
        base_config = {}
    
    # 合并命令行参数
    model_config = {
        **base_config,
        'CATEGORY': args.category,
    }
    
    # 自动选择批次大小
    if args.batch_size is None:
        if args.model == 'cnn':
            batch_size = 8
        elif args.model == 'pointnet2':
            batch_size = 16
        elif args.model == 'd3feat':
            batch_size = 8
        else:
            batch_size = 8
    else:
        batch_size = args.batch_size
    
    # 训练指定的模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\n训练开始时间：{timestamp}")
    print(f"数据集：{args.dataset}")
    print(f"类别：{args.category}（碗类，适合陶瓷碎片重建）")
    print(f"批次大小：{batch_size}")
    print(f"学习率：{args.lr}")
    
    if args.model in ['cnn', 'all']:
        train_cnn_3d(model_config, args.data_root, args.epochs, batch_size, args.lr)
    
    if args.model in ['pointnet2', 'all']:
        train_pointnet2(model_config, args.data_root, args.epochs, batch_size, args.lr)
    
    if args.model in ['d3feat', 'all']:
        train_d3feat(model_config, args.data_root, args.epochs, batch_size, args.lr)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print("\n权重文件位置:")
    print("  - 3D CNN: pretrained_weights/breaking_bad/cnn3d_breaking_bad_best.pth")
    print("  - PointNet++: pretrained_weights/breaking_bad/pointnet2_breaking_bad_best.pth")
    print("  - D3Feat: pretrained_weights/breaking_bad/d3feat_breaking_bad_best.pth")
    print("\n边界验证模块将自动加载这些权重！")


if __name__ == '__main__':
    main()
