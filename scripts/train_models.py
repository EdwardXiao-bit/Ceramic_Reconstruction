"""
模型训练脚本
支持 Predator 和 DCP 的训练
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
import open3d as o3d
from typing import List, Dict
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.predator import Predator
from src.models.dcp import DCP
from src.datasets.breaking_bad import BreakingBadDataset, create_dataloader


# ─────────────────────────────────────────────
#  辅助：Rodrigues 公式生成随机旋转矩阵
# ─────────────────────────────────────────────
def random_rotation_matrix() -> np.ndarray:
    """生成随机旋转矩阵（float32）"""
    angle = np.random.uniform(-np.pi, np.pi)
    axis  = np.random.randn(3)
    axis  = axis / np.linalg.norm(axis)
    K = np.array([
        [0,       -axis[2],  axis[1]],
        [axis[2],  0,       -axis[0]],
        [-axis[1], axis[0],  0      ]
    ])
    R = (np.eye(3, dtype=np.float32)
         + np.sin(angle) * K
         + (1 - np.cos(angle)) * (K @ K))
    return R.astype(np.float32)


# ─────────────────────────────────────────────
#  Dataset：Breaking Bad 真实数据
# ─────────────────────────────────────────────
class BreakingBadPairDataset(Dataset):
    """
    Breaking Bad 数据集配对版本，用于训练配准模型。

    期望目录结构：
        data_root/
        └── everyday_compressed/
            └── <category>/
                └── <shape_id>/
                    ├── compressed_fracture.npy
                    └── compressed_mesh.obj

    构造函数接收的 data_root 应为包含 everyday_compressed 的**父目录**。
    """

    def __init__(self,
                 data_root: str,
                 category: str = 'Bottle',
                 num_points: int = 1024):
        self.num_points = num_points
        self.category   = category

        # ★ 修复 1：自动拼接 everyday 子目录
        base = Path(data_root)
        candidate = base / 'everyday'
        if candidate.exists():
            self.data_root = candidate
        else:
            # 兼容：用户直接传入了 everyday 路径
            self.data_root = base

        self.samples = self._scan_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"在 {self.data_root / category} 下未找到任何样本。\n"
                f"请确认路径正确，且目录中存在 compressed_fracture.npy 和 compressed_mesh.obj 文件。"
            )

        print(f"✓ 找到 {len(self.samples)} 个 {category} 样本")

    def _scan_samples(self) -> List[Dict]:
        samples = []
        category_dir = self.data_root / self.category

        if not category_dir.exists():
            print(f"⚠️  目录不存在：{category_dir}")
            return samples

        for shape_dir in sorted(category_dir.iterdir()):
            if not shape_dir.is_dir():
                continue

            fracture_file = shape_dir / 'compressed_fracture.npy'
            mesh_file     = shape_dir / 'compressed_mesh.obj'

            if fracture_file.exists() and mesh_file.exists():
                try:
                    fracture_data = np.load(fracture_file, allow_pickle=True).item()
                    num_modes = len(fracture_data.get('ids', []))
                except Exception as e:
                    print(f"  ⚠️  无法读取 {fracture_file}：{e}，跳过")
                    continue

                for mode_id in range(max(num_modes, 1)):
                    samples.append({
                        'shape_id':     shape_dir.name,
                        'mode_id':      mode_id,
                        'category':     self.category,
                        'fracture_file': str(fracture_file),
                        'mesh_file':     str(mesh_file)
                    })

        return samples

    def __len__(self):
        # ★ 修复 2：不再返回假的 1，空数据集由构造函数抛出异常
        return len(self.samples)

    def __getitem__(self, idx):
        # ★ 修复 2（同上）：self.samples 一定非空，无需取模保护
        sample_info = self.samples[idx]

        mesh   = o3d.io.read_triangle_mesh(sample_info['mesh_file'])
        points = np.asarray(mesh.vertices)

        # 采样到固定点数
        if len(points) >= self.num_points:
            indices        = np.random.choice(len(points), self.num_points, replace=False)
            complete_points = points[indices]
        else:
            # 不足则补采样
            extra   = np.random.choice(len(points),
                                       self.num_points - len(points),
                                       replace=True)
            complete_points = np.vstack([points, points[extra]])

        complete_points = complete_points.astype(np.float32)

        # 随机刚体变换
        R = random_rotation_matrix()
        t = np.random.uniform(-0.01, 0.01, 3).astype(np.float32)

        points1 = complete_points
        points2 = (R @ complete_points.T + t.reshape(3, 1)).T
        points2 += np.random.randn(self.num_points, 3).astype(np.float32) * 0.001

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = R
        transform[:3, 3]  = t

        return {
            'points1':   torch.from_numpy(points1),
            'points2':   torch.from_numpy(points2),
            'transform': torch.from_numpy(transform),
            'shape_id':  sample_info['shape_id'],
            'mode_id':   sample_info['mode_id'],
            'category':  sample_info['category']
        }


# ─────────────────────────────────────────────
#  Dataset：合成数据（Predator 演示用）
# ─────────────────────────────────────────────
class SyntheticPairDataset(Dataset):
    """合成点对数据集，用于 Predator 快速演示。"""

    def __init__(self, num_samples: int = 1000, num_points: int = 512):
        self.num_samples = num_samples
        self.num_points  = num_points

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        points1 = np.random.randn(self.num_points, 3).astype(np.float32)
        R = random_rotation_matrix()
        t = np.random.uniform(-1, 1, 3).astype(np.float32)

        points2 = (R @ points1.T + t.reshape(3, 1)).T
        points2 += np.random.randn(self.num_points, 3).astype(np.float32) * 0.01

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = R
        transform[:3, 3]  = t

        return points1, points2.astype(np.float32), transform


# ─────────────────────────────────────────────
#  Dataset：合成数据（DCP 演示用）
# ─────────────────────────────────────────────
class SyntheticDCPDataset(Dataset):
    """合成 DCP 数据集，返回 src、tgt、R_gt、t_gt。"""

    def __init__(self, num_samples: int = 2000, num_points: int = 1024):
        self.num_samples = num_samples
        self.num_points  = num_points

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        src_points = np.random.randn(self.num_points, 3).astype(np.float32)

        euler = np.random.uniform(-np.pi, np.pi, 3).astype(np.float32)
        cx, sx = np.cos(euler[0]), np.sin(euler[0])
        cy, sy = np.cos(euler[1]), np.sin(euler[1])
        cz, sz = np.cos(euler[2]), np.sin(euler[2])

        R_x = np.array([[1, 0,   0  ],
                         [0, cx, -sx ],
                         [0, sx,  cx ]], dtype=np.float32)
        R_y = np.array([[ cy, 0, sy],
                         [  0, 1,  0],
                         [-sy, 0, cy]], dtype=np.float32)
        R_z = np.array([[cz, -sz, 0],
                         [sz,  cz, 0],
                         [ 0,   0, 1]], dtype=np.float32)
        R_gt = R_z @ R_y @ R_x
        t_gt = np.random.uniform(-1, 1, 3).astype(np.float32)

        tgt_points = (R_gt @ src_points.T + t_gt.reshape(3, 1)).T.astype(np.float32)

        return src_points, tgt_points, R_gt, t_gt


# ─────────────────────────────────────────────
#  训练：Breaking Bad 数据集 + Predator
# ─────────────────────────────────────────────
def train_breaking_bad(config_path: str, output_dir: str, data_root: str,
                       category: str = 'Bottle',
                       epochs: int = 50, batch_size: int = 8):
    print("\n" + "=" * 60)
    print(f"使用 Breaking Bad 数据集训练 ({category})")
    print("=" * 60)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[设备] 使用：{device}")

    model = Predator(config['MODEL']).to(device)
    print(f"[参数] 总参数量：{sum(p.numel() for p in model.parameters()):,}")

    criterion  = nn.MSELoss()
    optimizer  = optim.AdamW(
        model.parameters(),
        lr=config['TRAIN']['LR'],
        weight_decay=config['TRAIN']['WEIGHT_DECAY']
    )

    print(f"[数据] 加载 Breaking Bad 数据集 ...")
    # ★ 修复 1：构造函数内部会自动处理 everyday_compressed 子目录
    train_dataset = BreakingBadPairDataset(
        data_root=data_root,
        category=category,
        num_points=512
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[训练] 开始训练 {epochs} 轮 ...")
    print("-" * 60)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch in pbar:
            points1      = batch['points1'].to(device)
            points2      = batch['points2'].to(device)
            gt_transform = batch['transform'].to(device)

            optimizer.zero_grad()
            _, _, transform_pred = model(points1, points2)

            # ★ 修复 3：同时监督旋转和平移
            gt_R = gt_transform[:, :3, :3]          # (B, 3, 3)
            gt_t = gt_transform[:, :3, 3]            # (B, 3)

            # transform_pred 假设为 (B, 6)：前 3 为旋转轴角，后 3 为平移
            # 若模型输出格式不同，请对应调整切片索引
            rot_loss   = criterion(transform_pred[:, :3],
                                   _rotation_to_axis_angle(gt_R))
            trans_loss = criterion(transform_pred[:, 3:], gt_t)
            loss = rot_loss + trans_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = output_path / f'predator_breaking_bad_{category.lower()}_best.pth'
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':                avg_loss,
                'config':              config,
                'dataset':             'BreakingBad',
                'category':            category
            }, ckpt_path)
            print(f"  [保存] 最佳模型 → {ckpt_path}")

        print(f"Epoch {epoch + 1}/{epochs}  Loss: {avg_loss:.4f}")

    print("\n" + "=" * 60)
    print(f"训练完成！最佳损失：{best_loss:.4f}")
    print("=" * 60)


# ─────────────────────────────────────────────
#  训练：合成数据 + Predator
# ─────────────────────────────────────────────
def train_predator(config_path: str, output_dir: str,
                   epochs: int = 10, batch_size: int = 1):
    print("\n" + "=" * 60)
    print("训练 Predator 模型（合成数据）")
    print("=" * 60)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[设备] 使用：{device}")

    model = Predator(config['MODEL']).to(device)
    print(f"[参数] 总参数量：{sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['TRAIN']['LR'],
        weight_decay=config['TRAIN']['WEIGHT_DECAY']
    )

    print("[数据] 创建合成数据集 ...")
    train_dataset = SyntheticPairDataset(num_samples=1000, num_points=512)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[训练] 开始训练 {epochs} 轮 ...")
    print("-" * 60)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for points1, points2, gt_transform in pbar:
            points1      = points1.to(device)
            points2      = points2.to(device)
            gt_transform = gt_transform.to(device)

            optimizer.zero_grad()
            _, _, transform_pred = model(points1, points2)

            # ★ 修复 3：同时监督旋转和平移
            gt_R = gt_transform[:, :3, :3]
            gt_t = gt_transform[:, :3, 3]

            rot_loss   = criterion(transform_pred[:, :3],
                                   _rotation_to_axis_angle(gt_R))
            trans_loss = criterion(transform_pred[:, 3:], gt_t)
            loss = rot_loss + trans_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = output_path / 'predator_best.pth'
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':                avg_loss,
                'config':              config
            }, ckpt_path)
            print(f"  [保存] 最佳模型 → {ckpt_path}")

        print(f"Epoch {epoch + 1}/{epochs}  Loss: {avg_loss:.4f}")

    print("\n" + "=" * 60)
    print(f"训练完成！最佳损失：{best_loss:.4f}")
    print("=" * 60)


# ─────────────────────────────────────────────
#  训练：DCP
# ─────────────────────────────────────────────
def train_dcp(config_path: str, output_dir: str,
              epochs: int = 50, batch_size: int = 8):
    print("\n" + "=" * 60)
    print("训练 DCP 模型")
    print("=" * 60)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[设备] 使用：{device}")

    model = DCP(config['MODEL']).to(device)
    print(f"[参数] 总参数量：{sum(p.numel() for p in model.parameters()):,}")

    rot_criterion   = nn.MSELoss()
    trans_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['TRAIN']['LR'])

    print("[数据] 创建合成 DCP 数据集 ...")
    train_dataset = SyntheticDCPDataset(num_samples=2000, num_points=1024)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[训练] 开始训练 {epochs} 轮 ...")
    print("-" * 60)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for src_points, tgt_points, R_gt, t_gt in pbar:
            src_points = src_points.to(device)
            tgt_points = tgt_points.to(device)
            R_gt       = R_gt.to(device)
            t_gt       = t_gt.to(device)

            optimizer.zero_grad()
            R_pred, t_pred = model(src_points, tgt_points)

            loss = rot_criterion(R_pred, R_gt) + trans_criterion(t_pred, t_gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = output_path / 'dcp_best.pth'
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':                avg_loss,
                'config':              config
            }, ckpt_path)
            print(f"  [保存] 最佳模型 → {ckpt_path}")

        print(f"Epoch {epoch + 1}/{epochs}  Loss: {avg_loss:.4f}")

    print("\n" + "=" * 60)
    print(f"训练完成！最佳损失：{best_loss:.4f}")
    print("=" * 60)


# ─────────────────────────────────────────────
#  辅助：旋转矩阵 → 轴角向量（用于 Predator 损失）
# ─────────────────────────────────────────────
def _rotation_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵 (B, 3, 3) 转换为轴角向量 (B, 3)。
    角度编码在向量模长中：v = angle * axis。
    """
    # 旋转角度：theta = arccos((trace(R) - 1) / 2)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    theta = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))  # (B,)

    # 旋转轴（Rodrigues 反解）
    eps = 1e-6
    axis = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1]
    ], dim=1)  # (B, 3)

    sin_theta = torch.sin(theta).unsqueeze(1).clamp(min=eps)
    axis = axis / (2.0 * sin_theta)
    axis = torch.nn.functional.normalize(axis, dim=1)

    return axis * theta.unsqueeze(1)  # (B, 3)


# ─────────────────────────────────────────────
#  主函数（argparse 替代 input）
# ─────────────────────────────────────────────
def parse_args():
    # 项目根目录 = 本脚本所在目录的上一级（scripts/ → project_root/）
    _project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description='点云配准模型训练工具')

    parser.add_argument('--mode', type=str,
                        choices=['predator', 'dcp', 'breaking_bad', 'both'],
                        default=None,
                        help='训练模式')
    parser.add_argument('--predator_config', type=str,
                        default=str(_project_root / 'configs' / 'predator.yaml'))
    parser.add_argument('--dcp_config', type=str,
                        default=str(_project_root / 'configs' / 'dcp.yaml'))
    parser.add_argument('--output_dir', type=str,
                        default=str(_project_root / 'pretrained_weights'))
    parser.add_argument('--data_root', type=str,
                        default=None,
                        help='Breaking Bad 数据集根目录（含 everyday_compressed 的父目录）')
    parser.add_argument('--category', type=str, default='Bottle',
                        choices=['Bottle', 'Bowl', 'BeerBottle', 'Mug', 'Vase'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--synthetic', action='store_true',
                        help='Predator/DCP 使用合成数据快速演示')

    return parser.parse_args()


def main():
    print("\n" + "=" * 60)
    print("模型训练工具")
    print("=" * 60)

    args = parse_args()

    # ── 交互式补全未填参数（命令行已填则跳过）──
    if args.mode is None:
        print("\n可用训练模式:")
        print("  predator      - 训练 Predator 模型 (3DMatch)")
        print("  dcp           - 训练 DCP 模型 (ModelNet40)")
        print("  breaking_bad  - 使用 Breaking Bad 数据集训练")
        print("  both          - 同时训练 Predator 和 DCP")
        args.mode = input("\n请选择训练模式: ").strip().lower()

    if args.mode not in ['predator', 'dcp', 'breaking_bad', 'both']:
        print("❌ 无效的模式")
        return

    models_to_train = (['predator', 'dcp'] if args.mode == 'both'
                       else [args.mode])

    for model_name in models_to_train:
        print(f"\n{'=' * 60}")
        print(f"开始训练 {model_name.upper()}")
        print(f"{'=' * 60}\n")

        try:
            if model_name == 'predator':
                use_syn = args.synthetic or (
                    input("使用合成数据快速演示？(y/n) [y]: ").strip().lower() in ('', 'y')
                )
                train_predator(
                    config_path=args.predator_config,
                    output_dir=str(Path(args.output_dir) / 'predator'),
                    epochs=args.epochs or (5 if use_syn else 100),
                    batch_size=args.batch_size or (2 if use_syn else 1)
                )

            elif model_name == 'dcp':
                use_syn = args.synthetic or (
                    input("使用合成数据快速演示？(y/n) [y]: ").strip().lower() in ('', 'y')
                )
                train_dcp(
                    config_path=args.dcp_config,
                    output_dir=str(Path(args.output_dir) / 'dcp'),
                    epochs=args.epochs or (10 if use_syn else 250),
                    batch_size=args.batch_size or (4 if use_syn else 8)
                )

            elif model_name == 'breaking_bad':
                # 数据根目录
                if args.data_root is None:
                    default_root = (
                        r"D:\googledownload\Breaking-Bad-Dataset.github.io-main"
                        r"\Breaking-Bad-Dataset.github.io-main"
                    )
                    user_input = input(
                        f"\nBreaking Bad 数据集根目录 [{default_root}]: "
                    ).strip()
                    data_root = user_input if user_input else default_root
                else:
                    data_root = args.data_root

                print(f"\n训练配置:")
                print(f"  数据集路径 : {data_root}")
                print(f"  类别       : {args.category}")
                print(f"  配置文件   : {args.predator_config}")

                if input("\n开始训练？(y/n): ").strip().lower() != 'y':
                    print("训练已取消")
                    continue

                train_breaking_bad(
                    config_path=args.predator_config,
                    output_dir=str(Path(args.output_dir) / 'breaking_bad'),
                    data_root=data_root,
                    category=args.category,
                    epochs=args.epochs or 50,
                    batch_size=args.batch_size or 4
                )

        except RuntimeError as e:
            print(f"\n❌ {model_name} 训练失败（数据/配置错误）：{e}")
        except Exception as e:
            print(f"\n❌ {model_name} 训练失败：{e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("训练工具执行完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
