# 深度学习模型预训练实施指南

## 📋 目录

1. [快速开始](#快速开始)
2. [模型配置](#模型配置)
3. [下载预训练权重](#下载预训练权重)
4. [从头训练](#从头训练)
5. [集成到项目](#集成到项目)

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖（已安装）
pip install torch torchvision numpy open3d

# Predator 专用依赖（可选）
pip install MinkowskiEngine

# DCP 依赖（已满足）
# 使用 PyTorch 内置模块即可
```

### 2. 测试模型构建

```bash
# 测试 Predator 模型
python src/models/predator.py

# 测试 DCP 模型  
python src/models/dcp.py
```

### 3. 下载预训练权重

```bash
# 下载所有模型的预训练权重
python scripts/pretrain_models.py --model all --action download

# 只下载 Predator
python scripts/pretrain_models.py --model predator --action download

# 只下载 DCP
python scripts/pretrain_models.py --model dcp --action download
```

---

## 📦 模型配置

配置文件位于 `configs/` 目录：

- `predator.yaml` - Predator 配置
- `dcp.yaml` - DCP 配置  
- `pointnet2.yaml` - PointNet++ 配置

### 关键参数说明

**Predator:**
- `MODEL.TRANSFORMER.NUM_LAYERS`: Transformer 层数 (默认 6)
- `MODEL.UNET.PLANES`: ResUNet 各层通道数
- `TRAIN.BATCH_SIZE`: 批次大小 (默认 1，点云配准通常较小)

**DCP:**
- `MODEL.POINTNET.EMBEDDING_DIM`: PointNet 编码维度 (默认 1024)
- `MODEL.REGRESSOR.LAYERS`: 回归头网络结构 [512, 256, 128, 6]
- `TRAIN.NUM_EPOCHS`: 训练轮数 (默认 250)

---

## ⬇️ 下载预训练权重

### 方法 1: 自动下载（推荐）

```bash
python scripts/pretrain_models.py --action download
```

脚本会自动从以下来源下载：
- **Predator**: GitHub Releases / Google Drive
- **DCP**: 官方 GitHub 仓库

### 方法 2: 手动下载

如果自动下载失败，请手动下载并放置到 `models/weights/` 目录：

| 模型 | 下载地址 | 文件名 |
|------|---------|--------|
| Predator | https://github.com/Xiangyu-Wang-Ke/Awesome-Predator | `predator_3dmatch.pth` |
| DCP | https://github.com/oyyhjx/dcp | `dcp_modelnet.pth` |
| PointNet++ | https://github.com/charlesq34/pointnet2 | `pointnet2_cls.pth` |

### 验证下载

```bash
# 检查权重文件
ls -lh models/weights/

# 预期输出:
# superpoint_v1.pth        (5.1MB)  ✅
# superglue_indoor.pth     (47MB)   ✅
# superglue_outdoor.pth    (47MB)   ✅
# predator_3dmatch.pth     (~100MB) ⬜
# dcp_modelnet.pth         (~50MB)  ⬜
```

---

## 🔧 从头训练

### 准备数据集

**Predator 需要 3DMatch 数据集:**

```bash
# 下载 3DMatch
# 访问：http://3dvision.princeton.edu/projects/2016/3DMatch/
# 或使用简化版本：https://github.com/zgojcic/3DConvolutionalMatcher

# 数据集结构:
data/3dmatch/
├── train/
│   ├── kitchen/
│   ├── home/
│   └── ...
└── test/
```

**DCP 需要 ModelNet40 数据集:**

```bash
# 下载 ModelNet40
# https://modelnet.cs.princeton.edu/#

# 或使用预处理版本:
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
```

### 运行训练

```bash
# 训练 Predator
python scripts/pretrain_models.py --model predator --action train

# 训练 DCP
python scripts/pretrain_models.py --model dcp --action train

# 同时训练多个模型
python scripts/pretrain_models.py --model all --action both
```

### 训练监控

训练日志保存在:
- `logs/predator/` - Predator 训练日志
- `logs/dcp/` - DCP 训练日志

检查点保存在:
- `checkpoints/predator/` - Predator 检查点
- `checkpoints/dcp/` - DCP 检查点

---

## 🔗 集成到项目

### 1. 修改 FeatureMatcher 使用真实模型

编辑 `src/boundary_validation/feature_matcher.py`:

```python
def _load_predator_model(self):
    """加载真正的 Predator 模型"""
    try:
        from src.models.predator import Predator
        import yaml
        
        # 加载配置
        with open('configs/predator.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # 创建模型
        model = Predator(config['MODEL'])
        
        # 加载预训练权重
        weight_path = 'models/weights/predator_3dmatch.pth'
        if os.path.exists(weight_path):
            checkpoint = torch.load(weight_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("[特征匹配] Predator 模型加载成功")
        else:
            print("[特征匹配] 警告：未找到预训练权重，使用随机初始化")
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"[特征匹配] Predator 加载失败：{e}")
        return None  # 自动降级到 FPFH
```

### 2. 修改 LocalAligner 使用真实 DCP

编辑 `src/boundary_validation/local_aligner.py`:

```python
def _create_dcp_model(self):
    """创建真正的 DCP 模型"""
    try:
        from src.models.dcp import DCP
        import yaml
        
        # 加载配置
        with open('configs/dcp.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # 创建模型
        model = DCP(config['MODEL'])
        
        # 加载预训练权重
        weight_path = 'models/weights/dcp_modelnet.pth'
        if os.path.exists(weight_path):
            checkpoint = torch.load(weight_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("[局部对齐] DCP 模型加载成功")
        else:
            print("[局部对齐] 警告：未找到预训练权重")
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"[局部对齐] DCP 创建失败：{e}")
        return None  # 自动降级到 ICP
```

### 3. 运行测试

```bash
# 运行边界验证测试
python scripts/test_boundary_validation.py

# 观察日志确认模型是否被使用
# 查找："[特征匹配] Predator 模型加载成功"
# 查找："[局部对齐] DCP 模型加载成功"
```

---

## 🎯 性能对比

### Mock 模型 vs 预训练模型

| 模块 | Mock 模型 | 预训练模型 | 预期提升 |
|------|----------|-----------|---------|
| Predator | 随机匹配 | 学习到的特征 | +40-60% |
| DCP | 随机变换 | 精确的Δ(R,t) | +30-50% |
| 整体流程 | 基线 | 完整系统 | +50-80% |

### 评估指标

- **特征匹配**: 匹配准确率、召回率
- **对齐精度**: RMSE、Fitness Score
- **最终评分**: 边界验证总分

---

## 🐛 常见问题

### Q1: MinkowskiEngine 安装失败

**A:** 这是可选依赖，如果只需要 DCP 可以跳过 Predator。或者使用 Docker:

```bash
docker pull xiangyuwang/predator
```

### Q2: CUDA 内存不足

**A:** 减小 batch_size:

```yaml
# configs/predator.yaml
TRAIN:
  BATCH_SIZE: 1  # 保持为 1
```

### Q3: 训练收敛慢

**A:** 
1. 检查学习率是否合适
2. 增加数据增强
3. 使用预训练权重 fine-tune

---

## 📊 预期结果

使用预训练模型后，边界验证应该显示:

```
[特征匹配] Predator 匹配：150 对  (Mock: ~50 对)
[特征匹配] 重叠度得分：0.65  (Mock: ~0.0)
[局部对齐] DCP 精化完成
[局部对齐] 适应度得分：0.72  (Mock: ~0.0)
```

---

## 📝 下一步

1. ✅ 下载预训练权重
2. ⬜ 集成到项目模块
3. ⬜ 运行完整测试
4. ⬜ 在真实数据上评估
5. ⬜ （可选）领域自适应微调

---

## 🔗 参考资料

- **Predator**: https://arxiv.org/abs/2108.03279
- **DCP**: https://arxiv.org/abs/1903.07600
- **PointNet++**: https://arxiv.org/abs/1706.02413
- **3DMatch**: http://3dvision.princeton.edu/projects/2016/3DMatch/
