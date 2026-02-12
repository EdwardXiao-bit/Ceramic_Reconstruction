# SuperGlue纹理匹配集成说明

## 概述

本模块实现了算法文档中"(二) 纹样提取与特征编码"和"(四) 碎片匹配初筛"中的纹样匹配部分，提供了与几何匹配并行的另一条技术路线。

## 功能特点

### 核心功能
- **SuperGlue特征匹配**: 基于图神经网络的鲁棒特征匹配
- **多模态特征融合**: 结合SuperGlue局部特征和全局纹理embedding
- **几何约束**: 考虑碎片尺寸、厚度等几何兼容性
- **批量处理**: 支持多碎片并行特征提取和匹配
- **结果可视化**: 生成匹配结果的可视化图像

### 技术特色
- **独立技术分支**: 与现有的几何匹配流程完全分离
- **配置化管理**: 支持多种预定义配置模板
- **缓存机制**: 特征提取结果可缓存，提高重复运行效率
- **兼容性设计**: SuperGlue不可用时自动降级到传统特征匹配

## 目录结构

```
src/texture_matching/
├── superglue_integration.py    # 基础SuperGlue集成
├── advanced_matching.py        # 高级匹配功能
├── config.py                   # 配置管理
└── __init__.py                 # 包初始化

scripts/
├── run_texture_matching.py           # 基础运行脚本
└── run_advanced_texture_matching.py  # 增强版运行脚本
```

## 安装要求

### SuperGlue依赖（可选）
```bash
# 克隆SuperGlue仓库
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
cd SuperGluePretrainedNetwork
pip install -e .

# 或者安装必要依赖
pip install torch torchvision opencv-python
```

### 其他依赖
已在项目requirements.txt中包含：
- opencv-python
- numpy
- open3d
- pyyaml

## 使用方法

### 1. 基础使用
```bash
# 使用默认配置运行
python scripts/run_texture_matching.py --data_dir data/demo

# 指定输出目录
python scripts/run_texture_matching.py --data_dir data/demo --output_dir results/my_experiment
```

### 2. 高级使用
```bash
# 创建配置文件
python scripts/run_advanced_texture_matching.py --create_config

# 使用预定义模板
python scripts/run_advanced_texture_matching.py --config_template high_precision

# 使用自定义配置
python scripts/run_advanced_texture_matching.py --config configs/my_config.yaml
```

### 3. 配置选项

#### 预定义模板
- `high_precision`: 高精度匹配（较慢但准确）
- `fast_matching`: 快速匹配（较快但精度稍低）
- `balanced`: 平衡模式（推荐默认）

#### 主要配置参数
```yaml
# SuperPoint配置
superpoint:
  nms_radius: 4              # 非极大值抑制半径
  keypoint_threshold: 0.005  # 关键点检测阈值
  max_keypoints: 1024        # 最大关键点数

# SuperGlue配置
superglue:
  weights: indoor            # 权重类型
  sinkhorn_iterations: 20    # Sinkhorn迭代次数
  match_threshold: 0.2       # 匹配阈值

# 匹配参数
matching:
  similarity_threshold: 0.3  # 相似度阈值
  min_matches: 10           # 最少匹配点数
  feature_weight: 0.7       # SuperGlue特征权重
  embedding_weight: 0.3     # 全局embedding权重
```

## 输出结果

### 文件结构
```
results/texture_matching/
├── texture_matches.txt      # 文本格式匹配结果
├── texture_matches.npz      # 数值数据
├── matching_report.json     # 详细匹配报告
├── matches_summary.csv      # CSV格式结果汇总
├── used_config.json         # 使用的配置参数
├── match_*_*_.png           # 匹配可视化图像
└── texture_features.pkl     # 特征缓存文件
```

### 结果说明
- **匹配对**: 按相似度排序的碎片对
- **相似度得分**: 综合考虑SuperGlue匹配、全局特征和几何兼容性
- **组件分数**: 分别显示各部分的贡献度
- **可视化**: 显示匹配的关键点和连线

## 技术细节

### 算法流程
1. **纹样区域提取**: 基于3D几何特征识别表面区域
2. **2D投影**: 将3D纹样区域正交投影到图像平面
3. **特征提取**: 使用SuperGlue提取局部和全局特征
4. **相似度计算**: 多模态特征融合计算综合相似度
5. **几何约束**: 考虑碎片尺寸兼容性
6. **候选筛选**: 基于阈值和Top-K选择最佳匹配

### 性能优化
- **特征缓存**: 避免重复特征提取
- **批处理**: 支持多碎片并行处理
- **内存管理**: 适时释放不需要的中间结果
- **GPU加速**: 自动使用CUDA（如果可用）

## 与其他模块的关系

### 与几何匹配的区别
- **输入**: 纹理匹配使用表面纹样图像，几何匹配使用断面几何
- **方法**: 纹理匹配基于2D图像特征，几何匹配基于3D几何特征
- **应用**: 纹理匹配适合有装饰图案的碎片，几何匹配适合断面匹配

### 协同工作
两个模块可以：
1. **独立运行**: 分别产生各自的候选匹配对
2. **结果融合**: 将两种匹配结果进行加权融合
3. **相互验证**: 用一种方法验证另一种方法的结果

## 故障排除

### 常见问题

1. **SuperGlue不可用**
   - 系统会自动降级到传统ORB特征匹配
   - 可以继续运行，但精度可能有所下降

2. **内存不足**
   - 减少`max_keypoints`参数
   - 降低图像分辨率
   - 分批处理大量碎片

3. **匹配结果不佳**
   - 调整`simalarity_threshold`
   - 尝试不同的配置模板
   - 检查输入数据质量

### 调试建议
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.INFO)

# 检查中间结果
matcher = TextureMatcher()
features = matcher.extract_superglue_features(image)
print(f"提取到 {len(features['keypoints'])} 个关键点")
```

## 开发指南

### 扩展功能
1. **添加新的特征提取器**: 继承`PatternEncoder`类
2. **实现新的匹配算法**: 扩展`AdvancedTextureMatcher`类
3. **增加可视化选项**: 修改`visualize_matches`方法

### 性能基准
- 单个碎片特征提取: ~2-5秒（取决于点云大小）
- 一对碎片匹配: ~1-3秒
- 10个碎片完整流程: ~5-15分钟

## 参考文献
- [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763)
- [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)

---
*此模块严格遵循算法文档设计，与几何匹配形成互补的技术路线*