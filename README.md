# 陶瓷碎片三维重建系统

## 项目概述
基于多模态特征融合的陶瓷文物碎片自动匹配与重建系统。

## 主要功能
1. **几何特征匹配** - 基于断面几何特征的碎片匹配
2. **纹理特征匹配** - 基于表面装饰纹理的碎片匹配
3. **多模态融合** - 几何+纹理特征的综合匹配

## 目录结构
```
ceramic_reconstruction/
├── data/                 # 输入数据
├── models/              # SuperGlue模型文件
├── scripts/             # 运行脚本
├── src/                 # 源代码
│   ├── matching/        # 几何匹配模块
│   └── texture_matching/ # 纹理匹配模块
└── output/              # 输出结果
    ├── logs/           # 日志文件
    └── results/        # 匹配结果
```

## 运行方式

### 几何匹配
```bash
python scripts/run_mvp.py
```

### 纹理匹配
```bash
# 基础纹理匹配
python scripts/run_texture_matching.py

# 增强纹理匹配
python scripts/run_advanced_texture_matching.py

# 基于真实纹理贴图匹配
python scripts/run_texture_based_matching.py
```

## 依赖安装
```bash
pip install -r requirements.txt
```

## 技术特点
- 支持OBJ/PLY格式的3D模型
- 集成SuperGlue深度学习匹配算法
- 多模态特征融合匹配策略
- 完善的降级处理机制
