# D:\ceramic_reconstruction\scripts\run_mvp.py
import sys
import os
import numpy as np
import open3d as o3d

# 手动添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# ===== 导入模块 =====
from src.common.io import load_fragments
from src.preprocessing.normalize import normalize_fragment

from src.boundary.detect import detect_boundary
from src.boundary.patch import extract_section_patch
from src.boundary.rim import extract_rim_curve

from src.profile.extract import extract_profile
from src.features.profile_feat import encode_profile

# 几何特征学习
from src.geometry_features.patch_encoder import PatchEncoder

# 可视化
from src.geometry_features.visualize import visualize_geo_embeddings

# 匹配 & 装配（暂不动）
from src.matching.coarse import coarse_match
from src.assembly.graph import assemble


def main():
    # ===== 1. 加载碎片 =====
    fragments = load_fragments("data/demo")
    if not fragments:
        print("未加载到任何碎片，请检查 data/demo")
        return

    # ===== 2. 初始化几何特征编码器 =====
    geo_encoder = PatchEncoder()

    # embedding 存盘目录
    emb_dir = "data/processed/geo_embeddings"
    os.makedirs(emb_dir, exist_ok=True)

    # ===== 3. 逐碎片处理 =====
    for f in fragments:
        print(f"\n===== 开始处理碎片: {f.id} =====")

        if f.point_cloud is None:
            print(f"碎片 {f.id} 无点云，跳过")
            continue

        # 3.1 归一化
        normalized_pcd, meta = normalize_fragment(f)
        if normalized_pcd is None:
            print(f"碎片 {f.id} 归一化失败")
            continue

        # 3.2 边界检测
        detect_boundary(f, visualize=True, curvature_thresh=0.1)

        # 3.3 断面 patch
        extract_section_patch(f, visualize=True)

        # 3.4 rim 曲线
        extract_rim_curve(f, visualize=True)

        # 3.5 profile（旧流程，保留）
        profile, axis = extract_profile(f)
        f.profile_curve = profile
        f.main_axis = axis
        for f in fragments:
            print(f"\n===== 开始处理碎片: {f.id} =====")
            # 轮廓提取
            extract_profile(f)
            # 轮廓编码（增加异常处理）
            try:
                encode_profile(f)
            except Exception as e:
                print(f"[处理碎片{f.id}] 轮廓编码失败：{str(e)}")
                f.profile_feature = None

        # ===== 3.6 几何特征学习编码 =====
        if hasattr(f, "section_patch") and f.section_patch is not None:
            geo_emb = geo_encoder.encode(f.section_patch)
            f.geo_embedding = geo_emb

            np.save(
                os.path.join(emb_dir, f"fragment_{f.id}_geo.npy"),
                geo_emb
            )

            print(f"[几何特征] 碎片 {f.id} embedding shape = {geo_emb.shape}")
        else:
            print(f"[几何特征] 碎片 {f.id} 无断面 patch")
            f.geo_embedding = None

    # ===== 4. embedding 可视化（PCA / t-SNE）=====
    geo_embs = []
    geo_ids = []
    for f in fragments:
        if hasattr(f, "geo_embedding") and f.geo_embedding is not None:
            geo_embs.append(f.geo_embedding)
            geo_ids.append(f.id)

    if len(geo_embs) >= 2:
        visualize_geo_embeddings(
            geo_embs,
            geo_ids,
            method="pca"  # 或 "tsne"
        )

    # ===== 5. 后续流程（暂不改）=====
    print("\n===== 开始碎片匹配与装配 =====")
    matches = coarse_match(fragments)
    model = assemble(fragments, matches)

    if model is not None:
        o3d.io.write_point_cloud("data/assembled_model.ply", model)
        print("装配结果已保存")

    print("\nMVP pipeline 执行完成！")


if __name__ == "__main__":
    main()
