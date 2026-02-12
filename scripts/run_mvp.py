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
from src.boundary.normalize import normalize_patch

from src.profile.extract import extract_profile
from src.features.profile_feat import encode_profile

# 几何特征学习
from src.geometry_features.patch_encoder import PatchEncoder
from src.geometry_features.traditional_feat import compute_patch_fpfh

# 可视化
from src.geometry_features.visualize import visualize_geo_embeddings

# 匹配 & 装配
from src.matching.faiss_prescreen import faiss_prescreen
from src.matching.results_saver import save_matching_results
from src.assembly.graph import assemble


def main():
    # ===== 1. 加载碎片 =====
    # 使用相对于项目根目录的路径，确保在不同环境下都能正确找到数据
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "demo")
    fragments = load_fragments(data_dir)
    if not fragments:
        print(f"未加载到任何碎片，请检查 {data_dir}")
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

        # ===== 3.3.1 质量检查：断面/边界点过少则跳过 rim/profile/特征编码 =====
        n_patch = len(f.section_patch.points) if (hasattr(f, "section_patch") and f.section_patch is not None) else 0
        n_boundary = len(f.boundary_pts) if (hasattr(f, "boundary_pts") and f.boundary_pts is not None) else 0
        MIN_PATCH_POINTS = 50
        MIN_BOUNDARY_POINTS = 20
        if n_patch < MIN_PATCH_POINTS or n_boundary < MIN_BOUNDARY_POINTS:
            print(f"[质量检查] 碎片{f.id} 断面点={n_patch}、边界点={n_boundary} 不足，跳过 rim/profile/几何编码（纯点云或低质量 mesh 常见）")
            f.geo_embedding = None
            continue

        # 3.4 rim 曲线
        extract_rim_curve(f, visualize=True)

        # 3.5 profile（旧流程，保留）
        profile, axis = extract_profile(f)
        f.profile_curve = profile
        f.main_axis = axis

        # 轮廓编码（仅对当前碎片，增加异常处理）
        try:
            encode_profile(f)
        except Exception as e:
            print(f"[处理碎片{f.id}] 轮廓编码失败：{str(e)}")
            f.profile_feature = None

        # ===== 3.5.1 边界规范化（尺度/局部坐标系/重采样/法向一致）=====
        normalize_patch(f, n_points=2048)

        # ===== 3.6 几何特征学习编码 =====
        if hasattr(f, "section_patch") and f.section_patch is not None:
            # PointNet 深度特征
            geo_emb = geo_encoder.encode(f.section_patch)
            f.geo_embedding = geo_emb

            # FPFH 传统特征（fallback）
            fpfh = compute_patch_fpfh(f.section_patch)
            f.fpfh_feature = fpfh

            np.save(
                os.path.join(emb_dir, f"fragment_{f.id}_geo.npy"),
                geo_emb
            )
            if fpfh is not None:
                np.save(
                    os.path.join(emb_dir, f"fragment_{f.id}_fpfh.npy"),
                    fpfh
                )

            fpfh_str = str(fpfh.shape) if fpfh is not None else "None"
            print(f"[几何特征] 碎片 {f.id} geo_embedding={geo_emb.shape}, fpfh={fpfh_str}")
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

    # ===== 5. 碎片匹配初筛（FAISS + 多模态相似度）& 装配 =====
    print("\n===== 开始碎片匹配初筛与装配 =====")
    
    # 执行FAISS初筛并获取详细信息
    matches, process_info = faiss_prescreen(
        fragments, 
        top_m_geo=50, 
        top_m_fpfh=50, 
        top_m_texture=30,  # 添加纹样特征支持
        top_k=10, 
        alpha=0.5,         # 调整权重以平衡三种特征
        beta=0.2,
        gamma=0.3
    )
    
    print(f"[匹配初筛] 得到 {len(matches)} 个候选对")
    
    # 保存详细的匹配结果
    if matches:
        print("\n===== 保存匹配结果 =====")
        save_matching_results(
            matches=matches,
            fragments=fragments,
            output_dir="results/matching",
            detailed_info=process_info
        )
    
    # 执行装配
    model = assemble(fragments, matches)

    if model is not None:
        o3d.io.write_point_cloud("data/assembled_model.ply", model)
        print("装配结果已保存")

    print("\nMVP pipeline 执行完成！")
    
    # 显示匹配统计摘要
    if matches and process_info:
        print("\n===== 匹配统计摘要 =====")
        print(f"总碎片数: {process_info.get('total_fragments', 'N/A')}")
        print(f"有效碎片数: {process_info.get('valid_fragments', 'N/A')}")
        print(f"使用的特征类型: {list(process_info.get('feature_types', {}).keys())}")
        if 'similarity_stats' in process_info:
            stats = process_info['similarity_stats']
            print(f"相似度统计 - 平均: {stats['mean']:.4f}, 最高: {stats['max']:.4f}, 最低: {stats['min']:.4f}")


if __name__ == "__main__":
    main()
