# D:\ceramic_reconstruction\scripts\run_mvp.py
import sys
import os
import open3d as o3d

# 手动添加项目根目录到Python搜索路径，解决No module named 'src'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# 导入核心模块（包括新实现的边界相关函数）
from src.common.io import load_fragments
from src.preprocessing.normalize import normalize_fragment
from src.boundary import detect_boundary, extract_section_patch, extract_rim_curve
from src.profile.extract import extract_profile
from src.features.profile_feat import encode_profile
from src.matching.coarse import coarse_match
from src.assembly.graph import assemble

def main():
    # 1. 加载碎片（data/demo目录下放.ply/.obj测试文件）
    fragments = load_fragments("data/demo")
    if not fragments:
        print("未加载到任何碎片，请检查data/demo目录")
        return

    # 2. 逐个处理碎片：归一化→边界检测→断面提取→rim提取→轮廓提取→特征编码
    for f in fragments:
        print(f"\n===== 开始处理碎片: {f.id} =====")
        # 2.1 点云归一化（必做，提升后续几何提取精度）
        if f.point_cloud is None:
            print(f"碎片{f.id}无点云数据，跳过所有处理")
            continue
        normalized_pcd, meta = normalize_fragment(f)
        if normalized_pcd is None:
            print(f"碎片{f.id}归一化失败，跳过后续处理")
            continue

        # 2.2 边界检测（核心前置，断面提取依赖此结果）
        # 若边界点过多，调大curvature_thresh（如0.2）；过少则调小（如0.05）
        detect_boundary(f, visualize=True, curvature_thresh=0.1)  # 新增曲率阈值参数

        # 2.3 断面patch提取（k_neighbors根据点云密度调整，默认50）
        extract_section_patch(f, k_neighbors=50, visualize=True)

        # 2.4 Rim曲线提取（n_bins根据器型调整，默认100）
        extract_rim_curve(f, n_bins=100, visualize=True)

        # 2.5 原有流程：轮廓提取+特征编码
        profile, axis = extract_profile(f)
        f.profile_curve = profile
        f.main_axis = axis
        encode_profile(f)

    # 3. 粗匹配+装配（原有流程）
    print(f"\n===== 开始碎片匹配与装配 =====")
    matches = coarse_match(fragments)
    print(f"匹配结果（前3对）: {matches[:3]}")
    model = assemble(fragments, matches)

    # 4. 保存装配结果
    if model is not None:
        o3d.io.write_point_cloud("data/assembled_model.ply", model)
        print("装配结果已保存到data/assembled_model.ply")

    print("\nMVP pipeline 执行完成！")

if __name__ == "__main__":
    main()