import open3d as o3d
from src.common.io import load_fragments
from src.preprocessing.normalize import normalize_fragment
from src.boundary.detect import detect_boundary
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

    # 2. 逐个处理碎片：归一化→边界检测→轮廓提取→特征编码
    for f in fragments:
        print(f"处理碎片: {f.id}")

        # 修正1：传入Fragment对象f，而非f.point_cloud
        normalized_pcd, meta = normalize_fragment(f)
        if normalized_pcd is None:
            print(f"碎片{f.id}归一化失败，跳过后续处理")
            continue

        # 修正2：传入Fragment对象f，而非f.mesh
        if f.mesh is not None:
            f.boundary_patch = detect_boundary(f)

        # 修正3：传入Fragment对象f，而非f.point_cloud
        profile, axis = extract_profile(f)
        f.profile_curve = profile
        f.main_axis = axis

        # 编码轮廓特征
        encode_profile(f)

    # 3. 粗匹配碎片（注意：仅1个碎片时匹配结果为空）
    matches = coarse_match(fragments)
    print(f"匹配结果（前3对）: {matches[:3]}")

    # 4. 装配碎片（可视化结果）
    model = assemble(fragments, matches)

    # 5. 保存装配结果（可选）
    o3d.io.write_point_cloud("data/assembled_model.ply", model)
    print("MVP pipeline finished，装配结果已保存到data/assembled_model.ply")


if __name__ == "__main__":
    main()