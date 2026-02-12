# D:\Users\Lenovo\Documents\GitHub\Ceramic_Reconstruction\scripts\run_mvp.py

import sys
import os
import json
from pathlib import Path
import open3d as o3d
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import load_fragments
from src.preprocessing.normalize import normalize_fragment
from src.boundary import detect_boundary_robust, extract_section_patch, extract_rim_curve
from src.profile.extract import extract_profile
from src.features.profile_feat import encode_profile
from src.matching.coarse import coarse_match
from src.assembly.graph import assemble


# ============================================================
# 可视化配置
# ============================================================
class VisualizationMode:
    NONE = 0
    SAVE_ONLY = 1
    INTERACTIVE = 2


# 🔧 默认使用交互模式
VIS_MODE = VisualizationMode.INTERACTIVE


def visualize_step(geometry_list, window_name="Visualization", mode=VIS_MODE):
    if mode == VisualizationMode.NONE:
        return
    if not isinstance(geometry_list, list):
        geometry_list = [geometry_list]

    if mode == VisualizationMode.INTERACTIVE:
        o3d.visualization.draw_geometries(
            geometry_list,
            window_name=window_name,
            width=1024,
            height=768,
            left=50,
            top=50,
            point_show_normal=False,
        )
    elif mode == VisualizationMode.SAVE_ONLY:
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=window_name,
            width=1024,
            height=768,
            visible=False,
        )
        for geom in geometry_list:
            vis.add_geometry(geom)
        vis.poll_events()
        vis.update_renderer()
        output_dir = PROJECT_ROOT / "data" / "output" / "screenshots"
        output_dir.mkdir(parents=True, exist_ok=True)
        img_path = output_dir / f"{window_name.replace(' ', '_')}.png"
        vis.capture_screen_image(str(img_path))
        vis.destroy_window()
        print(f"  📸 截图已保存: {img_path}")


def visualize_boundary_extraction(fragment, mode=VIS_MODE):
    if mode == VisualizationMode.NONE:
        return
    if not hasattr(fragment, "boundary_points") or fragment.boundary_points is None:
        return

    geoms = []
    if fragment.mesh:
        vis_base = o3d.geometry.TriangleMesh(fragment.mesh)
        vis_base.compute_vertex_normals()
        vis_base.paint_uniform_color([0.8, 0.8, 0.8])
    else:
        vis_base = o3d.geometry.PointCloud(fragment.point_cloud)
        vis_base.paint_uniform_color([0.8, 0.8, 0.8])
    geoms.append(vis_base)

    vis_boundary = o3d.geometry.PointCloud(fragment.boundary_points)
    vis_boundary.paint_uniform_color([1.0, 0.0, 0.0])

    if vis_boundary.has_normals():
        pts = np.asarray(vis_boundary.points)
        nms = np.asarray(vis_boundary.normals)
        vis_boundary.points = o3d.utility.Vector3dVector(pts + nms * 0.005)

    geoms.append(vis_boundary)
    visualize_step(
        geoms, window_name=f"Boundary_Robust_{fragment.id}", mode=mode
    )


def visualize_section_patch(fragment, mode=VIS_MODE):
    if mode == VisualizationMode.NONE:
        return
    if not hasattr(fragment, "section_patch"):
        return

    vis_base = o3d.geometry.PointCloud(fragment.point_cloud)
    vis_base.paint_uniform_color([0.8, 0.8, 0.8])

    vis_patch = o3d.geometry.PointCloud(fragment.section_patch)
    vis_patch.paint_uniform_color([0.0, 0.4, 0.8])

    if vis_patch.has_normals():
        vis_patch.points = o3d.utility.Vector3dVector(
            np.asarray(vis_patch.points) + np.asarray(vis_patch.normals) * 0.01
        )

    visualize_step(
        [vis_base, vis_patch], window_name=f"Patch_{fragment.id}", mode=mode
    )


def visualize_rim(fragment, mode=VIS_MODE, always_show=False):
    if mode == VisualizationMode.NONE:
        return
    if not hasattr(fragment, "rim_curve"):
        return

    rim_curve = fragment.rim_curve
    if len(rim_curve) < 10:
        return

    rim_pcd = o3d.geometry.PointCloud()
    rim_pcd.points = o3d.utility.Vector3dVector(rim_curve)
    rim_pcd.paint_uniform_color([0, 1, 0])

    lines = [[i, (i + 1) % len(rim_curve)] for i in range(len(rim_curve))]
    rim_lines = o3d.geometry.LineSet()
    rim_lines.points = o3d.utility.Vector3dVector(rim_curve)
    rim_lines.lines = o3d.utility.Vector2iVector(lines)
    rim_lines.paint_uniform_color([0, 1, 0])

    vis_geoms = [fragment.point_cloud, rim_pcd, rim_lines]
    if always_show or (
        mode == VisualizationMode.INTERACTIVE and len(vis_geoms) > 0
    ):
        visualize_step(
            vis_geoms, window_name=f"Rim_{fragment.id}", mode=mode
        )


def main():
    data_dir = PROJECT_ROOT / "data" / "eg1"
    output_dir = PROJECT_ROOT / "data" / "output"

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    print("=" * 60)
    print("开始重建流程 (Robust Version)")
    print("=" * 60)

    fragments = load_fragments(str(data_dir))
    if not fragments:
        print("成功加载 0 个碎片")
        return

    print(f"成功加载 {len(fragments)} 个碎片")

    successful_fragments = []

    for i, f in enumerate(fragments, 1):
        print(f"\n--- 处理碎片 {f.id} ({i}/{len(fragments)}) ---")
        try:
            # 1. 归一化
            print("→ 归一化...")
            normalized_pcd, meta = normalize_fragment(f)
            if normalized_pcd is None:
                continue
            f.point_cloud = normalized_pcd
            f.norm_metadata = meta

            # 2. 边界检测 (Robust)
            print("→ 边界检测 (Robust Dual-Threshold)...")
            detect_boundary_robust(
                f,
                smooth_iter=2,
                angle_thresh=60.0,
                low_angle_thresh=15.0,
                min_cluster_size=50,
                visualize=False,
            )
            visualize_boundary_extraction(f, mode=VIS_MODE)

            # 3. 断面 Patch (Thickness-Based)
            print("→ 提取断面 Patch (Thickness-Based)...")
            extract_section_patch(
                f,
                thickness_ratio=0.3,
                normal_to_surface_thresh=50.0,
                visualize=True,
            )
            visualize_section_patch(f, mode=VIS_MODE)

            # # 4. Rim 曲线
            # print("→ 提取 Rim 曲线...")
            # rim_curve, rim_pcd = extract_rim_curve(
            #     fragment=f,
            #     n_bins=100,
            #     normalize=True,
            #     use_arc_length=True,
            #     visualize=(VIS_MODE == VisualizationMode.INTERACTIVE and i == 1),
            # )
            #
            # if rim_curve is not None and len(rim_curve) > 0:
            #     f.rim_curve = rim_curve
            #     f.rim_pcd = rim_pcd
            #     # ✅ 修复点：用 f，而不是 fragment！
            #     visualize_rim(f, mode=VIS_MODE, always_show=False)
            # else:
            #     print(f"[Rim] 碎片{f.id} rim 提取失败或点数不足")
            #
            # # 5. 特征编码
            # print("→ 特征编码...")
            # profile, axis = extract_profile(f)
            # f.profile_curve = profile
            # f.main_axis = axis
            # encode_profile(f)

            successful_fragments.append(f)
            print(f"✓ 碎片{f.id}处理完成")

        except Exception as e:
            print(f"❌ 碎片{f.id}处理异常: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print(f"装配阶段: 有效碎片 {len(successful_fragments)} 个")
    if len(successful_fragments) >= 2:
        # 假设 coarse_match 返回 4x4 位姿列表
        poses = coarse_match(successful_fragments)
        model = assemble(successful_fragments, poses)
        if model:
            o3d.io.write_point_cloud(
                str(output_dir / "assembled.ply"), model
            )
            print("✓ 模型已保存")
            visualize_step(
                model, window_name="Assembled_Model", mode=VIS_MODE
            )


if __name__ == "__main__":
    main()
