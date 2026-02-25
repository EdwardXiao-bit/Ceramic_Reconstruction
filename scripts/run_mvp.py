# D:\Users\Lenovo\Documents\GitHub\Ceramic_Reconstruction\scripts\run_mvp.py

import sys
import os
import json
from pathlib import Path
import open3d as o3d
import numpy as np
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import load_fragments
from src.preprocessing.normalize import normalize_fragment
from src.boundary import detect_boundary_robust, extract_section_patch
from src.boundary.rim import extract_centerline_rim_from_boundaries, extract_rim_curve
from src.profile.extract import extract_profile
from src.features.profile_feat import encode_profile
from src.matching.coarse import coarse_match
from src.assembly.graph import assemble

# 添加多模态特征提取模块
from src.geometry_features.traditional_feat import compute_patch_fpfh
from src.geometry_features.patch_encoder import PatchEncoder
from src.texture_matching.texture_analysis import TextureExtractor, EnhancedTextureMatcher
from src.texture_matching.superglue_features import generate_superglue_embedding, extract_3d_superglue_features

# 添加匹配初筛模块
from src.matching.coarse import coarse_match
from src.matching.faiss_prescreen import faiss_prescreen
from src.matching.results_saver import save_matching_results

import json
import pickle


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

    # 灰色主体
    vis_base = o3d.geometry.PointCloud(fragment.point_cloud)
    vis_base.paint_uniform_color([0.8, 0.8, 0.8])

    # rim 点云，偏移避免 z-fighting
    rim_pcd = o3d.geometry.PointCloud()
    rim_pcd.points = o3d.utility.Vector3dVector(rim_curve)
    rim_pcd.paint_uniform_color([0, 1, 0])
    rim_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    bbox = vis_base.get_axis_aligned_bounding_box()
    offset = np.linalg.norm(bbox.get_extent()) * 0.005
    rim_pcd.points = o3d.utility.Vector3dVector(
        np.asarray(rim_pcd.points) + np.asarray(rim_pcd.normals) * offset
    )

    # rim 连线
    lines = [[i, (i + 1) % len(rim_curve)] for i in range(len(rim_curve))]
    rim_lines = o3d.geometry.LineSet()
    rim_lines.points = rim_pcd.points  # 用偏移后的点
    rim_lines.lines = o3d.utility.Vector2iVector(lines)
    rim_lines.paint_uniform_color([0, 1, 0])

    if always_show or mode == VisualizationMode.INTERACTIVE:
        visualize_step(
            [vis_base, rim_pcd, rim_lines],
            window_name=f"Rim_{fragment.id}",
            mode=mode
        )
        visualize_step(
            [rim_pcd, rim_lines],
            window_name=f"Rim_Only_{fragment.id}",
            mode=mode
        )


def save_features(fragments, output_dir):
    """保存提取的特征到文件"""
    features_data = {}
    
    for i, frag in enumerate(fragments):
        frag_data = {
            'id': frag.id,
            'file_name': getattr(frag, 'file_name', f'fragment_{frag.id}')
        }
        
        # 保存各种特征
        if hasattr(frag, 'profile_feature') and frag.profile_feature is not None:
            frag_data['profile_feature'] = frag.profile_feature.tolist()
        
        if hasattr(frag, 'fpfh_feature') and frag.fpfh_feature is not None:
            frag_data['fpfh_feature'] = frag.fpfh_feature.tolist()
        
        if hasattr(frag, 'geo_embedding') and frag.geo_embedding is not None:
            frag_data['geo_embedding'] = frag.geo_embedding.tolist()
        
        if hasattr(frag, 'texture_embedding') and frag.texture_embedding is not None:
            frag_data['texture_embedding'] = frag.texture_embedding.tolist()
        
        features_data[f'fragment_{frag.id}'] = frag_data
    
    # 保存JSON格式
    json_path = output_dir / "extracted_features.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(features_data, f, indent=2, ensure_ascii=False)
    
    # 保存pickle格式（保留numpy数组）
    pickle_path = output_dir / "extracted_features.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(features_data, f)
    
    print(f"特征已保存:")
    print(f"  JSON格式: {json_path}")
    print(f"  Pickle格式: {pickle_path}")


def save_boundary_data(fragments, output_dir):
    """保存边界相关的数据到文件"""
    boundary_data = {}
    
    for frag in fragments:
        frag_boundary_data = {
            'id': frag.id,
            'file_name': getattr(frag, 'file_name', f'fragment_{frag.id}')
        }
        
        # 保存边界点数据
        if hasattr(frag, 'boundary_pts') and frag.boundary_pts is not None:
            frag_boundary_data['boundary_points'] = frag.boundary_pts.tolist()
            print(f"  ✓ 碎片{frag.id}边界点: {len(frag.boundary_pts)}个")
        
        # 保存断面patch数据
        if hasattr(frag, 'section_patch') and frag.section_patch is not None:
            # 转换点云为numpy数组
            section_points = np.asarray(frag.section_patch.points)
            frag_boundary_data['section_patch_points'] = section_points.tolist()
            print(f"  ✓ 碎片{frag.id}断面patch: {len(section_points)}个点")
        
        # 保存Rim曲线数据
        if hasattr(frag, 'rim_curve') and frag.rim_curve is not None:
            frag_boundary_data['rim_curve'] = frag.rim_curve.tolist()
            print(f"  ✓ 碎片{frag.id}Rim曲线: {len(frag.rim_curve)}个点")
        
        # 保存主轴数据
        if hasattr(frag, 'main_axis') and frag.main_axis is not None:
            frag_boundary_data['main_axis'] = frag.main_axis.tolist()
        
        # 保存轮廓曲线数据
        if hasattr(frag, 'profile_curve') and frag.profile_curve is not None:
            frag_boundary_data['profile_curve'] = frag.profile_curve.tolist()
        
        boundary_data[f'fragment_{frag.id}'] = frag_boundary_data
    
    # 保存边界数据
    boundary_json_path = output_dir / "boundary_data.json"
    with open(boundary_json_path, 'w', encoding='utf-8') as f:
        json.dump(boundary_data, f, indent=2, ensure_ascii=False)
    
    # 保存为pickle格式（保留numpy数组）
    boundary_pkl_path = output_dir / "boundary_data.pkl"
    with open(boundary_pkl_path, 'wb') as f:
        pickle.dump(boundary_data, f)
    
    print(f"\n边界数据已保存:")
    print(f"  JSON格式: {boundary_json_path}")
    print(f"  Pickle格式: {boundary_pkl_path}")
    
    return boundary_data


def visualize_features(fragments, output_dir):
    """可视化特征分布"""
    try:
        import matplotlib.pyplot as plt
        
        # 准备数据
        profile_feats = []
        fpfh_feats = []
        geo_feats = []
        texture_feats = []
        
        for frag in fragments:
            if hasattr(frag, 'profile_feature') and frag.profile_feature is not None:
                profile_feats.append(frag.profile_feature)
            if hasattr(frag, 'fpfh_feature') and frag.fpfh_feature is not None:
                fpfh_feats.append(frag.fpfh_feature)
            if hasattr(frag, 'geo_embedding') and frag.geo_embedding is not None:
                geo_feats.append(frag.geo_embedding)
            if hasattr(frag, 'texture_embedding') and frag.texture_embedding is not None:
                texture_feats.append(frag.texture_embedding)
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建可视化图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Multimodal Feature Distribution Visualization', fontsize=16)
        
        # 轮廓特征可视化
        if profile_feats:
            profile_feats = np.array(profile_feats)
            axes[0,0].boxplot(profile_feats)
            axes[0,0].set_title('Profile Features')
            axes[0,0].set_xlabel('Feature Dimensions')
            axes[0,0].set_ylabel('Feature Values')
        
        # FPFH特征可视化
        if fpfh_feats:
            fpfh_feats = np.array(fpfh_feats)
            axes[0,1].boxplot(fpfh_feats[:, ::3])  # 每3个点采样一个避免过多箱线
            axes[0,1].set_title('FPFH Features')
            axes[0,1].set_xlabel('Feature Dimensions')
            axes[0,1].set_ylabel('Feature Values')
        
        # 深度几何特征可视化
        if geo_feats:
            geo_feats = np.array(geo_feats)
            axes[1,0].boxplot(geo_feats[:, ::10])  # 每10个点采样一个
            axes[1,0].set_title('Deep Geometric Features')
            axes[1,0].set_xlabel('Feature Dimensions')
            axes[1,0].set_ylabel('Feature Values')
        
        # 纹理特征可视化
        if texture_feats:
            texture_feats = np.array(texture_feats)
            axes[1,1].boxplot(texture_feats)
            axes[1,1].set_title('Texture Features')
            axes[1,1].set_xlabel('Feature Dimensions')
            axes[1,1].set_ylabel('Feature Values')
        
        plt.tight_layout()
        plt.savefig(output_dir / "feature_distributions.png", dpi=300, bbox_inches='tight')
        print(f"特征分布图已保存: {output_dir / 'feature_distributions.png'}")
        plt.close()
        
    except ImportError:
        print("matplotlib未安装，跳过特征可视化")
    except Exception as e:
        print(f"特征可视化失败: {e}")


def visualize_rim(fragment, mode=VIS_MODE, always_show=False):
    """可视化rim曲线，使用同学修改的版本"""
    if mode == VisualizationMode.NONE and not always_show:
        return
    
    if not (hasattr(fragment, 'rim_curve') and fragment.rim_curve is not None):
        return
    
    rim_curve = fragment.rim_curve
    if len(rim_curve) < 10:
        return
    
    # 灰色主体
    vis_base = o3d.geometry.PointCloud(fragment.point_cloud)
    vis_base.paint_uniform_color([0.8, 0.8, 0.8])
    
    # rim 点云，偏移避免 z-fighting
    rim_pcd = o3d.geometry.PointCloud()
    rim_pcd.points = o3d.utility.Vector3dVector(rim_curve)
    rim_pcd.paint_uniform_color([0, 1, 0])  # 纯绿色
    rim_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    bbox = vis_base.get_axis_aligned_bounding_box()
    offset = np.linalg.norm(bbox.get_extent()) * 0.005
    rim_pcd.points = o3d.utility.Vector3dVector(
        np.asarray(rim_pcd.points) + np.asarray(rim_pcd.normals) * offset
    )
    
    # rim 连线
    lines = [[i, (i + 1) % len(rim_curve)] for i in range(len(rim_curve))]
    rim_lines = o3d.geometry.LineSet()
    rim_lines.points = rim_pcd.points  # 用偏移后的点
    rim_lines.lines = o3d.utility.Vector2iVector(lines)
    rim_lines.paint_uniform_color([0, 1, 0])  # 纯绿色
    
    # 完整显示（主体+rim）
    if always_show or mode == VisualizationMode.INTERACTIVE:
        visualize_step(
            [vis_base, rim_pcd, rim_lines],
            window_name=f"Rim_{fragment.id}",
            mode=mode
        )
        # 仅rim显示
        visualize_step(
            [rim_pcd, rim_lines],
            window_name=f"Rim_Only_{fragment.id}",
            mode=mode
        )


def main():
    data_dir = PROJECT_ROOT / "data" / "eg1"
    
    # 为每次运行创建带时间戳的独立输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "data" / "output" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"输出目录: {output_dir}")

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

            # 4. Rim 曲线（使用新的中心线提取方法）
            print("→ 提取中心线 Rim 曲线...")
            rim_curve, rim_pcd = extract_centerline_rim_from_boundaries(
                fragment=f,
                n_samples=100,
                visualize=(VIS_MODE == VisualizationMode.INTERACTIVE and i == 1),
            )

            if rim_curve is not None and len(rim_curve) > 0:
                f.rim_curve = rim_curve
                f.rim_pcd = rim_pcd
                # ✅ 修复点：用 f，而不是 fragment！
                visualize_rim(f, mode=VIS_MODE, always_show=False)
            else:
                print(f"[Rim] 碎片{f.id} 中心线rim提取失败或点数不足")
                # 降级到传统方法
                print("→ 降级到传统Rim提取方法...")
                rim_curve, rim_pcd = extract_rim_curve(
                    fragment=f,
                    n_samples=100,
                    visualize=False,
                )
                if rim_curve is not None and len(rim_curve) > 0:
                    f.rim_curve = rim_curve
                    f.rim_pcd = rim_pcd
                    visualize_rim(f, mode=VIS_MODE, always_show=False)
                    print(f"[Rim] 碎片{f.id} 使用传统方法提取rim成功")
                else:
                    print(f"[Rim] 碎片{f.id} 传统方法也失败")

            # 5. 多模态特征提取
            print("→ 多模态特征提取...")
            
            # 5.1 轮廓特征
            profile, axis = extract_profile(f)
            f.profile_curve = profile
            f.main_axis = axis
            encode_profile(f)
            
            # 5.2 FPFH几何特征
            print("  → 提取FPFH特征...")
            if hasattr(f, 'section_patch') and f.section_patch is not None:
                fpfh_feature = compute_patch_fpfh(f.section_patch, knn=20)
                f.fpfh_feature = fpfh_feature
                if fpfh_feature is not None:
                    print(f"    ✓ FPFH特征提取成功: {len(fpfh_feature)}维")
                else:
                    print(f"    ✗ FPFH特征提取失败")
            else:
                f.fpfh_feature = None
                print(f"    ✗ 无断面patch，跳过FPFH特征提取")
            
            # 5.3 深度几何特征（PointNet）
            print("  → 提取深度几何特征...")
            if hasattr(f, 'section_patch') and f.section_patch is not None:
                try:
                    encoder = PatchEncoder()
                    geo_embedding = encoder.encode(f.section_patch)
                    f.geo_embedding = geo_embedding
                    print(f"    ✓ 深度几何特征提取成功: {len(geo_embedding)}维")
                except Exception as e:
                    print(f"    ✗ 深度几何特征提取失败: {e}")
                    f.geo_embedding = None
            else:
                f.geo_embedding = None
                print(f"    ✗ 无断面patch，跳过深度几何特征提取")
            
            # 5.4 纹理特征（集成SuperGlue）
            print("  → 提取SuperGlue纹理特征...")
            try:
                # 获取原始OBJ文件路径
                obj_file_path = str(data_dir / f"{f.id + 2}.obj")  # 假设文件名为2.obj, 3.obj等
                
                # 使用增强纹理匹配器（包含SuperGlue）
                texture_matcher = EnhancedTextureMatcher()
                enhanced_features = texture_matcher.extract_enhanced_features(obj_file_path)
                
                if enhanced_features:
                    texture_data = enhanced_features.get('texture_data', {})
                    
                    # 提取多种纹理特征
                    texture_features = []
                    
                    # 1. 基础颜色统计特征
                    if 'vertex_colors' in texture_data and texture_data['vertex_colors'] is not None:
                        vertex_colors = texture_data['vertex_colors']
                        color_mean = np.mean(vertex_colors, axis=0)
                        color_std = np.std(vertex_colors, axis=0)
                        color_features = np.concatenate([color_mean, color_std])
                        texture_features.append(color_features)
                        print(f"    ✓ 颜色统计特征: {len(color_features)}维")
                    
                    # 2. 材质特征
                    if 'materials' in texture_data and texture_data['materials']:
                        materials = texture_data['materials']
                        # 提取材质属性作为特征
                        material_features = []
                        for mat_name, mat_props in materials.items():
                            # 漫反射颜色
                            diffuse = np.array(mat_props.get('diffuse_color', [0.8, 0.8, 0.8]))
                            # 环境光颜色
                            ambient = np.array(mat_props.get('ambient_color', [0.2, 0.2, 0.2]))
                            # 镜面反射颜色
                            specular = np.array(mat_props.get('specular_color', [1.0, 1.0, 1.0]))
                            
                            material_feature = np.concatenate([diffuse, ambient, specular])
                            material_features.append(material_feature)
                        
                        if material_features:
                            # 平均所有材质特征
                            avg_material_features = np.mean(material_features, axis=0)
                            texture_features.append(avg_material_features)
                            print(f"    ✓ 材质特征: {len(avg_material_features)}维")
                    
                    # 3. 真正的SuperGlue特征
                    print(f"    → 生成SuperGlue关键点特征...")
                    try:
                        # 从3D模型直接生成SuperGlue特征
                        superglue_embedding = generate_superglue_embedding(
                            f, 
                            resolution=(512, 512),
                            max_keypoints=512
                        )
                        
                        if superglue_embedding is not None:
                            texture_features.append(superglue_embedding)
                            print(f"    ✓ SuperGlue特征: {len(superglue_embedding)}维")
                        else:
                            # 降级到基础特征
                            superglue_basic = np.zeros(256)  # 256维零向量作为占位符
                            texture_features.append(superglue_basic)
                            print(f"    ↓ SuperGlue特征生成失败，使用占位符: 256维")
                    except Exception as sg_error:
                        print(f"    ✗ SuperGlue特征生成异常: {sg_error}")
                        # 仍然添加占位符以保持维度一致性
                        superglue_basic = np.zeros(256)
                        texture_features.append(superglue_basic)
                    
                    # 合并所有纹理特征
                    if texture_features:
                        combined_texture_embedding = np.concatenate(texture_features)
                        f.texture_embedding = combined_texture_embedding.astype(np.float32)
                        feature_types = ["颜色统计", "材质属性", "SuperGlue"]
                        print(f"    ✓ 综合纹理特征提取成功: {len(combined_texture_embedding)}维")
                        print(f"      包含: {', '.join(feature_types[:len(texture_features)])}特征")
                    else:
                        f.texture_embedding = None
                        print(f"    ✗ 无有效纹理特征")
                else:
                    f.texture_embedding = None
                    print(f"    ✗ 纹理特征提取失败")
                    
            except Exception as e:
                print(f"    ✗ SuperGlue纹理特征提取异常: {e}")
                # 降级到基础纹理特征
                try:
                    obj_file_path = str(data_dir / f"{f.id + 2}.obj")
                    texture_extractor = TextureExtractor(str(data_dir))
                    texture_data = texture_extractor.extract_with_materials(obj_file_path)
                    
                    if texture_data and 'vertex_colors' in texture_data and texture_data['vertex_colors'] is not None:
                        vertex_colors = texture_data['vertex_colors']
                        color_mean = np.mean(vertex_colors, axis=0)
                        color_std = np.std(vertex_colors, axis=0)
                        texture_embedding = np.concatenate([color_mean, color_std])
                        f.texture_embedding = texture_embedding.astype(np.float32)
                        print(f"    ↓ 降级到基础纹理特征: {len(texture_embedding)}维")
                    else:
                        f.texture_embedding = None
                        print(f"    ✗ 基础纹理特征也失败")
                except Exception as fallback_e:
                    print(f"    ✗ 纹理特征完全失败: {fallback_e}")
                    f.texture_embedding = None

            successful_fragments.append(f)
            print(f"✓ 碎片{f.id}处理完成")

        except Exception as e:
            print(f"❌ 碎片{f.id}处理异常: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print(f"特征提取学习阶段完成: 成功处理 {len(successful_fragments)} 个碎片")
    
    # 多模态特征统计
    print("\n特征提取统计:")
    profile_count = sum(1 for f in successful_fragments if hasattr(f, 'profile_feature') and f.profile_feature is not None)
    fpfh_count = sum(1 for f in successful_fragments if hasattr(f, 'fpfh_feature') and f.fpfh_feature is not None)
    geo_count = sum(1 for f in successful_fragments if hasattr(f, 'geo_embedding') and f.geo_embedding is not None)
    texture_count = sum(1 for f in successful_fragments if hasattr(f, 'texture_embedding') and f.texture_embedding is not None)
    
    print(f"  轮廓特征: {profile_count}/{len(successful_fragments)}")
    print(f"  FPFH特征: {fpfh_count}/{len(successful_fragments)}")
    print(f"  深度几何特征: {geo_count}/{len(successful_fragments)}")
    print(f"  纹理特征: {texture_count}/{len(successful_fragments)}")
    
    # 保存特征到文件供后续分析
    save_features(successful_fragments, output_dir)
    print(f"\n✓ 特征已保存至: {output_dir}")
    
    # 保存边界数据到文件
    save_boundary_data(successful_fragments, output_dir)
    print(f"\n✓ 边界数据已保存至: {output_dir}")
    
    # 可视化特征分布
    if len(successful_fragments) >= 2:
        visualize_features(successful_fragments, output_dir)
    
    # ============================================
    # 匹配初筛阶段
    # ============================================
    print("\n" + "=" * 60)
    print("开始匹配初筛阶段...")
    print("=" * 60)
    
    # 1. 基于轮廓特征的粗匹配
    print("\n1. 执行轮廓特征粗匹配...")
    coarse_matches = coarse_match(successful_fragments)
    print(f"   ✓ 粗匹配完成，获得 {len(coarse_matches)} 个候选对")
    
    if coarse_matches:
        print("   前5个匹配对:")
        for i, (id1, id2, score) in enumerate(coarse_matches[:5]):
            frag1_name = getattr(successful_fragments[id1], 'file_name', f'fragment_{id1}')
            frag2_name = getattr(successful_fragments[id2], 'file_name', f'fragment_{id2}')
            print(f"     {i+1}. {frag1_name} ↔ {frag2_name} (相似度: {score:.4f})")
    
    # 2. FAISS多模态聚类初筛
    print("\n2. 执行FAISS多模态聚类初筛...")
    try:
        # 使用FAISS进行高效的多模态匹配
        faiss_matches, process_info = faiss_prescreen(
            fragments=successful_fragments,
            top_m_geo=30,      # 几何特征候选数
            top_m_fpfh=30,     # FPFH特征候选数  
            top_m_texture=20,  # 纹理特征候选数
            top_k=15,          # 每个碎片保留的候选对数
            alpha=0.4,         # 几何特征权重
            beta=0.3,          # FPFH特征权重
            gamma=0.3,         # 纹理特征权重
            s_min=0.1          # 最低相似度阈值
        )
        
        print(f"   ✓ FAISS初筛完成，获得 {len(faiss_matches)} 个高质量候选对")
        
        if faiss_matches:
            print("   前10个高质量匹配对:")
            for i, (id1, id2, score) in enumerate(faiss_matches[:10]):
                frag1_name = getattr(successful_fragments[id1], 'file_name', f'fragment_{id1}')
                frag2_name = getattr(successful_fragments[id2], 'file_name', f'fragment_{id2}')
                print(f"     {i+1:2d}. {frag1_name[:15]:<15} ↔ {frag2_name[:15]:<15} (综合得分: {score:.4f})")
        
        # 3. 保存匹配结果
        print("\n3. 保存匹配结果...")
        match_output_dir = PROJECT_ROOT / "results" / "matching"
        timestamp = save_matching_results(
            matches=faiss_matches, 
            fragments=successful_fragments,
            output_dir=str(match_output_dir),
            detailed_info=process_info,
            create_run_folder=True  # 为每次运行创建独立文件夹
        )
        print(f"   ✓ 匹配结果已保存到独立文件夹: run_{timestamp}")
        
    except Exception as e:
        print(f"   ✗ FAISS初筛失败: {e}")
        print("   将使用基础轮廓匹配结果作为替代")
        faiss_matches = coarse_matches
        process_info = {
            'method': 'coarse_only',
            'fallback_reason': str(e)
        }
    
    # 4. 匹配结果统计
    print("\n4. 匹配结果统计:")
    print(f"   总候选对数: {len(faiss_matches)}")
    if faiss_matches:
        scores = [match[2] for match in faiss_matches]
        print(f"   平均相似度: {np.mean(scores):.4f}")
        print(f"   最高相似度: {np.max(scores):.4f}")
        print(f"   最低相似度: {np.min(scores):.4f}")
    
    print("\n" + "=" * 60)
    print("匹配初筛阶段完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
