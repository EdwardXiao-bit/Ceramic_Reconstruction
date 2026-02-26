#!/usr/bin/env python3
"""
基于真实纹理贴图的SuperGlue匹配
充分利用OBJ文件的MTL材质和JPEG纹理信息
"""
import sys
import os
from pathlib import Path
import cv2
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.texture_matching.texture_analysis import TextureExtractor
from src.texture_matching.enhanced_superglue import EnhancedTextureMatcher


def load_texture_images(data_dir: str = "data/demo") -> dict:
    """加载真实的纹理贴图"""
    extractor = TextureExtractor(data_dir)
    texture_images = {}
    
    obj_files = list(Path(data_dir).glob("*.obj"))
    print(f"=== 加载纹理贴图 ===")
    
    for obj_file in obj_files:
        print(f"处理: {obj_file.name}")
        texture_data = extractor.extract_with_materials(str(obj_file))
        
        if texture_data and texture_data['materials']:
            # 获取纹理贴图路径
            for mat_name, mat_props in texture_data['materials'].items():
                if mat_props.get('texture_map'):
                    texture_path = mat_props['texture_map']
                    try:
                        # 读取纹理图像
                        img = cv2.imread(texture_path)
                        if img is not None:
                            # 转换为RGB
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            texture_images[obj_file.stem] = img_rgb
                            print(f"  ✓ 加载纹理: {Path(texture_path).name} ({img.shape[1]}×{img.shape[0]})")
                        break  # 每个OBJ文件只需要一张纹理图
                    except Exception as e:
                        print(f"  ✗ 纹理加载失败: {e}")
    
    return texture_images


def run_texture_based_matching():
    """运行基于真实纹理的匹配"""
    print("=" * 60)
    print(" 基于真实纹理贴图的SuperGlue匹配 ")
    print("=" * 60)
    
    # 1. 加载纹理贴图
    texture_images = load_texture_images()
    
    if not texture_images:
        print("✗ 未找到有效的纹理贴图")
        return
    
    print(f"\n✓ 成功加载 {len(texture_images)} 张纹理贴图")
    
    # 2. 初始化增强匹配器
    matcher = EnhancedTextureMatcher()
    
    # 3. 提取特征并匹配
    print("\n=== 特征提取与匹配 ===")
    features = {}
    
    # 提取每张纹理的特征
    for name, texture_img in texture_images.items():
        print(f"处理纹理: {name}")
        
        if matcher.matcher is not None:
            # 使用SuperGlue提取特征
            try:
                # 调整图像大小以适应SuperGlue
                resized_img = cv2.resize(texture_img, (512, 512))
                
                # 转换为灰度图
                gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
                
                # 提取SuperGlue特征
                sg_features = matcher._extract_superglue_features(resized_img)
                if sg_features:
                    features[name] = sg_features
                    print(f"  ✓ 提取到 {len(sg_features['keypoints'])} 个SuperGlue关键点")
                else:
                    print(f"  ✗ SuperGlue特征提取失败")
                    
            except Exception as e:
                print(f"  ✗ 特征提取错误: {e}")
        else:
            print(f"  ✗ SuperGlue不可用")
    
    # 4. 计算相似度
    print("\n=== 纹理相似度计算 ===")
    texture_names = list(features.keys())
    
    for i in range(len(texture_names)):
        for j in range(i + 1, len(texture_names)):
            name1, name2 = texture_names[i], texture_names[j]
            
            # 使用传统方法计算相似度
            feat1 = features[name1]
            feat2 = features[name2]
            
            # 简单的颜色直方图比较
            img1 = texture_images[name1]
            img2 = texture_images[name2]
            
            # 转换为HSV色彩空间进行比较
            hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
            hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
            
            # 计算直方图
            hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            
            # 直方图相似度
            hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            print(f"{name1} vs {name2}: 纹理相似度 = {hist_similarity:.3f}")
    
    # 5. 保存结果
    output_dir = Path("results/texture_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存纹理图像
    for name, img in texture_images.items():
        output_path = output_dir / f"{name}_texture.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"✓ 纹理图像已保存: {output_path}")
    
    print(f"\n=== 完成 ===")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    run_texture_based_matching()