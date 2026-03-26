"""
3DMatch 数据集自动下载脚本
"""
import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import shutil

def download_3dmatch():
    """下载 3DMatch 完整数据集"""
    
    print("\n" + "="*60)
    print("3DMatch 数据集下载")
    print("="*60)
    
    # 创建数据目录
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / '3dmatch-dataset.zip'
    extract_dir = data_dir / '3dmatch'
    
    # 检查是否已存在
    if output_path.exists():
        print(f"✓ 压缩包已存在：{output_path}")
        size_gb = output_path.stat().st_size / (1024**3)
        print(f"  大小：{size_gb:.2f} GB")
        
        if extract_dir.exists():
            print(f"✓ 数据已解压：{extract_dir}")
            return True
    
    # 多个下载源尝试
    download_urls = [
        'http://3dvision.princeton.edu/projects/2016/3DMatch/dataset/3dmatch-dataset.zip',
        'https://github.com/andyzeng/3dmatch-toolbox/raw/master/data/3dmatch-dataset.zip',
        'https://zenodo.org/record/578693/files/3dmatch-dataset.zip',  # Zenodo 备用
    ]
    
    print(f"\n开始下载:")
    print(f"尝试 {len(download_urls)} 个下载源")
    print(f"目标：{output_path}")
    print(f"大小：约 2.5 GB")
    print("\n这可能需要几分钟...")
    
    try:
        # 尝试多个下载源
        for download_url in download_urls:
            print(f"\n尝试从：{download_url}")
            try:
                # 使用 urllib 下载（带进度显示）
                def report_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(downloaded * 100.0 / total_size, 100.0)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r进度：{percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
                
                urllib.request.urlretrieve(download_url, output_path, reporthook=report_progress)
                
                print(f"\n✓ 下载完成！")
                size_gb = output_path.stat().st_size / (1024**3)
                print(f"  文件大小：{size_gb:.2f} GB")
                break  # 成功后退出循环
            except Exception as e:
                print(f"❌ 失败：{e}")
                if download_url == download_urls[-1]:
                    raise  # 如果是最后一个 URL，则抛出异常
                continue
        
        # 解压
        print(f"\n正在解压到 {extract_dir} ...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print(f"✓ 解压完成！")
        
        # 重命名目录
        extracted_folder = data_dir / '3dmatch-dataset'
        if extracted_folder.exists() and not extract_dir.exists():
            extracted_folder.rename(extract_dir)
            print(f"✓ 目录已重命名为：{extract_dir}")
        
        # 验证结构
        print(f"\n验证数据集结构...")
        if (extract_dir / 'train').exists():
            print(f"✓ train/ 目录存在")
        if (extract_dir / 'val').exists():
            print(f"✓ val/ 目录存在")
        if (extract_dir / 'test').exists():
            print(f"✓ test/ 目录存在")
        
        print(f"\n✅ 3DMatch 数据集准备完成！")
        print(f"位置：{extract_dir.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败：{e}")
        print(f"\n建议:")
        print(f"1. 检查网络连接")
        print(f"2. 手动下载：{download_url}")
        print(f"3. 保存到：{output_path}")
        return False


if __name__ == '__main__':
    success = download_3dmatch()
    sys.exit(0 if success else 1)
