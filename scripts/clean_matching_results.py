#!/usr/bin/env python3
"""
清理匹配结果目录中的混乱文件
将所有按时间戳命名的文件移动到对应的run文件夹中
"""

import os
import shutil
from pathlib import Path
import re
from datetime import datetime

def clean_matching_results(base_dir="results/matching"):
    """清理匹配结果目录"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"目录不存在: {base_path}")
        return
    
    # 查找所有run文件夹
    run_folders = list(base_path.glob("run_*"))
    print(f"找到 {len(run_folders)} 个run文件夹")
    
    # 查找所有独立的文件（不在run文件夹中的文件）
    standalone_files = []
    for item in base_path.iterdir():
        if item.is_file():
            standalone_files.append(item)
    
    print(f"找到 {len(standalone_files)} 个独立文件")
    
    # 按时间戳分组文件
    timestamp_groups = {}
    
    # 正则表达式匹配时间戳格式
    timestamp_pattern = re.compile(r'(\d{8}_\d{6})')
    
    for file_path in standalone_files:
        match = timestamp_pattern.search(file_path.name)
        if match:
            timestamp = match.group(1)
            if timestamp not in timestamp_groups:
                timestamp_groups[timestamp] = []
            timestamp_groups[timestamp].append(file_path)
    
    print(f"识别出 {len(timestamp_groups)} 个不同的时间戳组")
    
    # 为每个时间戳组创建或验证run文件夹
    for timestamp, files in timestamp_groups.items():
        run_folder = base_path / f"run_{timestamp}"
        
        # 如果run文件夹不存在，创建它
        if not run_folder.exists():
            run_folder.mkdir(parents=True)
            print(f"创建文件夹: {run_folder}")
        
        # 移动文件到对应文件夹
        moved_count = 0
        for file_path in files:
            target_path = run_folder / file_path.name
            if not target_path.exists():
                shutil.move(str(file_path), str(target_path))
                moved_count += 1
                print(f"  移动: {file_path.name} -> {run_folder.name}/")
            else:
                print(f"  跳过(已存在): {file_path.name}")
        
        print(f"  时间戳 {timestamp}: 移动了 {moved_count} 个文件")
    
    # 统计最终状态
    final_run_folders = list(base_path.glob("run_*"))
    remaining_files = [f for f in base_path.iterdir() if f.is_file()]
    
    print("\n" + "="*50)
    print("清理完成!")
    print(f"Run文件夹总数: {len(final_run_folders)}")
    print(f"剩余独立文件数: {len(remaining_files)}")
    
    if remaining_files:
        print("剩余文件:")
        for file_path in remaining_files:
            print(f"  - {file_path.name}")

def organize_existing_runs(base_dir="results/matching"):
    """整理已有的run文件夹，确保结构一致"""
    base_path = Path(base_dir)
    
    run_folders = list(base_path.glob("run_*"))
    
    for run_folder in run_folders:
        print(f"\n检查文件夹: {run_folder.name}")
        
        # 统计文件类型
        file_types = {}
        for file_path in run_folder.iterdir():
            if file_path.is_file():
                suffix = file_path.suffix
                if suffix not in file_types:
                    file_types[suffix] = []
                file_types[suffix].append(file_path.name)
        
        print(f"  文件类型统计:")
        for ext, files in file_types.items():
            print(f"    {ext}: {len(files)} 个文件")

if __name__ == "__main__":
    print("开始清理匹配结果目录...")
    clean_matching_results()
    print("\n检查整理后的结构...")
    organize_existing_runs()