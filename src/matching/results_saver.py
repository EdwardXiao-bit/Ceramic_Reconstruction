"""
匹配结果保存模块
提供详细的匹配过程记录和结果保存功能
"""
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path


class MatchResultsSaver:
    """匹配结果保存器"""
    
    def __init__(self, output_dir="results/matching", create_run_folder=True):
        self.base_output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 为每次运行创建独立的文件夹
        if create_run_folder:
            self.output_dir = self.base_output_dir / f"run_{self.timestamp}"
        else:
            self.output_dir = self.base_output_dir
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_detailed_matches(self, matches, fragments, detailed_info=None):
        """
        保存详细的匹配结果
        :param matches: [(frag1_id, frag2_id, similarity), ...]
        :param fragments: Fragment列表
        :param detailed_info: 详细的匹配过程信息
        """
        # 1. 保存匹配对详情
        self._save_match_pairs(matches, fragments)
        
        # 2. 保存每个碎片的匹配列表
        self._save_fragment_matches(matches, fragments)
        
        # 3. 保存详细过程信息
        if detailed_info:
            self._save_process_details(detailed_info)
        
        # 4. 生成汇总报告
        self._generate_summary_report(matches, fragments, detailed_info)
        
        print(f"[结果保存] 匹配结果已保存至: {self.output_dir}")
        print(f"[结果保存] 本次运行标识: run_{self.timestamp}")
    
    def _save_match_pairs(self, matches, fragments):
        """保存匹配对详情"""
        match_file = self.output_dir / f"match_pairs_{self.timestamp}.txt"
        
        with open(match_file, 'w', encoding='utf-8') as f:
            f.write("FAISS初筛匹配对详情\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总匹配对数: {len(matches)}\n\n")
            
            # 按相似度排序
            sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)
            
            for i, (id1, id2, similarity) in enumerate(sorted_matches, 1):
                frag1_name = getattr(fragments[id1], 'file_name', f'fragment_{id1}')
                frag2_name = getattr(fragments[id2], 'file_name', f'fragment_{id2}')
                
                f.write(f"{i:2d}. {frag1_name[:20]:<20} ↔ {frag2_name[:20]:<20} "
                       f"(相似度: {similarity:.4f})\n")
        
        print(f"✓ 匹配对详情已保存: {match_file}")
    
    def _save_fragment_matches(self, matches, fragments):
        """保存每个碎片的匹配列表"""
        # 构建每个碎片的匹配字典
        fragment_matches = {}
        for frag in fragments:
            fragment_matches[frag.id] = []
        
        # 填充匹配信息
        for id1, id2, similarity in matches:
            fragment_matches[id1].append({
                'matched_fragment': id2,
                'similarity': float(similarity),
                'fragment_name': getattr(fragments[id2], 'file_name', f'fragment_{id2}')
            })
            fragment_matches[id2].append({
                'matched_fragment': id1,
                'similarity': float(similarity),
                'fragment_name': getattr(fragments[id1], 'file_name', f'fragment_{id1}')
            })
        
        # 保存JSON格式
        json_file = self.output_dir / f"fragment_matches_{self.timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(fragment_matches, f, indent=2, ensure_ascii=False)
        
        # 保存人类可读格式
        txt_file = self.output_dir / f"fragment_matches_{self.timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("各碎片匹配列表\n")
            f.write("=" * 50 + "\n\n")
            
            for frag_id, matches_list in fragment_matches.items():
                if matches_list:  # 只显示有匹配的碎片
                    frag_name = getattr(fragments[frag_id], 'file_name', f'fragment_{frag_id}')
                    f.write(f"碎片 {frag_id} ({frag_name}):\n")
                    
                    # 按相似度排序
                    sorted_matches = sorted(matches_list, key=lambda x: x['similarity'], reverse=True)
                    for match_info in sorted_matches:
                        f.write(f"  → 与碎片 {match_info['matched_fragment']} "
                               f"({match_info['fragment_name']}) "
                               f"相似度: {match_info['similarity']:.4f}\n")
                    f.write("\n")
        
        print(f"✓ 碎片匹配列表已保存: {json_file}")
        print(f"✓ 人类可读格式已保存: {txt_file}")
    
    def _save_process_details(self, detailed_info):
        """保存详细过程信息"""
        if not detailed_info:
            return
            
        detail_file = self.output_dir / f"process_details_{self.timestamp}.json"
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_info, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✓ 过程详情已保存: {detail_file}")
    
    def _generate_summary_report(self, matches, fragments, detailed_info):
        """生成汇总报告"""
        report_file = self.output_dir / f"matching_report_{self.timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# FAISS初筛匹配报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基本统计
            f.write("## 基本统计\n\n")
            f.write(f"- 总碎片数: {len(fragments)}\n")
            f.write(f"- 有效匹配对数: {len(matches)}\n")
            
            if matches:
                similarities = [m[2] for m in matches]
                f.write(f"- 平均相似度: {np.mean(similarities):.4f}\n")
                f.write(f"- 最高相似度: {np.max(similarities):.4f}\n")
                f.write(f"- 最低相似度: {np.min(similarities):.4f}\n")
            
            f.write("\n")
            
            # 匹配分布
            if matches:
                f.write("## 匹配分布\n\n")
                similarities = [m[2] for m in matches]
                hist, bins = np.histogram(similarities, bins=10)
                
                f.write("| 相似度区间 | 匹配对数 |\n")
                f.write("|-----------|----------|\n")
                for i in range(len(hist)):
                    f.write(f"| {bins[i]:.3f} ~ {bins[i+1]:.3f} | {hist[i]} |\n")
                f.write("\n")
            
            # 详细匹配对
            if matches:
                f.write("## 详细匹配对\n\n")
                sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)
                
                f.write("| 排名 | 碎片1 | 碎片2 | 相似度 |\n")
                f.write("|------|-------|-------|--------|\n")
                for i, (id1, id2, similarity) in enumerate(sorted_matches[:20], 1):  # 前20个
                    frag1_name = getattr(fragments[id1], 'file_name', f'fragment_{id1}')[:15]
                    frag2_name = getattr(fragments[id2], 'file_name', f'fragment_{id2}')[:15]
                    f.write(f"| {i} | {frag1_name} | {frag2_name} | {similarity:.4f} |\n")
                
                if len(sorted_matches) > 20:
                    f.write(f"\n... 还有 {len(sorted_matches) - 20} 个匹配对\n")
            
            # 过程信息
            if detailed_info:
                f.write("\n## 过程信息\n\n")
                for key, value in detailed_info.items():
                    f.write(f"**{key}**: {value}\n")
        
        print(f"✓ 汇总报告已保存: {report_file}")


def save_matching_results(matches, fragments, output_dir="results/matching", 
                         detailed_info=None, create_run_folder=True):
    """
    便捷函数：保存匹配结果
    :param matches: 匹配对列表
    :param fragments: 碎片列表
    :param output_dir: 输出目录
    :param detailed_info: 详细信息字典
    :param create_run_folder: 是否为每次运行创建独立文件夹
    """
    saver = MatchResultsSaver(output_dir, create_run_folder=create_run_folder)
    timestamp = saver.timestamp  # 获取本次运行的时间戳
    saver.save_detailed_matches(matches, fragments, detailed_info)
    return timestamp  # 返回时间戳以便外部使用