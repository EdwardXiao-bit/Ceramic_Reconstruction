import json

# 模拟测试脚本中的逻辑来调试
def debug_create_test_pairs(matches_data, similarity_threshold=0.0):
    processed_pairs = set()
    all_candidates = []
    
    print("处理每个碎片的匹配结果:")
    for frag_id, candidates in matches_data.items():
        print(f"\n碎片 {frag_id} 的候选匹配:")
        frag1_original = int(frag_id)  # 保存原始frag_id
        
        if candidates:
            for candidate in candidates:
                frag2 = int(candidate['matched_fragment'])
                similarity = candidate['similarity']
                
                # 确保较小的ID在前
                frag1 = frag1_original  # 使用原始值而不是被修改的frag1
                if frag1 > frag2:
                    frag1, frag2 = frag2, frag1
                
                pair_key = (frag1, frag2)
                
                print(f"  匹配到碎片 {candidate['matched_fragment']} (相似度: {similarity:.4f})")
                print(f"  处理为对 ({frag1}, {frag2})")
                
                # 检查条件
                if (similarity >= similarity_threshold and 
                    frag1 != frag2 and 
                    pair_key not in processed_pairs):
                    
                    print(f"  ✓ 符合条件，添加到候选列表")
                    all_candidates.append({
                        'pair': pair_key,
                        'similarity': similarity,
                        'original_order': (frag1_original, int(candidate['matched_fragment']))
                    })
                    processed_pairs.add(pair_key)
                else:
                    if pair_key in processed_pairs:
                        print(f"  ✗ 已处理过，跳过")
                    elif similarity < similarity_threshold:
                        print(f"  ✗ 相似度低于阈值 {similarity_threshold}")
                    else:
                        print(f"  ✗ 其他原因被过滤")
    
    print(f"\n最终候选对数量: {len(all_candidates)}")
    all_candidates.sort(key=lambda x: x['similarity'], reverse=True)
    
    for i, candidate in enumerate(all_candidates):
        frag1, frag2 = candidate['pair']
        similarity = candidate['similarity']
        print(f"  {i+1}. ({frag1}, {frag2}) - 相似度: {similarity:.4f}")

# 读取匹配数据
with open('results/matching/run_20260226_151218/fragment_matches_20260226_151218.json', 'r') as f:
    matches_data = json.load(f)

debug_create_test_pairs(matches_data, 0.0)