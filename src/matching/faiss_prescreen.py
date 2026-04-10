"""
碎片匹配初筛：FAISS + 多模态相似度
文档(四)：全局相似度检索、多模态相似度计算、综合打分 Top-K
"""
import numpy as np

# FAISS 可选，失败时回退到 sklearn
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    from sklearn.neighbors import NearestNeighbors


def _build_index(embeddings, ids, metric="cosine"):
    """
    构建 ANN 索引
    :param embeddings: (N, D) 特征矩阵
    :param ids: 碎片 id 列表，与 embeddings 对应
    :param metric: "cosine" 或 "l2"
    :return: index 对象, ids
    """
    X = np.asarray(embeddings, dtype=np.float32)
    if metric == "cosine":
        # 余弦：L2 归一化后等价于内积
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        X = X / norms

    if HAS_FAISS:
        dim = X.shape[1]
        if metric == "cosine":
            index = faiss.IndexFlatIP(dim)  # 内积
        else:
            index = faiss.IndexFlatL2(dim)
        index.add(X)
        return index, ids
    else:
        nn = NearestNeighbors(n_neighbors=len(ids), metric=metric if metric == "l2" else "cosine", algorithm="brute")
        nn.fit(X)
        return nn, ids


def _search_index(index, query_emb, top_k, ids, use_faiss=True, metric="cosine"):
    """
    检索 Top-K 最近邻
    :return: list of (fragment_id, score) for each query row
    """
    Q = np.asarray(query_emb, dtype=np.float32)
    if metric == "cosine":
        norms = np.linalg.norm(Q, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        Q = Q / norms

    # 首先处理元组情况：(index_object, ids_array)
    actual_index = index
    if isinstance(index, tuple) and len(index) == 2:
        actual_index, index_ids = index
        # 使用索引中存储的 ids 而不是传入的 ids
        ids = index_ids

    # 最可靠的判断方式：检查对象的实际类型
    if HAS_FAISS:
        # 检查是否为 FAISS 索引类型
        import faiss
        if isinstance(actual_index, faiss.Index):
            # FAISS 索引对象
            k = min(top_k + 1, len(ids))
            scores, indices = actual_index.search(Q, k)
            results = []
            for i in range(len(Q)):
                row = []
                for j, idx in enumerate(indices[i]):
                    if idx >= 0 and idx != i:  # 排除自身
                        row.append((ids[idx], float(scores[i][j])))
                    if len(row) >= top_k:
                        break
                results.append(row)
            return results
    
    # 如果不是 FAISS 索引，则使用 sklearn 模式
    # sklearn 模式：需要安全的解包
    try:
        # 对于 sklearn 模式，actual_index 应该就是 NearestNeighbors 对象
        if not hasattr(actual_index, 'kneighbors'):
            raise AttributeError(f"Object {type(actual_index)} missing 'kneighbors' method")
            
        k = min(top_k + 1, len(ids))
        dists, indices = actual_index.kneighbors(Q, n_neighbors=k)
        results = []
        for i in range(len(Q)):
            row = []
            for j, idx in enumerate(indices[i]):
                if idx != i:  # 排除自身
                    fid = ids[idx]
                    # sklearn cosine 返回的是距离，1 - dist = 相似度
                    s = 1.0 - dists[i][j] if metric == "cosine" else -dists[i][j]
                    row.append((fid, float(s)))
                if len(row) >= top_k:
                    break
            results.append(row)
        return results
    except (ValueError, AttributeError) as e:
        # 如果方法不存在，抛出更清晰的错误
        raise RuntimeError(f"Failed to search index: {e}. Index type: {type(actual_index)}, HAS_FAISS: {HAS_FAISS}")


def faiss_prescreen(
    fragments,
    top_m_geo=50,
    top_m_fpfh=50,
    top_m_texture=30,
    top_k=10,
    alpha=0.5,
    beta=0.2,
    gamma=0.3,
    s_min=0.0,
):
    """
    碎片匹配初筛：FAISS 检索 + 多模态相似度 + Top-K
    支持几何特征、FPFH特征和纹样特征的融合匹配
    
    :param fragments: list[Fragment]
    :param top_m_geo: 几何 embedding 检索候选数
    :param top_m_fpfh: FPFH 检索候选数
    :param top_m_texture: 纹样特征检索候选数
    :param top_k: 每个碎片保留的 Top-K 候选对
    :param alpha: S_geo 权重
    :param beta: S_fpfh 权重
    :param gamma: S_texture 权重（alpha + beta + gamma = 1）
    :param s_min: 最低相似度阈值
    :return: tuple (matches_list, process_info)
             matches_list: list[tuple] (frag1_id, frag2_id, s_total)
             process_info: dict 包含详细的过程信息
    """
    # 检查可用的特征类型
    valid_geo = [f for f in fragments if hasattr(f, "geo_embedding") and f.geo_embedding is not None]
    valid_texture = [f for f in fragments if hasattr(f, "texture_embedding") and f.texture_embedding is not None]
    
    # 至少需要一种特征类型
    if len(valid_geo) < 2 and len(valid_texture) < 2:
        print("[匹配初筛] 可用特征不足，无法进行匹配")
        return [], {
            'status': 'failed',
            'reason': 'insufficient_features',
            'geo_fragments': len(valid_geo),
            'texture_fragments': len(valid_texture)
        }
    
    # 使用具有最多特征类型的碎片集合
    valid = valid_geo if len(valid_geo) >= len(valid_texture) else valid_texture
    if len(valid) < 2:
        return [], {
            'status': 'failed',
            'reason': 'insufficient_valid_fragments',
            'valid_count': len(valid)
        }

    id_to_idx = {f.id: i for i, f in enumerate(valid)}

    # 几何特征
    geo_embs = np.stack([f.geo_embedding for f in valid], axis=0) if len(valid_geo) >= 2 else None
    
    # FPFH特征
    has_fpfh = all(hasattr(f, "fpfh_feature") and f.fpfh_feature is not None for f in valid)
    fpfh_embs = np.stack([f.fpfh_feature for f in valid], axis=0) if has_fpfh else None
    
    # 纹样特征
    has_texture = all(hasattr(f, "texture_embedding") and f.texture_embedding is not None for f in valid)
    texture_embs = np.stack([f.texture_embedding for f in valid], axis=0) if has_texture else None

    ids = [f.id for f in valid]
    n = len(valid)

    # 记录过程信息
    process_info = {
        'status': 'success',
        'total_fragments': len(fragments),
        'valid_fragments': len(valid),
        'feature_types': {},
        'parameters': {
            'top_m_geo': top_m_geo,
            'top_m_fpfh': top_m_fpfh,
            'top_m_texture': top_m_texture,
            'top_k': top_k,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma
        }
    }
    
    # 记录特征类型信息
    if geo_embs is not None:
        process_info['feature_types']['geometry'] = {
            'count': len(valid_geo),
            'dimension': geo_embs.shape[1]
        }
    if fpfh_embs is not None:
        process_info['feature_types']['fpfh'] = {
            'count': sum(1 for f in valid if hasattr(f, 'fpfh_feature') and f.fpfh_feature is not None),
            'dimension': fpfh_embs.shape[1]
        }
    if texture_embs is not None:
        process_info['feature_types']['texture'] = {
            'count': len(valid_texture),
            'dimension': texture_embs.shape[1]
        }
    
    print(f"[匹配初筛] 使用特征类型: {list(process_info['feature_types'].keys())}")
    
    # 1. 构建索引并检索：E_geo Top-M_geo
    geo_neighbors = None
    if geo_embs is not None:
        print(f"[匹配初筛] 构建几何特征索引...")
        index_geo = _build_index(geo_embs, ids, metric="cosine")
        geo_neighbors = _search_index(index_geo, geo_embs, top_m_geo, ids, metric="cosine")
        print(f"[匹配初筛] 几何特征检索完成，获得候选集")

    # 2. E_fpfh Top-M_fpfh（可选）
    fpfh_neighbors = None
    if fpfh_embs is not None:
        print(f"[匹配初筛] 构建FPFH特征索引...")
        index_fpfh = _build_index(fpfh_embs, ids, metric="cosine")
        fpfh_neighbors = _search_index(index_fpfh, fpfh_embs, top_m_fpfh, ids, metric="cosine")
        print(f"[匹配初筛] FPFH特征检索完成")

    # 3. E_texture Top-M_texture（可选）
    texture_neighbors = None
    if texture_embs is not None:
        print(f"[匹配初筛] 构建纹样特征索引...")
        index_texture = _build_index(texture_embs, ids, metric="cosine")
        texture_neighbors = _search_index(index_texture, texture_embs, top_m_texture, ids, metric="cosine")
        print(f"[匹配初筛] 纹样特征检索完成")

    # 4. 合并候选池 + 多模态相似度
    print(f"[匹配初筛] 开始多模态相似度计算...")
    pair_scores = {}  # (i, j) -> (s_geo, s_fpfh, s_texture, s_total)

    for i in range(n):
        cand_ids = set()
        if geo_neighbors is not None:
            for (nid, s) in geo_neighbors[i]:
                cand_ids.add(nid)
        if fpfh_neighbors is not None:
            for (nid, s) in fpfh_neighbors[i]:
                cand_ids.add(nid)
        if texture_neighbors is not None:
            for (nid, s) in texture_neighbors[i]:
                cand_ids.add(nid)

        for j_id in cand_ids:
            j = id_to_idx.get(j_id)
            if j is None or j <= i:
                continue

            # S_geo
            s_geo = 0.0
            if geo_neighbors is not None:
                for (nid, sc) in geo_neighbors[i]:
                    if nid == j_id:
                        s_geo = sc
                        break
                if s_geo == 0.0:
                    for (nid, sc) in geo_neighbors[j]:
                        if nid == valid[i].id:
                            s_geo = sc
                            break
                # 直接计算补充
                if s_geo == 0.0 and geo_embs is not None:
                    a, b = geo_embs[i], geo_embs[j]
                    s_geo = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

            # S_fpfh
            s_fpfh = 0.0
            if fpfh_neighbors is not None:
                for (nid, sc) in fpfh_neighbors[i]:
                    if nid == j_id:
                        s_fpfh = sc
                        break
                if s_fpfh == 0.0:
                    for (nid, sc) in fpfh_neighbors[j]:
                        if nid == valid[i].id:
                            s_fpfh = sc
                            break
                # 直接计算补充
                if s_fpfh == 0.0 and fpfh_embs is not None:
                    a, b = fpfh_embs[i], fpfh_embs[j]
                    s_fpfh = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

            # S_texture
            s_texture = 0.0
            if texture_neighbors is not None:
                for (nid, sc) in texture_neighbors[i]:
                    if nid == j_id:
                        s_texture = sc
                        break
                if s_texture == 0.0:
                    for (nid, sc) in texture_neighbors[j]:
                        if nid == valid[i].id:
                            s_texture = sc
                            break
                # 直接计算补充
                if s_texture == 0.0 and texture_embs is not None:
                    a, b = texture_embs[i], texture_embs[j]
                    s_texture = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

            # 权重计算
            w_geo = alpha if geo_embs is not None else 0.0
            w_fpfh = beta if fpfh_embs is not None else 0.0
            w_texture = gamma if texture_embs is not None else 0.0
            
            # 归一化权重
            total_weight = w_geo + w_fpfh + w_texture
            if total_weight > 0:
                w_geo /= total_weight
                w_fpfh /= total_weight
                w_texture /= total_weight
            
            s_total = w_geo * s_geo + w_fpfh * s_fpfh + w_texture * s_texture

            pair_scores[(valid[i].id, j_id)] = (s_geo, s_fpfh, s_texture, s_total)
            
            # 记录详细信息
            if 'pair_details' not in process_info:
                process_info['pair_details'] = {}
            process_info['pair_details'][f"{valid[i].id}-{j_id}"] = {
                'geo_similarity': float(s_geo),
                'fpfh_similarity': float(s_fpfh),
                'texture_similarity': float(s_texture),
                'total_similarity': float(s_total),
                'weights': {
                    'geo': float(w_geo),
                    'fpfh': float(w_fpfh),
                    'texture': float(w_texture)
                }
            }

    # 5. 排序、阈值、Top-K
    pairs = [(i, j, st) for (i, j), (_, _, _, st) in pair_scores.items() if st >= s_min]
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"[匹配初筛] 相似度计算完成，原始候选对数: {len(pair_scores)}, 筛选后: {len(pairs)}")

    # 每个碎片最多保留 top_k 个候选对
    frag_count = {f.id: 0 for f in valid}
    result = []
    for (i, j, st) in pairs:
        if frag_count[i] < top_k or frag_count[j] < top_k:
            result.append((i, j, st))
            frag_count[i] += 1
            frag_count[j] += 1
    
    # 更新过程信息
    process_info['final_matches'] = len(result)
    process_info['threshold_applied'] = s_min
    
    if result:
        similarities = [m[2] for m in result]
        process_info['similarity_stats'] = {
            'mean': float(np.mean(similarities)),
            'max': float(np.max(similarities)),
            'min': float(np.min(similarities)),
            'std': float(np.std(similarities))
        }
    
    print(f"[匹配初筛] Top-K筛选完成，最终匹配对数: {len(result)}")
    
    return result, process_info