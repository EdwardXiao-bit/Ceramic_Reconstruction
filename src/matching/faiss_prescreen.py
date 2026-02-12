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
    top_k=10,
    alpha=0.7,
    beta=0.3,
    s_min=0.0,
):
    """
    碎片匹配初筛：FAISS 检索 + 多模态相似度 + Top-K
    :param fragments: list[Fragment]
    :param top_m_geo: 几何 embedding 检索候选数
    :param top_m_fpfh: FPFH 检索候选数
    :param top_k: 每个碎片保留的 Top-K 候选对
    :param alpha: S_geo 权重
    :param beta: S_fpfh 权重（alpha + beta 建议 = 1）
    :param s_min: 最低相似度阈值
    :return: list[tuple] (frag1_id, frag2_id, s_total)
    """
    valid = [f for f in fragments if hasattr(f, "geo_embedding") and f.geo_embedding is not None]
    if len(valid) < 2:
        return []

    id_to_idx = {f.id: i for i, f in enumerate(valid)}

    geo_embs = np.stack([f.geo_embedding for f in valid], axis=0)
    has_fpfh = all(hasattr(f, "fpfh_feature") and f.fpfh_feature is not None for f in valid)
    fpfh_embs = np.stack([f.fpfh_feature for f in valid], axis=0) if has_fpfh else None

    ids = [f.id for f in valid]
    n = len(valid)

    # 1. 构建索引并检索：E_geo Top-M_geo
    index_geo = _build_index(geo_embs, ids, metric="cosine")
    # 移除 use_faiss 参数，让 _search_index 自动判断
    geo_neighbors = _search_index(index_geo, geo_embs, top_m_geo, ids, metric="cosine")

    # 2. E_fpfh Top-M_fpfh（可选）
    if fpfh_embs is not None:
        index_fpfh = _build_index(fpfh_embs, ids, metric="cosine")
        # 移除 use_faiss 参数，让 _search_index 自动判断
        fpfh_neighbors = _search_index(index_fpfh, fpfh_embs, top_m_fpfh, ids, metric="cosine")
    else:
        fpfh_neighbors = None

    # 3. 合并候选池 + 多模态相似度
    pair_scores = {}  # (i, j) -> (s_geo, s_fpfh, s_total)

    for i in range(n):
        cand_ids = set()
        for (nid, s) in geo_neighbors[i]:
            cand_ids.add(nid)
        if fpfh_neighbors is not None:
            for (nid, s) in fpfh_neighbors[i]:
                cand_ids.add(nid)

        for j_id in cand_ids:
            j = id_to_idx.get(j_id)
            if j is None or j <= i:
                continue

            # S_geo
            s_geo = 0.0
            for (nid, sc) in geo_neighbors[i]:
                if nid == j_id:
                    s_geo = sc
                    break
            if s_geo == 0.0:
                for (nid, sc) in geo_neighbors[j]:
                    if nid == valid[i].id:
                        s_geo = sc
                        break

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

            # 若 FPFH 未命中，用直接余弦相似度补
            if s_fpfh == 0.0 and fpfh_embs is not None:
                a, b = fpfh_embs[i], fpfh_embs[j]
                s_fpfh = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            if s_geo == 0.0:
                a, b = geo_embs[i], geo_embs[j]
                s_geo = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

            w_fpfh = beta if fpfh_embs is not None else 0.0
            w_geo = alpha if w_fpfh > 0 else 1.0
            s_total = w_geo * s_geo + w_fpfh * s_fpfh

            pair_scores[(valid[i].id, j_id)] = (s_geo, s_fpfh, s_total)

    # 4. 排序、阈值、Top-K
    pairs = [(i, j, st) for (i, j), (_, _, st) in pair_scores.items() if st >= s_min]
    pairs.sort(key=lambda x: x[2], reverse=True)

    # 每个碎片最多保留 top_k 个候选对
    frag_count = {f.id: 0 for f in valid}
    result = []
    for (i, j, st) in pairs:
        if frag_count[i] < top_k or frag_count[j] < top_k:
            result.append((i, j, st))
            frag_count[i] += 1
            frag_count[j] += 1

    return result