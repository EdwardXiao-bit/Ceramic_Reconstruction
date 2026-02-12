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


def faiss_prescreen(fragments, top_m_geo=50, top_m_fpfh=50, top_k=20, alpha=0.7, beta=0.3, metric="cosine"):
    """
    FAISS 初筛 Top-K 候选对（兼容旧接口）
    :param fragments: Fragment 列表，每个需有 profile_feature 和 geometry_feature
    :param top_m_geo: 几何特征Top-M数量
    :param top_m_fpfh: FPFH特征Top-M数量
    :param top_k: 返回候选对数量
    :param alpha: 几何特征权重
    :param beta: FPFH特征权重
    :param metric: 距离度量
    :return: [(i, j, score), ...] 候选对列表
    """
    if len(fragments) < 2:
        return []

    # 1. 特征拼接（文档要求：多模态融合）
    embeddings = []
    valid_ids = []
    
    for i, frag in enumerate(fragments):
        if hasattr(frag, 'profile_feature') and hasattr(frag, 'geometry_feature'):
            if frag.profile_feature is not None and frag.geometry_feature is not None:
                # 拼接特征向量
                combined = np.concatenate([frag.profile_feature, frag.geometry_feature])
                embeddings.append(combined)
                valid_ids.append(i)
    
    if len(embeddings) < 2:
        return []

    # 2. 构建索引
    embeddings = np.array(embeddings)
    index, ids = _build_index(embeddings, valid_ids, metric)

    # 3. 查询近邻
    if HAS_FAISS:
        if metric == "cosine":
            # FAISS 余弦相似度（内积越大越相似）
            scores, neighbors = index.search(embeddings, min(top_k + 1, len(embeddings)))
            # 转换为相似度分数（越大越好）
            scores = (scores + 1) / 2  # 归一化到[0,1]
        else:
            # L2距离（越小越相似）
            distances, neighbors = index.search(embeddings, min(top_k + 1, len(embeddings)))
            scores = 1.0 / (1.0 + distances)  # 转换为相似度
    else:
        # sklearn版本
        distances, neighbors = index.kneighbors(embeddings, n_neighbors=min(top_k + 1, len(embeddings)))
        if metric == "cosine":
            scores = 1.0 - distances  # 余弦距离转相似度
        else:
            scores = 1.0 / (1.0 + distances)

    # 4. 构造候选对
    candidates = []
    for i in range(len(embeddings)):
        query_id = ids[i]
        for j in range(1, min(top_k + 1, len(neighbors[i]))):  # 跳过自己
            neighbor_id = ids[neighbors[i][j]]
            score = float(scores[i][j])
            candidates.append((query_id, neighbor_id, score))

    # 5. 按分数排序并去重
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    # 去重（避免重复对）
    seen_pairs = set()
    unique_candidates = []
    for cand in candidates:
        pair = tuple(sorted([cand[0], cand[1]]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_candidates.append(cand)
    
    return unique_candidates[:top_k]


def compute_multimodal_similarity(frag1, frag2, weights=None):
    """
    计算两个碎片的多模态综合相似度
    :param frag1, frag2: Fragment 对象
    :param weights: 各模态权重 {'profile': 0.5, 'geometry': 0.5}
    :return: 综合相似度分数 [0, 1]
    """
    if weights is None:
        weights = {'profile': 0.5, 'geometry': 0.5}
    
    total_score = 0.0
    total_weight = 0.0
    
    # 轮廓特征相似度
    if (hasattr(frag1, 'profile_feature') and frag1.profile_feature is not None and
        hasattr(frag2, 'profile_feature') and frag2.profile_feature is not None):
        sim = cosine_similarity(frag1.profile_feature, frag2.profile_feature)
        total_score += weights.get('profile', 0.5) * sim
        total_weight += weights.get('profile', 0.5)
    
    # 几何特征相似度
    if (hasattr(frag1, 'geometry_feature') and frag1.geometry_feature is not None and
        hasattr(frag2, 'geometry_feature') and frag2.geometry_feature is not None):
        sim = cosine_similarity(frag1.geometry_feature, frag2.geometry_feature)
        total_score += weights.get('geometry', 0.5) * sim
        total_weight += weights.get('geometry', 0.5)
    
    if total_weight > 0:
        return total_score / total_weight
    else:
        return 0.0


def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)
