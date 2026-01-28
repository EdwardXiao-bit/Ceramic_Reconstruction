import numpy as np
from scipy.spatial.distance import cosine


def coarse_match(fragments):
    """
    粗匹配：计算碎片间的轮廓特征相似度，返回匹配对
    :param fragments: list[Fragment] 碎片实例列表
    :return: list[tuple] 匹配对 (frag1_id, frag2_id, 相似度)
    """
    # 过滤无特征的碎片
    valid_frags = [f for f in fragments if f.profile_feature is not None]
    if len(valid_frags) < 2:
        return []

    matches = []
    # 两两计算相似度
    for i in range(len(valid_frags)):
        for j in range(i + 1, len(valid_frags)):
            f1, f2 = valid_frags[i], valid_frags[j]
            # 余弦相似度（1 - 余弦距离）
            similarity = 1 - cosine(f1.profile_feature, f2.profile_feature)
            matches.append((f1.id, f2.id, similarity))

    # 按相似度排序（取前N对，MVP级仅返回所有匹配）
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches