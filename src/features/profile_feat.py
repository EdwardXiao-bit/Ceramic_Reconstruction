import numpy as np


def encode_profile(fragment):
    """
    编码碎片的轮廓特征（为匹配做准备）
    :param fragment: Fragment实例（已提取profile_curve）
    """
    # 确保轮廓已提取
    if fragment.profile_curve is None:
        raise ValueError(f"Fragment {fragment.id} 未提取轮廓，请先执行extract_profile")

    profile = fragment.profile_curve
    # MVP级特征：归一化轮廓的均值/方差/最大值/最小值
    h_vals = profile[:, 0]
    r_vals = profile[:, 1]

    feat = np.concatenate([
        [h_vals.mean(), h_vals.var(), h_vals.max(), h_vals.min()],
        [r_vals.mean(), r_vals.var(), r_vals.max(), r_vals.min()]
    ])
    fragment.profile_feature = feat
    return feat