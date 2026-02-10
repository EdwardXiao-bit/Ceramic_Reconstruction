import numpy as np


def encode_profile(fragment):
    """
    编码碎片的轮廓特征（为匹配做准备）
    :param fragment: Fragment实例（已提取profile_curve）
    """
    # 1. 基础检查：是否存在轮廓
    if fragment.profile_curve is None:
        raise ValueError(f"Fragment {fragment.id} 未提取轮廓，请先执行extract_profile")

    profile = fragment.profile_curve

    # 2. 形状检查：必须是二维数组且列数=2
    if len(profile.shape) != 2 or profile.shape[1] != 2:
        print(f"[轮廓编码] 碎片{fragment.id}轮廓形状异常（{profile.shape}），跳过编码")
        fragment.profile_feature = None
        return None

    # 3. 空轮廓处理
    if len(profile) == 0:
        print(f"[轮廓编码] 碎片{fragment.id}轮廓为空，跳过编码")
        fragment.profile_feature = None
        return None

    # 4. 正常编码（MVP级特征）
    h_vals = profile[:, 0]
    r_vals = profile[:, 1]

    feat = np.concatenate([
        [h_vals.mean(), h_vals.var(), h_vals.max(), h_vals.min()],
        [r_vals.mean(), r_vals.var(), r_vals.max(), r_vals.min()]
    ])
    fragment.profile_feature = feat
    return feat