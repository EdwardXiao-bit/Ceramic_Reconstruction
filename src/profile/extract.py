import numpy as np
import open3d as o3d


def extract_profile(fragment, n_bins=100):
    """
    修正：接收Fragment对象，提取其点云的轮廓
    :param fragment: Fragment对象（包含point_cloud）
    :param n_bins: 分箱数量
    :return: 轮廓数组 + 主轴线
    """
    if fragment.point_cloud is None:
        print(f"碎片{fragment.id}无点云数据，跳过轮廓提取")
        # 返回空二维数组，避免后续索引错误
        fragment.profile_curve = np.empty((0, 2))
        fragment.main_axis = None
        return fragment.profile_curve, fragment.main_axis

    pcd = fragment.point_cloud
    pts = np.asarray(pcd.points)
    center = pts.mean(axis=0)
    pts -= center

    _, _, Vt = np.linalg.svd(pts)
    axis = Vt[0]

    h = pts @ axis
    r = np.linalg.norm(pts - np.outer(h, axis), axis=1)

    bins = np.linspace(h.min(), h.max(), n_bins)
    profile = []
    for i in range(len(bins) - 1):
        mask = (h >= bins[i]) & (h < bins[i + 1])
        if mask.sum() > 10:  # 有效分箱：点数>10
            profile.append([bins[i], r[mask].mean()])
        else:
            print(f"[轮廓提取] 碎片{fragment.id}分箱{i}点数不足，跳过")

    # 关键修复：确保profile_arr始终是二维数组
    profile_arr = np.array(profile) if profile else np.empty((0, 2))

    # 保存到Fragment，兼容空轮廓
    fragment.profile_curve = profile_arr
    fragment.main_axis = axis

    if len(profile_arr) == 0:
        print(f"[轮廓提取] 碎片{fragment.id}无有效轮廓分箱，返回空轮廓")
    else:
        print(f"[轮廓提取] 碎片{fragment.id}提取到{len(profile_arr)}个轮廓分箱")

    return profile_arr, axis