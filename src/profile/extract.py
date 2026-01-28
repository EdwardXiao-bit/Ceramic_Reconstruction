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
        return None, None

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
        if mask.sum() > 10:
            profile.append([bins[i], r[mask].mean()])

    profile_arr = np.array(profile)
    # 保存轮廓到Fragment对象
    fragment.profile_curve = profile_arr
    fragment.main_axis = axis
    return profile_arr, axis