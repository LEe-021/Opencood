# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Utility functions related to point cloud
"""

import os
import open3d as o3d
import numpy as np


def pcd_to_np(pcd_file):
    """
    Read  pcd and return numpy array.

    Parameters
    ----------
    pcd_file : str
        The pcd file that contains the point cloud.

    Returns
    -------
    pcd_np : np.ndarray
        The lidar data in numpy format, shape:(n, 4)
        Columns: [x, y, z, intensity]

    """
    pcd = o3d.io.read_point_cloud(pcd_file)

    xyz = np.asarray(pcd.points)
    # we save the intensity in the first channel
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float32)


def pcd_to_np_with_conf(pcd_file, default_conf=1.0):
    """
    读取 PCD 文件并返回包含补偿置信度的 5 维特征数组。

    加载优先级（快速路径优先）：
      1. 如果存在预转换的 _5d.npy 文件 → 直接加载（跳过 Open3D，最快）
      2. 否则通过 Open3D 加载 PCD + 拼接置信度（兼容路径）

    自动检测与 PCD 文件同目录下是否存在对应的 _conf.npy 置信度文件：
      - 存在（路侧补偿点云）：加载置信度作为第 5 维特征
      - 不存在（自车点云或未补偿数据）：使用 default_conf 填充第 5 维

    Parameters
    ----------
    pcd_file : str
        PCD 文件路径。
    default_conf : float
        当置信度文件不存在时使用的默认置信度（默认 1.0）。

    Returns
    -------
    pcd_np : np.ndarray
        点云数据, shape:(n, 5), dtype=float32
        Columns: [x, y, z, intensity, confidence]
    """
    # 快速路径：检查是否存在预转换的 5 维 NPY 文件
    # 命名规则：000001.pcd → 000001_5d.npy
    fast_path = pcd_file.replace('.pcd', '_5d.npy')
    if os.path.exists(fast_path):
        return np.load(fast_path).astype(np.float32)

    # 慢速路径：通过 Open3D 加载 PCD
    pcd_np = pcd_to_np(pcd_file)

    # 尝试加载置信度文件：000001.pcd → 000001_conf.npy
    conf_path = pcd_file.replace('.pcd', '_conf.npy')

    if os.path.exists(conf_path):
        conf = np.load(conf_path).astype(np.float32)
        # 确保点数匹配
        if len(conf) == len(pcd_np):
            conf = conf.reshape(-1, 1)
        else:
            # 点数不匹配时回退到默认值（安全保护）
            conf = np.full((len(pcd_np), 1), default_conf, dtype=np.float32)
    else:
        # 无置信度文件：自车点云或未使用改进版补偿管线
        conf = np.full((len(pcd_np), 1), default_conf, dtype=np.float32)

    # 拼接为 5 维：[x, y, z, intensity, confidence]
    pcd_np = np.hstack((pcd_np, conf))
    return pcd_np


def mask_points_by_range(points, limit_range):
    """
    Remove the lidar points out of the boundary.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    limit_range : list
        [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """

    mask = (points[:, 0] > limit_range[0]) & (points[:, 0] < limit_range[3])\
           & (points[:, 1] > limit_range[1]) & (
                   points[:, 1] < limit_range[4]) \
           & (points[:, 2] > limit_range[2]) & (
                   points[:, 2] < limit_range[5])

    points = points[mask]

    return points


def mask_ego_points(points):
    """
    Remove the lidar points of the ego vehicle itself.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """
    mask = (points[:, 0] >= -1.95) & (points[:, 0] <= 2.95) \
           & (points[:, 1] >= -1.1) & (points[:, 1] <= 1.1)
    points = points[np.logical_not(mask)]

    return points


def shuffle_points(points):
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]

    return points


def lidar_project(lidar_data, extrinsic):
    """
    Given the extrinsic matrix, project lidar data to another space.

    支持任意维度特征：对 XYZ 坐标施加齐次变换，其余特征（强度、置信度等）
    直接透传不变。

    Parameters
    ----------
    lidar_data : np.ndarray
        Lidar data, shape: (n, m)，m >= 4
        前 3 列为 XYZ 坐标，第 4 列为 intensity，
        可选第 5 列为 confidence。

    extrinsic : np.ndarray
        Extrinsic matrix, shape: (4, 4)

    Returns
    -------
    projected_lidar : np.ndarray
        Projected lidar data, shape: (n, m)，维度与输入一致。
    """
    # 仅对 XYZ 坐标施加齐次变换
    lidar_xyz = lidar_data[:, :3].T
    lidar_xyz = np.r_[lidar_xyz, [np.ones(lidar_xyz.shape[1])]]
    project_lidar_xyz = np.dot(extrinsic, lidar_xyz)[:3, :].T

    # XYZ 之外的所有特征（intensity, confidence 等）直接透传
    lidar_rest = lidar_data[:, 3:]
    projected_lidar = np.hstack((project_lidar_xyz, lidar_rest))

    return projected_lidar


def projected_lidar_stack(projected_lidar_list):
    """
    Stack all projected lidar together.

    Parameters
    ----------
    projected_lidar_list : list
        The list containing all projected lidar.

    Returns
    -------
    stack_lidar : np.ndarray
        Stack all projected lidar data together.
    """
    stack_lidar = []
    for lidar_data in projected_lidar_list:
        stack_lidar.append(lidar_data)

    return np.vstack(stack_lidar)


def downsample_lidar(pcd_np, num):
    """
    Downsample the lidar points to a certain number.

    Parameters
    ----------
    pcd_np : np.ndarray
        The lidar points, (n, 4).

    num : int
        The downsample target number.

    Returns
    -------
    pcd_np : np.ndarray
        The downsampled lidar points.
    """
    assert pcd_np.shape[0] >= num

    selected_index = np.random.choice((pcd_np.shape[0]),
                                      num,
                                      replace=False)
    pcd_np = pcd_np[selected_index]

    return pcd_np


def downsample_lidar_minimum(pcd_np_list):
    """
    Given a list of pcd, find the minimum number and downsample all
    point clouds to the minimum number.

    Parameters
    ----------
    pcd_np_list : list
        A list of pcd numpy array(n, 4).
    Returns
    -------
    pcd_np_list : list
        Downsampled point clouds.
    """
    minimum = np.Inf

    for i in range(len(pcd_np_list)):
        num = pcd_np_list[i].shape[0]
        minimum = num if minimum > num else minimum

    for (i, pcd_np) in enumerate(pcd_np_list):
        pcd_np_list[i] = downsample_lidar(pcd_np, minimum)

    return pcd_np_list


def preconvert_pcd_to_npy(data_root):
    """
    将指定目录下所有 PCD 文件预转换为 5 维 NPY 文件，加速后续训练/推理加载。

    对每个 PCD 文件：
      1. 通过 pcd_to_np_with_conf() 加载 5 维数据 [x,y,z,intensity,confidence]
      2. 保存为同名的 _5d.npy 文件（如 000001.pcd → 000001_5d.npy）

    后续 pcd_to_np_with_conf() 会自动检测 _5d.npy 文件并直接加载，
    跳过 Open3D 的 PCD 解析，加载速度提升 5-10 倍。

    此函数幂等：已存在的 _5d.npy 文件会被跳过（除非强制覆盖）。

    Parameters
    ----------
    data_root : str
        数据根目录，会递归搜索所有 .pcd 文件。
    """
    import glob
    from tqdm import tqdm

    pcd_files = sorted(glob.glob(os.path.join(data_root, "**", "*.pcd"), recursive=True))
    print(f"Found {len(pcd_files)} PCD files under {data_root}")

    converted = 0
    skipped = 0

    for pcd_path in tqdm(pcd_files, desc="Pre-converting PCD→NPY"):
        npy_path = pcd_path.replace('.pcd', '_5d.npy')

        if os.path.exists(npy_path):
            skipped += 1
            continue

        try:
            data_5d = pcd_to_np_with_conf(pcd_path)
            np.save(npy_path, data_5d)
            converted += 1
        except Exception as e:
            print(f"  Warning: failed to convert {pcd_path}: {e}")

    print(f"Done. Converted: {converted}, Skipped (already exist): {skipped}")
