"""
点云补偿与置信度引导采样模块（改进版）
=======================================

基于改进版 IMM-UKF 预测结果（含协方差矩阵），对路侧 LiDAR 点云进行
实例级刚体变换补偿，并基于补偿置信度进行概率性点云采样。

核心策略：
  - 高置信度点（补偿可靠）→ 保留
  - 低置信度点（补偿不可靠）→ 以概率丢弃
  - 背景点（无需补偿）→ 始终保留
  - 输出标准 4D PCD（无需置信度特征通道，兼容任意检测模型）

置信度计算：
  - 位置不确定性来自协方差矩阵对角线元素
  - 航向角不确定性通过杠杆臂效应影响距中心较远的点
  - sigmoid 映射：(0,1]，1.0 = 完全信任

输出结构：
  output_dir/
    test_1_compensated_100ms/
      -1/
        000001.pcd          ← 补偿+采样后点云（标准格式）
        000001.yaml         ← 车辆信息（复制自原始数据）
    ...
"""

import open3d as o3d
import numpy as np
import yaml
import json
import os
import glob
import copy
from tqdm import tqdm
import shutil


# ========================= 配置区域 =========================
# 1. 原始数据路径 (路侧 -1 文件夹，清洁仿真数据)
INPUT_PC_ROOT = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1/-1"

# 2. 改进版预测结果 JSON 路径（含协方差矩阵和模型权重）
PRED_JSON_PATH = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene_noisy/test_1_noisy/prediction_registry_imm_ukf_improved.json"

# 3. 输出数据根目录
OUTPUT_BASE_DIR = "/home/step/data/pcPredict_data/cross_fullcar_clean/IMM_predict_improved"

# 4. 要处理的预测步列表（1~6 对应 50ms~300ms）
STEPS_TO_PROCESS = [1, 2, 3, 4, 5, 6]

# 5. 数据采集间隔（毫秒，用于生成文件夹命名）
TIME_INTERVAL_MS = 50

# 6. RSU 在世界坐标系的安装位置 (lidar_pose)
RSU_POSITION = np.array([-27.0, -2.0, 6.0])

# 7. 置信度计算参考尺度（米）
#    当 effective_sigma = SIGMA_REF 时，confidence = 0.5
#    SIGMA_REF 越小 → 对不确定性越敏感（更保守）
#    SIGMA_REF 越大 → 对不确定性越宽容（更激进）
SIGMA_REF = 0.3

# 8. 未补偿点的默认置信度
#    不在任何目标边界框内的点（背景点）无需补偿，置信度为 1.0
DEFAULT_CONFIDENCE = 1.0

# 9. 置信度引导采样开关
#    True  = 置信度引导采样模式：低置信度点以概率被丢弃
#    False = 特征通道模式：保存置信度 NPY 供模型作为第5维特征
CONF_SAMPLING_MODE = True

# 10. 保留概率放大因子
#     keep_prob = min(confidence × CONF_SCALE_FACTOR, 1.0)
#     放大置信度以提高整体保留率
CONF_SCALE_FACTOR = 1.7

# 11. 最低保留概率
#     防止丢点过多导致目标完全消失
CONF_DROP_FLOOR = 0.5
# ============================================================


# ========================= 坐标变换工具 =========================

def deg2rad(deg):
    """角度转弧度"""
    return deg * np.pi / 180.0


def rsu2world(points):
    """
    将点云从 RSU 局部坐标系转换到世界坐标系。

    变换方式：平移至 RSU 安装位置。
    假设 RSU 朝向与世界坐标系对齐（无旋转偏移）。
    """
    points_world = copy.deepcopy(points)
    points_world += RSU_POSITION
    return points_world


def world2rsu(points_world):
    """
    将点云从世界坐标系转换回 RSU 局部坐标系。
    """
    points_local = points_world - RSU_POSITION
    return points_local


def get_rotation_matrix_z(theta_rad):
    """
    构建 Z 轴旋转矩阵（2D 旋转在 3D 空间中的表示）。

    用于对裁剪出的车辆点云施加航向角补偿。
    """
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


# ========================= 车辆包围盒构建 =========================

def get_vehicle_obb_world(vinfo):
    """
    根据 YAML 中的车辆信息构建世界坐标系下的定向包围盒 (OBB)。

    OBB 用于从路侧点云中裁剪出属于特定目标的点簇，
    以便后续对该目标的点云单独施加刚体变换补偿。

    Args:
        vinfo: 车辆信息字典，包含以下字段：
            - location: [x, y, z] 车辆参考点位置
            - angle: [pitch, yaw, roll] 航向角（度）
            - extent: [half_w, half_l, half_h] 半尺寸
            - center: [cx, cy, cz] 局部中心偏移

    Returns:
        obb: Open3D OrientedBoundingBox 实例
        center_world: OBB 中心在世界坐标系下的坐标, shape=(3,)
        yaw_rad: 航向角 (弧度)
    """
    loc = np.array(vinfo["location"], dtype=np.float64)
    angle = vinfo["angle"]
    yaw_rad = deg2rad(float(angle[1]))

    R = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    extent = np.array(vinfo["extent"], dtype=np.float64)
    full_extent = extent * 2.0

    center_local = np.array(vinfo.get("center", [0, 0, 0]), dtype=np.float64)
    center_world = loc + R @ center_local

    obb = o3d.geometry.OrientedBoundingBox(center_world, R, full_extent)
    return obb, center_world, yaw_rad


def get_vehicle_obb_params(vinfo):
    """
    提取车辆 OBB 参数（纯 numpy，不创建 Open3D 对象）。

    返回的参数用于 crop_points_by_obb_numpy 进行高效的向量化点裁剪。
    相比 Open3D 的 OBB 裁剪，避免了创建 PointCloud 对象的开销。

    Args:
        vinfo: 车辆信息字典

    Returns:
        center_world: (3,) OBB 中心，世界坐标
        R: (3, 3) 旋转矩阵（local → world）
        half_extent: (3,) 半尺寸
        yaw_rad: 航向角 (rad)
    """
    loc = np.array(vinfo["location"], dtype=np.float64)
    yaw_rad = deg2rad(float(vinfo["angle"][1]))

    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    half_extent = np.array(vinfo["extent"], dtype=np.float64)
    center_local = np.array(vinfo.get("center", [0, 0, 0]), dtype=np.float64)
    center_world = loc + R @ center_local

    return center_world, R, half_extent, yaw_rad


def crop_points_by_obb_numpy(points, center, R, half_extent):
    """
    纯 numpy 向量化 OBB 裁剪，替代 Open3D 的 get_point_indices_within_bounding_box。

    原理：将世界坐标点变换到 OBB 局部坐标系，检查各轴是否在半尺寸范围内。
    完全向量化操作，无 Python 循环，比 Open3D 方法快 3-5 倍。

    Args:
        points: 世界坐标点云, shape=(N, 3)
        center: OBB 中心, shape=(3,)
        R: 旋转矩阵 local→world, shape=(3, 3)
        half_extent: 半尺寸, shape=(3,)

    Returns:
        indices: 落在 OBB 内的点的索引, shape=(M,)
    """
    # world → local: R^T @ (p - center)
    # 行向量等价: d_row @ R = (R^T @ d_col)^T
    pts_local = (points - center) @ R
    mask = np.all(np.abs(pts_local) <= half_extent, axis=1)
    return np.where(mask)[0]


# ========================= 置信度计算 =========================

def compute_point_confidence(points_world, bbox_center, covariance):
    """
    基于预测协方差矩阵计算逐点补偿置信度。

    置信度反映"补偿后该点位置坐标的可靠程度"：
        - 预测协方差小 → 预测准确 → 置信度高（接近 1.0）
        - 预测协方差大 → 预测不确定 → 置信度低（但始终 > 0）

    杠杆臂效应：
        距目标中心越远的点，航向角误差造成的位置偏移越大（杠杆原理），
        因此其置信度越低。这是物理上的直觉：车辆边缘的点更容易受到
        航向角预测误差的影响，而车辆中心附近的点受影响较小。

    数学公式：
        effective_sigma_i = sqrt( sigma_pos² + (dx_i² + dy_i²) * sigma_yaw² )
        confidence_i = 1 / (1 + (effective_sigma_i / sigma_ref)²)

    其中：
        sigma_pos² = P[0,0] + P[1,1]      位置不确定性（来自协方差矩阵）
        sigma_yaw² = P[3,3]               航向角不确定性（来自协方差矩阵）
        dx_i, dy_i = 点 i 到包围盒中心的 XY 偏移
        sigma_ref = 参考尺度参数（全局配置）

    sigmoid 映射的特性：
        - effective_sigma = 0    → confidence = 1.0（完全信任）
        - effective_sigma = σ_ref → confidence = 0.5
        - effective_sigma → ∞    → confidence → 0（但始终 > 0）
        - 曲线平滑，无断点，不会硬性丢弃任何点

    Args:
        points_world: 待评估的点云坐标, shape=(N, 3), 世界坐标系
        bbox_center:  目标包围盒中心坐标, shape=(3,), 世界坐标系
        covariance:   预测协方差矩阵, shape=(6, 6)
                      状态向量顺序: [px, py, v, yaw, yaw_rate, a]

    Returns:
        confidences: 各点置信度分数, shape=(N,), dtype=float32, 范围 (0, 1]
    """
    # 从协方差矩阵提取位置和航向角的方差
    # P[0,0] = px 方差, P[1,1] = py 方差, P[3,3] = yaw 方差
    sigma_pos_sq = covariance[0, 0] + covariance[1, 1]    # 总位置不确定性 (m²)
    sigma_yaw_sq = covariance[3, 3]                        # 航向角不确定性 (rad²)

    # 计算每个点到目标包围盒中心的 XY 偏移
    dx = points_world[:, 0] - bbox_center[0]   # shape=(N,)
    dy = points_world[:, 1] - bbox_center[1]   # shape=(N,)

    # 综合位置不确定性和杠杆臂效应
    # 位置不确定性对全体点均匀影响
    # 角度不确定性通过杠杆臂放大，距中心越远影响越大
    effective_sigma_sq = sigma_pos_sq + (dx ** 2 + dy ** 2) * sigma_yaw_sq
    effective_sigma = np.sqrt(np.maximum(effective_sigma_sq, 0.0))  # 数值保护

    # sigmoid 映射到 (0, 1]
    confidences = 1.0 / (1.0 + (effective_sigma / SIGMA_REF) ** 2)

    return confidences.astype(np.float32)


# ========================= 单帧处理 =========================

def process_single_frame_all_steps(pcd_path, yaml_path, predictions, output_dirs):
    """
    处理单帧数据：对所有指定的预测步执行点云补偿和置信度计算。

    性能优化（相对于原版）：
        1. OBB 参数预计算并缓存：所有步共享同一组 OBB 参数和点索引
        2. 纯 numpy 向量化 OBB 裁剪：替代 Open3D 的 PointCloud + OBB 方式
        3. 二进制 PCD 输出：write_ascii=False，写入速度快 3-5 倍

    Args:
        pcd_path: 输入 PCD 文件路径（原始路侧点云）
        yaml_path: 对应的 YAML 文件路径（车辆信息）
        predictions: 改进版预测结果字典（含 covariance 和 model_weights）
        output_dirs: 各步对应的输出目录字典 {step_number: directory_path}
    """
    # 1. 读取基础数据（整帧只读一次）
    pcd = o3d.io.read_point_cloud(pcd_path)
    original_points_local = np.asarray(pcd.points)

    if len(original_points_local) == 0:
        return

    has_colors = pcd.has_colors()
    colors = np.asarray(pcd.colors) if has_colors else None

    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    # 2. 转世界坐标
    base_points_world = rsu2world(original_points_local)

    frame_id_str = os.path.splitext(os.path.basename(pcd_path))[0]
    file_name = os.path.basename(pcd_path)

    # 查找当前帧的预测数据
    current_frame_preds = None
    for k, v in predictions.items():
        if k.endswith(f"/{frame_id_str}") or k == frame_id_str:
            current_frame_preds = v
            break

    # 3. 预计算所有车辆的 OBB 参数和点索引（所有步共享）
    #    OBB 裁剪基于原始未变换点云，所有步的裁剪结果相同
    vehicle_obbs = []  # [(vid_str, indices, curr_center, curr_yaw)]
    if current_frame_preds and 'vehicles' in yaml_data and yaml_data['vehicles']:
        for vid_str, vinfo in yaml_data['vehicles'].items():
            vid_str = str(vid_str)
            if vid_str in current_frame_preds:
                center, R, half_ext, yaw_rad = get_vehicle_obb_params(vinfo)
                indices = crop_points_by_obb_numpy(base_points_world, center, R, half_ext)
                if len(indices) > 0:
                    vehicle_obbs.append((vid_str, indices, center, yaw_rad))

    # === 循环处理每个预测步 ===
    for step in STEPS_TO_PROCESS:

        points_world_step = base_points_world.copy()
        confidences = np.full(len(points_world_step), DEFAULT_CONFIDENCE, dtype=np.float32)

        if vehicle_obbs:
            for vid_str, indices, curr_center, curr_yaw in vehicle_obbs:
                # 查找对应步的预测结果
                target_pred = None
                for p in current_frame_preds[vid_str]:
                    if p['step'] == step:
                        target_pred = p
                        break

                if target_pred is None:
                    continue

                vehicle_points = points_world_step[indices]

                # 刚体变换补偿
                pred_x = target_pred['x']
                pred_y = target_pred['y']
                pred_yaw = target_pred['yaw']

                delta_pos = np.array([pred_x, pred_y, curr_center[2]]) - curr_center
                delta_yaw = pred_yaw - curr_yaw

                pts_centered = vehicle_points - curr_center
                R_mat = get_rotation_matrix_z(delta_yaw)
                pts_rotated = pts_centered @ R_mat.T
                pts_final = pts_rotated + (curr_center + delta_pos)

                points_world_step[indices] = pts_final

                # 置信度计算
                if 'covariance' in target_pred:
                    cov_matrix = np.array(target_pred['covariance'])
                    pt_confidences = compute_point_confidence(
                        vehicle_points, curr_center, cov_matrix
                    )
                    confidences[indices] = pt_confidences
                else:
                    confidences[indices] = 0.5

        # 4. 转回 RSU 局部坐标
        points_out_local = world2rsu(points_world_step)

        # 5. 置信度引导采样 or 特征通道模式
        if CONF_SAMPLING_MODE:
            # 概率性丢弃低置信度点
            keep_probs = np.clip(confidences * CONF_SCALE_FACTOR, CONF_DROP_FLOOR, 1.0)
            keep_mask = np.random.random(len(keep_probs)) <= keep_probs

            points_out_local = points_out_local[keep_mask]
            frame_colors = colors[keep_mask] if has_colors else None
        else:
            frame_colors = colors if has_colors else None

        # 6. 保存（二进制格式，比 ASCII 快 3-5 倍）
        target_dir = output_dirs[step]
        save_pcd_path = os.path.join(target_dir, file_name)

        out_pcd = o3d.geometry.PointCloud()
        out_pcd.points = o3d.utility.Vector3dVector(points_out_local)
        if frame_colors is not None:
            out_pcd.colors = o3d.utility.Vector3dVector(frame_colors)

        o3d.io.write_point_cloud(save_pcd_path, out_pcd)

        # 7. 保存置信度 NPY（仅特征通道模式）
        if not CONF_SAMPLING_MODE:
            conf_filename = os.path.splitext(file_name)[0] + "_conf.npy"
            np.save(os.path.join(target_dir, conf_filename), confidences)

        # 8. 复制 YAML
        shutil.copy(yaml_path, save_pcd_path.replace('.pcd', '.yaml'))


# ========================= 主函数 =========================

def main():
    """
    主处理流程：
    1. 加载改进版 IMM-UKF 预测结果（含协方差矩阵）
    2. 遍历所有路侧点云帧
    3. 对每帧执行多步补偿 + 逐点置信度计算
    4. 保存输出：补偿后点云 (PCD) + 置信度数组 (NPY) + 车辆信息 (YAML)
    """
    # 加载预测结果
    print(f"Loading predictions from {PRED_JSON_PATH}...")
    with open(PRED_JSON_PATH, 'r') as f:
        predictions = json.load(f)

    # 检查预测结果是否包含协方差字段
    has_covariance = False
    for k, v in predictions.items():
        for vid, preds in v.items():
            if preds and 'covariance' in preds[0]:
                has_covariance = True
            break
        if has_covariance:
            break

    if has_covariance:
        if CONF_SAMPLING_MODE:
            print("  检测到协方差字段，将执行置信度引导采样（低置信度点概率性丢弃）。")
        else:
            print("  检测到协方差字段，将计算逐点补偿置信度并保存为 NPY。")
    else:
        print("  警告：预测结果不含协方差字段，将使用默认置信度 (0.5)。")

    # 获取源文件列表
    search_pattern = os.path.join(INPUT_PC_ROOT, "*.pcd")
    pcd_files = sorted(glob.glob(search_pattern))
    print(f"Found {len(pcd_files)} PCD files.")

    # 初始化输出目录
    output_dirs = {}
    dataset_name = os.path.basename(os.path.dirname(INPUT_PC_ROOT))
    if dataset_name == "-1":
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(INPUT_PC_ROOT)))

    print(f"Preparing output directories in {OUTPUT_BASE_DIR}...")
    for step in STEPS_TO_PROCESS:
        latency_ms = step * TIME_INTERVAL_MS
        folder_name = f"{dataset_name}_compensated_{latency_ms}ms"
        full_path = os.path.join(OUTPUT_BASE_DIR, folder_name)
        full_path_with_sensor = os.path.join(full_path, "-1")

        if not os.path.exists(full_path_with_sensor):
            os.makedirs(full_path_with_sensor)

        output_dirs[step] = full_path_with_sensor
        print(f"  - Step {step} ({latency_ms}ms) -> {full_path_with_sensor}")

    # 批量处理
    print("Starting batch processing...")
    for pcd_path in tqdm(pcd_files):
        yaml_path = pcd_path.replace('.pcd', '.yaml')
        if not os.path.exists(yaml_path):
            continue

        process_single_frame_all_steps(pcd_path, yaml_path, predictions, output_dirs)

    print("All done!")


if __name__ == "__main__":
    main()
