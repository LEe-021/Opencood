"""
预测结果验证模块（改进版 - 高性能）
====================================

验证改进版 IMM-UKF 预测结果的精度和协方差校准度。

当前预测方法特征：
  - 均值预测：从滤波状态直接预测 N×DT 步（与原始 IMM 一致，避免迭代误差累积）
  - 模型权重：固定为滤波阶段的值（不做马尔可夫演化，所有步的权重相同）
  - 协方差：滤波协方差 + 模型散布 + 过程噪声线性增长

性能优化（相对于初版）：
  1. 预加载所有真值 YAML 文件到内存字典（避免数万次重复文件 I/O）
  2. 向量化批量统计计算（numpy 数组操作替代逐项 Python 循环）
  3. 内联计算消除函数调用开销

输入：改进版预测 JSON（含 covariance 和 model_weights 字段）
输出：
  一、预测误差统计
  二、协方差校准度检验
  三、模型权重分布
  四、预估置信度统计
  五、协方差一致性检验
"""

import json
import yaml
import os
import numpy as np
from tqdm import tqdm


# ================= 配置区域 =================
INPUT_ROOT = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1/-1"
PRED_JSON_PATH = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene_noisy/test_1_noisy/prediction_registry_imm_ukf_improved.json"
FILENAME_INCREMENT_PER_STEP = 1
SIGMA_REF = 0.3
CALIBRATION_SIGMA_LEVELS = [1.0, 2.0, 3.0]
# ===========================================


def preload_ground_truth(data_root):
    """
    一次性预加载所有真值 YAML 文件，构建高效查找字典。

    Returns:
        gt_cache: dict, {frame_id_int: {vid_str: (np.array([x,y]), yaw_deg)}}
    """
    gt_cache = {}
    yaml_files = sorted(f for f in os.listdir(data_root) if f.endswith('.yaml'))

    print(f"预加载真值文件: {len(yaml_files)} 个 YAML ...")

    for yf in tqdm(yaml_files, desc="Loading GT"):
        frame_id = int(os.path.splitext(yf)[0])
        yaml_path = os.path.join(data_root, yf)

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if 'vehicles' not in data or data['vehicles'] is None:
            gt_cache[frame_id] = {}
            continue

        vehicles = {}
        for vid, vinfo in data['vehicles'].items():
            loc = vinfo['location']
            vehicles[str(vid)] = (np.array(loc[:2], dtype=np.float64),
                                  float(vinfo['angle'][1]))

        gt_cache[frame_id] = vehicles

    return gt_cache


def main():
    if not os.path.exists(INPUT_ROOT):
        print(f"错误: 找不到目录 {INPUT_ROOT}")
        return

    # 1. 预加载所有真值
    gt_cache = preload_ground_truth(INPUT_ROOT)

    print(f"\n加载预测记录: {PRED_JSON_PATH}")
    with open(PRED_JSON_PATH, 'r') as f:
        predictions = json.load(f)

    # 检测协方差字段
    has_covariance = False
    for k, v in predictions.items():
        for vid, preds in v.items():
            if preds and 'covariance' in preds[0]:
                has_covariance = True
            break
        if has_covariance:
            break

    if has_covariance:
        print("  检测到协方差字段，将执行完整验证。")
    else:
        print("  警告：预测结果不含协方差字段，仅执行基础误差验证。")

    # 2. 收集所有数据（单次遍历，内联计算）
    step_data = {}

    print("解析预测数据并匹配真值...")
    for frame_key, vehicles_pred in tqdm(predictions.items()):
        frame_name = os.path.basename(frame_key)
        try:
            current_frame_id = int(frame_name)
        except ValueError:
            continue

        for vid, steps_list in vehicles_pred.items():
            vid_str = str(vid)

            for item in steps_list:
                step = item['step']
                target_frame_id = current_frame_id + step * FILENAME_INCREMENT_PER_STEP

                if target_frame_id not in gt_cache:
                    continue
                target_vehicles = gt_cache[target_frame_id]
                if vid_str not in target_vehicles:
                    continue

                gt_pos, gt_yaw_deg = target_vehicles[vid_str]

                dx = item['x'] - gt_pos[0]
                dy = item['y'] - gt_pos[1]
                pos_err = np.sqrt(dx * dx + dy * dy)

                pred_yaw_deg = np.degrees(item['yaw'])
                yaw_diff = abs(pred_yaw_deg - gt_yaw_deg) % 360.0
                yaw_err = min(yaw_diff, 360.0 - yaw_diff)

                if step not in step_data:
                    step_data[step] = {
                        'pos_errs': [], 'yaw_errs': [],
                        'cov_available': False,
                    }

                sd = step_data[step]
                sd['pos_errs'].append(pos_err)
                sd['yaw_errs'].append(yaw_err)

                if has_covariance and 'covariance' in item:
                    if not sd['cov_available']:
                        sd['cov_available'] = True
                        sd['pos_sigmas'] = []
                        sd['yaw_sigmas'] = []
                        sd['model_weights'] = []
                        sd['confidences'] = []

                    cov = item['covariance']
                    sigma_pos = np.sqrt(cov[0][0] + cov[1][1])
                    sigma_yaw = np.sqrt(max(cov[3][3], 0.0))

                    sd['pos_sigmas'].append(sigma_pos)
                    sd['yaw_sigmas'].append(sigma_yaw)
                    sd['confidences'].append(1.0 / (1.0 + (sigma_pos / SIGMA_REF) ** 2))

                    if 'model_weights' in item:
                        sd['model_weights'].append(item['model_weights'])

    if not step_data:
        print("\n没有有效对比数据，请检查预测 JSON 的路径和 ID 格式。")
        return

    # 3. 转为 numpy 数组
    steps = sorted(step_data.keys())
    total_valid = 0

    for s in steps:
        sd = step_data[s]
        sd['pos_errs'] = np.array(sd['pos_errs'])
        sd['yaw_errs'] = np.array(sd['yaw_errs'])
        total_valid += len(sd['pos_errs'])
        if sd['cov_available']:
            sd['pos_sigmas'] = np.array(sd['pos_sigmas'])
            sd['yaw_sigmas'] = np.array(sd['yaw_sigmas'])
            sd['confidences'] = np.array(sd['confidences'])
            if sd['model_weights']:
                sd['model_weights'] = np.array(sd['model_weights'])

    # ================= 一、预测误差统计 =================
    print("\n" + "=" * 110)
    print("一、预测误差统计")
    print("  方法: 从滤波状态直接预测 N×DT 步，固定模型权重混合")
    print("-" * 110)
    print(f"{'Step':<5} {'Lat(ms)':<8} | "
          f"{'Mean Pos(m)':<12} {'Var Pos':<12} | "
          f"{'Mean Yaw(°)':<12} {'Var Yaw':<12} | "
          f"{'Count'}")
    print("-" * 110)

    for step in steps:
        sd = step_data[step]
        pe, ye = sd['pos_errs'], sd['yaw_errs']
        time_ms = step * 50

        print(f"{step:<5} {time_ms:<8} | "
              f"{pe.mean():.4f}        {pe.var():.4f}        | "
              f"{ye.mean():.4f}        {ye.var():.4f}        | "
              f"{len(pe)}")

    print("=" * 110)

    if has_covariance:
        theory_rates = {k: (1.0 - np.exp(-k**2 / 2.0)) * 100.0
                        for k in CALIBRATION_SIGMA_LEVELS}

        # ================= 二、协方差校准度检验 =================
        print("\n二、协方差校准度检验")
        print("  含义：实际误差落在 kσ 椭圆内的比例。")
        print("  理论值：1σ→39.3%, 2σ→86.5%, 3σ→98.9%（2D 高斯分布）")
        print("  协方差公式：P_filter + P_dispersion + step × Q_avg")
        print("-" * 110)
        print(f"{'Step':<5} {'Lat(ms)':<8} | "
              f"{'1σ Rate':<10} {'Theory':<8} | "
              f"{'2σ Rate':<10} {'Theory':<8} | "
              f"{'3σ Rate':<10} {'Theory':<8} | "
              f"{'Mean σ_pos(m)':<14} {'Mean σ_yaw(°)':<14}")
        print("-" * 110)

        for step in steps:
            sd = step_data[step]
            time_ms = step * 50

            if sd['cov_available']:
                pe, sp = sd['pos_errs'], sd['pos_sigmas']

                rates = {}
                for k in CALIBRATION_SIGMA_LEVELS:
                    rates[k] = (pe <= k * sp).sum() / len(pe) * 100.0

                mean_sy_deg = np.degrees(sd['yaw_sigmas'].mean())

                print(f"{step:<5} {time_ms:<8} | "
                      f"{rates[1.0]:>7.1f}%   {theory_rates[1.0]:>5.1f}%  | "
                      f"{rates[2.0]:>7.1f}%   {theory_rates[2.0]:>5.1f}%  | "
                      f"{rates[3.0]:>7.1f}%   {theory_rates[3.0]:>5.1f}%  | "
                      f"{sp.mean():<14.4f} {mean_sy_deg:<14.4f}")
            else:
                print(f"{step:<5} {time_ms:<8} | {'N/A':<10}")

        print("=" * 110)

        # ================= 三、模型权重分布 =================
        print("\n三、模型权重分布（均值）")
        print("  权重顺序: [CV, CTRV, CTRA]")
        print("  注意：当前方法在预测阶段保持权重固定，所有步使用滤波阶段的权重。")
        print("-" * 80)
        print(f"{'Step':<5} {'Lat(ms)':<8} | {'CV':<10} {'CTRV':<10} {'CTRA':<10} | {'主导模型'}")
        print("-" * 80)

        model_names = ['CV', 'CTRV', 'CTRA']

        for step in steps:
            sd = step_data[step]
            time_ms = step * 50

            if sd['cov_available'] and len(sd.get('model_weights', [])) > 0:
                mean_w = sd['model_weights'].mean(axis=0)
                dominant = model_names[np.argmax(mean_w)]

                print(f"{step:<5} {time_ms:<8} | "
                      f"{mean_w[0]:<10.4f} {mean_w[1]:<10.4f} {mean_w[2]:<10.4f} | "
                      f"{dominant}")
            else:
                print(f"{step:<5} {time_ms:<8} | {'N/A':<10} {'N/A':<10} {'N/A':<10} | {'N/A'}")

        print("=" * 80)

        # ================= 四、预估置信度统计 =================
        print("\n四、预估置信度统计（目标中心点）")
        print(f"  参数: SIGMA_REF = {SIGMA_REF} m")
        print("  注意：此为简化估计，实际 PR2 输出的逐点置信度还包含杠杆臂效应。")
        print("-" * 90)
        print(f"{'Step':<5} {'Lat(ms)':<8} | {'Mean Conf':<12} {'Min Conf':<12} {'Max Conf':<12} | {'<0.5 比例'}")
        print("-" * 90)

        for step in steps:
            sd = step_data[step]
            time_ms = step * 50

            if sd['cov_available'] and len(sd.get('confidences', [])) > 0:
                confs = sd['confidences']
                below_half = (confs < 0.5).sum() / len(confs) * 100.0

                print(f"{step:<5} {time_ms:<8} | "
                      f"{confs.mean():<12.4f} {confs.min():<12.4f} {confs.max():<12.4f} | "
                      f"{below_half:.1f}%")
            else:
                print(f"{step:<5} {time_ms:<8} | {'N/A':<12} {'N/A':<12} {'N/A':<12} | {'N/A'}")

        print("=" * 90)

        # ================= 五、协方差一致性检验 =================
        print("\n五、协方差一致性检验")
        print("  含义：平均预测误差 / 平均协方差预测标准差（σ_pos）。")
        print("  理想值 ≈ 1.0（误差与协方差预测的不确定性一致）。")
        print("  > 1.0 表示协方差低估了不确定性（过于乐观）。")
        print("  < 1.0 表示协方差高估了不确定性（过于保守）。")
        print("-" * 60)
        print(f"{'Step':<5} {'Lat(ms)':<8} | {'Mean Err(m)':<14} {'Mean σ(m)':<14} | {'Ratio'}")
        print("-" * 60)

        for step in steps:
            sd = step_data[step]
            time_ms = step * 50

            if sd['cov_available']:
                mean_err = sd['pos_errs'].mean()
                mean_sigma = sd['pos_sigmas'].mean()
                ratio = mean_err / mean_sigma if mean_sigma > 0 else float('inf')

                print(f"{step:<5} {time_ms:<8} | "
                      f"{mean_err:<14.4f} {mean_sigma:<14.4f} | "
                      f"{ratio:.4f}")
            else:
                print(f"{step:<5} {time_ms:<8} | {'N/A':<14} {'N/A':<14} | {'N/A'}")

        print("=" * 60)

    print(f"\n验证完成。总有效对比数: {total_valid}")


if __name__ == "__main__":
    main()
