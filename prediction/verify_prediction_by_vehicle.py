import json
import yaml
import os
import numpy as np
from tqdm import tqdm
import csv

# ================= 配置区域 =================
# 1. 原始数据路径
INPUT_ROOT = "/home/step/data/pcPredict_data/cross_multicar_clean/scene/test_1/-1"

# 2. 预测结果 JSON
PRED_JSON_PATH = "/home/step/data/pcPredict_data/cross_multicar_clean/scene/test_1/prediction_registry.json"

# 3. 结果保存 CSV
OUTPUT_CSV_PATH = "/home/step/data/pcPredict_data/cross_multicar_clean/scene/test_1/prediction_error_by_direction.csv"

# 4. 步长设置
FILENAME_INCREMENT_PER_STEP = 1
TIME_INTERVAL_MS = 50

# 5. 直行角度波动阈值 (度)
#    如果角度变化在 +/- 10度以内，视为直行
STRAIGHT_YAW_THRESHOLD = 5.0 
# ===========================================

def get_gt_data(yaml_path, vehicle_id):
    if not os.path.exists(yaml_path):
        return None, None
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    if 'vehicles' not in data or data['vehicles'] is None:
        return None, None

    vinfo = None
    if vehicle_id in data['vehicles']:
        vinfo = data['vehicles'][vehicle_id]
    elif int(vehicle_id) in data['vehicles']:
        vinfo = data['vehicles'][int(vehicle_id)]
    elif str(vehicle_id) in data['vehicles']:
        vinfo = data['vehicles'][str(vehicle_id)]

    if vinfo is None:
        return None, None

    loc = vinfo['location']
    loc_np = np.array(loc[:2])
    yaw_deg = float(vinfo['angle'][1])
    return loc_np, yaw_deg

def calculate_yaw_error(pred_deg, gt_deg):
    diff = abs(pred_deg - gt_deg)
    diff = diff % 360.0 
    error = min(diff, 360.0 - diff)
    return error

def get_angle_diff(start, end):
    """
    计算角度变化量 (End - Start)
    处理跨越 0/360 度的情况
    返回值的范围在 -180 到 180 之间
    """
    diff = end - start
    while diff > 180: diff -= 360
    while diff < -180: diff += 360
    return diff

def classify_path_by_yaw(yaw_history):
    """
    根据 Yaw 变化判断意图
    """
    if not yaw_history or len(yaw_history) < 2:
        return "Unknown"
    
    start_yaw = yaw_history[0]
    end_yaw = yaw_history[-1]
    
    delta = get_angle_diff(start_yaw, end_yaw)
    
    # 判定逻辑 (减小=Left, 增大=Right)
    if abs(delta) <= STRAIGHT_YAW_THRESHOLD:
        return "Straight"
    elif delta < -STRAIGHT_YAW_THRESHOLD:
        return "Left"  # 角度减小
    else:
        return "Right" # 角度增大

def print_summary_table(title, stats_dict):
    """打印特定方向的统计表格 (已修复: 增加 Max Yaw)"""
    print(f"\n>>> 统计表格: {title} <<<")
    # 加长分割线
    print("-" * 110)
    # 增加 Max Yaw(°) 列
    print(f"{'Step':<5} {'Lat(ms)':<8} | {'Mean Pos(m)':<12} {'Max Pos(m)':<12} | {'Mean Yaw(°)':<12} {'Max Yaw(°)':<12} | {'Count'}")
    print("-" * 110)
    
    steps = sorted(stats_dict.keys())
    if not steps:
        print(" (无数据)")
        return

    for step in steps:
        data = stats_dict[step]
        all_pos_errs = []
        all_yaw_errs = []
        for veh_errs in data:
            all_pos_errs.extend(veh_errs['pos'])
            all_yaw_errs.extend(veh_errs['yaw'])
            
        if not all_pos_errs:
            continue
            
        mean_pos = np.mean(all_pos_errs)
        max_pos = np.max(all_pos_errs)
        mean_yaw = np.mean(all_yaw_errs)
        
        # [新增] 计算最大角度误差
        max_yaw = np.max(all_yaw_errs)
        
        time_ms = step * TIME_INTERVAL_MS
        
        print(f"{step:<5} {time_ms:<8} | "
              f"{mean_pos:.4f}       {max_pos:.4f}       | "
              f"{mean_yaw:.4f}       {max_yaw:.4f}       | "
              f"{len(all_pos_errs)}")
    print("=" * 110)

def main():
    if not os.path.exists(INPUT_ROOT):
        print(f"错误: 找不到目录 {INPUT_ROOT}")
        return

    print(f"加载预测记录: {PRED_JSON_PATH}")
    with open(PRED_JSON_PATH, 'r') as f:
        predictions = json.load(f)

    # 1. 数据容器
    vehicle_data = {}
    vehicle_yaw_trace = {}
    
    sorted_frames = sorted(predictions.keys(), key=lambda x: int(os.path.basename(x)))

    print("正在计算误差并提取轨迹...")
    for frame_key in tqdm(sorted_frames):
        vehicles_pred = predictions[frame_key]
        try:
            current_frame_id = int(os.path.basename(frame_key))
        except ValueError: continue

        for vid, steps_data in vehicles_pred.items():
            vid_str = str(vid)
            
            if vid_str not in vehicle_data:
                vehicle_data[vid_str] = {}
                vehicle_yaw_trace[vid_str] = []

            for step_item in steps_data:
                step = step_item['step']
                if step not in vehicle_data[vid_str]:
                    vehicle_data[vid_str][step] = {'pos': [], 'yaw': []}

                pred_pos = np.array([step_item['x'], step_item['y']])
                pred_yaw_deg = np.degrees(step_item['yaw'])
                
                target_id = current_frame_id + (step * FILENAME_INCREMENT_PER_STEP)
                target_path = os.path.join(INPUT_ROOT, f"{target_id:06d}.yaml")
                
                gt_pos, gt_yaw_deg = get_gt_data(target_path, vid)
                
                if gt_pos is not None:
                    pos_err = np.linalg.norm(pred_pos - gt_pos)
                    yaw_err = calculate_yaw_error(pred_yaw_deg, gt_yaw_deg)
                    
                    vehicle_data[vid_str][step]['pos'].append(pos_err)
                    vehicle_data[vid_str][step]['yaw'].append(yaw_err)
                    
                    # 记录真实 Yaw
                    vehicle_yaw_trace[vid_str].append(gt_yaw_deg)

    # 2. 分类汇总容器
    summary_stats = {
        'Straight': {},
        'Left': {},
        'Right': {},
        'Unknown': {}
    }

    csv_rows = []
    dir_counts = {'Straight': 0, 'Left': 0, 'Right': 0, 'Unknown': 0}
    
    print("\n正在分类并汇总统计...")
    for vid, steps_info in vehicle_data.items():
        # 判断路径类型
        path_type = classify_path_by_yaw(vehicle_yaw_trace.get(vid, []))

        # 累加计数
        if path_type in dir_counts:
            dir_counts[path_type] += 1
        
        for step, errs in steps_info.items():
            if not errs['pos']: continue
            
            # CSV 行
            mean_pos = np.mean(errs['pos'])
            max_pos = np.max(errs['pos'])
            std_pos = np.std(errs['pos'])
            mean_yaw = np.mean(errs['yaw'])
            max_yaw = np.max(errs['yaw'])
            
            csv_rows.append([
                vid, path_type, step, step*TIME_INTERVAL_MS,
                f"{mean_pos:.4f}", f"{max_pos:.4f}", f"{std_pos:.4f}", 
                f"{mean_yaw:.4f}", f"{max_yaw:.4f}", # CSV 中已经包含了
                len(errs['pos'])
            ])
            
            # 汇总容器
            if step not in summary_stats[path_type]:
                summary_stats[path_type][step] = []
            
            summary_stats[path_type][step].append(errs)

    # 3. 写入 CSV
    header = ["Vehicle_ID", "Direction", "Step", "Latency_ms", 
              "Mean_Pos_Err", "Max_Pos_Err", "Std_Pos_Err", 
              "Mean_Yaw_Err", "Max_Yaw_Err", "Count"]
    
    csv_rows.sort(key=lambda x: (x[1], x[0], int(x[2])))
    
    with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)
    print(f"CSV 已保存至: {OUTPUT_CSV_PATH}")
    print(f"\n共处理了{len(vehicle_data)}辆车的数据，其中直行{dir_counts['Straight']}辆，左转{dir_counts['Left']}辆，右转{dir_counts['Right']}辆")

    # 4. 打印统计表
    print_summary_table("Straight (直行)", summary_stats['Straight'])
    print_summary_table("Left (左转 - Yaw减小)", summary_stats['Left'])
    print_summary_table("Right (右转 - Yaw增大)", summary_stats['Right'])

if __name__ == "__main__":
    main()