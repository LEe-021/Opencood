#目前效率较低,每次验证一条轨迹时，都会去磁盘里 open() 并读取 7 个 YAML 文件。
#如果有几千条轨迹，程序相当于反复读取了几十万次 YAML 文件，大部分时间都花在硬盘读写上了。

import json
import yaml
import os
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 原始真值数据路径 (带噪声或干净的真值目录均可)
INPUT_ROOT = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1/-1"

# 2. 基准预测文件 (仅用于获取需要统计的 (Frame, VID) 列表)
BASE_JSON_PATH = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1_noisy/prediction_registry_cv_kf.json"

# 3. 步长设置
FILENAME_INCREMENT_PER_STEP = 1

# 4. 场景分类阈值 (你可以在这里反复微调，观察输出的数量变化)
STATIC_SPEED_THRESH = 0.5   # 过滤静止车辆 (m/s)
SPEED_CHANGE_THRESH = 0.5   # 加减速阈值 (m/s) -> 约 1.8 km/h
YAW_CHANGE_THRESH = 1.0     # 转弯阈值 (度)
# ===========================================

def get_gt_data(yaml_path, vehicle_id):
    """读取 YAML 获取真值 (位置, 航向角, 速度)"""
    if not os.path.exists(yaml_path):
        return None, None, None
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'vehicles' not in data or data['vehicles'] is None:
        return None, None, None

    vinfo = None
    if vehicle_id in data['vehicles']:
        vinfo = data['vehicles'][vehicle_id]
    elif int(vehicle_id) in data['vehicles']:
        vinfo = data['vehicles'][int(vehicle_id)]
    elif str(vehicle_id) in data['vehicles']:
        vinfo = data['vehicles'][str(vehicle_id)]

    if vinfo is None:
        return None, None, None

    loc_np = np.array(vinfo['location'][:2])
    yaw_deg = float(vinfo['angle'][1])
    speed_mps = float(vinfo['speed']) / 3.6  
    
    return loc_np, yaw_deg, speed_mps

def analyze_trajectory(gt_states):
    """
    分析未来 6 帧的真值，返回类别以及变化量
    """
    speeds = [s['speed'] for s in gt_states]
    yaws_rad = [np.radians(s['yaw']) for s in gt_states]
    
    max_speed = max(speeds)
    speed_change = max(speeds) - min(speeds)
    
    unwrapped_yaws = np.unwrap(yaws_rad)
    yaw_change_deg = np.degrees(max(unwrapped_yaws) - min(unwrapped_yaws))
    
    # 判断标志
    is_static = max_speed < STATIC_SPEED_THRESH
    is_accel = speed_change >= SPEED_CHANGE_THRESH
    is_turn = yaw_change_deg >= YAW_CHANGE_THRESH
    
    # 类别归属
    if is_static:
        category = "Stationary"
    elif is_accel:
        category = "Accel_Decel"
    elif is_turn:
        category = "Turning"
    else:
        category = "Straight"
        
    return category, speed_change, yaw_change_deg, is_accel, is_turn

def main():
    if not os.path.exists(BASE_JSON_PATH):
        print(f"错误: 找不到基准 JSON 文件: {BASE_JSON_PATH}")
        return

    print("正在加载基准车辆列表...")
    with open(BASE_JSON_PATH, 'r') as f:
        base_preds = json.load(f)

    # 统计数据结构
    trajectory_counts = {
        "Straight": 0, 
        "Turning": 0, 
        "Accel_Decel": 0, 
        "Stationary": 0, 
        "Invalid": 0
    }
    
    # 用于收集 "既有加减速，又有转弯" 的复杂场景
    complex_scenarios = []

    print("\n开始扫描并分析场景分布...")
    
    for frame_key, vehicles_pred in tqdm(base_preds.items()):
        frame_name = os.path.basename(frame_key)
        try:
            current_frame_id = int(frame_name)
        except ValueError:
            continue

        for vid in vehicles_pred.keys():
            gt_future_states = []
            is_valid_sequence = True
            
            # 读取 T0 到 T6 的真值
            for step in range(0, 7): 
                target_frame_id = current_frame_id + (step * FILENAME_INCREMENT_PER_STEP)
                target_yaml = os.path.join(INPUT_ROOT, f"{target_frame_id:06d}.yaml")
                
                gt_pos, gt_yaw, gt_speed = get_gt_data(target_yaml, vid)
                if gt_pos is None:
                    is_valid_sequence = False
                    break
                gt_future_states.append({'pos': gt_pos, 'yaw': gt_yaw, 'speed': gt_speed})
                
            if not is_valid_sequence:
                trajectory_counts["Invalid"] += 1
                continue
                
            # 获取场景分类和变化量
            category, dv, dy, is_accel, is_turn = analyze_trajectory(gt_future_states)
            trajectory_counts[category] += 1
            
            # 专门收集 "加减速 + 转弯" 的复合困难场景 (虽然它们被归类在了 Accel_Decel 里)
            if not category == "Stationary" and is_accel and is_turn:
                # 定义困难度得分：速度变化量 * 角度变化量
                score = dv * dy 
                complex_scenarios.append({
                    'frame_id': current_frame_id,
                    'vid': vid,
                    'speed_change': dv,
                    'yaw_change': dy,
                    'score': score
                })

    # ================= 输出分布报告 =================
    print("\n" + "="*50)
    print("场景分类数据分布报告")
    print("="*50)
    total_valid = trajectory_counts['Straight'] + trajectory_counts['Turning'] + trajectory_counts['Accel_Decel']
    
    if total_valid > 0:
        print(f" 匀速直行 (Straight)    : {trajectory_counts['Straight']:<5} 占比: {trajectory_counts['Straight']/total_valid*100:.1f}%")
        print(f" 匀速转弯 (Turning)     : {trajectory_counts['Turning']:<5} 占比: {trajectory_counts['Turning']/total_valid*100:.1f}%")
        print(f" 加减速机动 (Accel_Decel): {trajectory_counts['Accel_Decel']:<5} 占比: {trajectory_counts['Accel_Decel']/total_valid*100:.1f}%")
    else:
        print("没有找到有效的轨迹数据！")
    print("-" * 50)
    print(f" [已忽略] 静止不动      : {trajectory_counts['Stationary']}")
    print(f" [已忽略] 帧数不完整    : {trajectory_counts['Invalid']}")
    print("="*50)

    # ================= 输出 Top 10 困难场景 =================
    print("\nTop 10 最剧烈的复合机动场景 (加减速 + 转弯)")
    print(" (非常适合用于论文中的可视化绘图和 IMM 优势展示)")
    print("-" * 75)
    print(f"{'排名':<4} | {'Frame ID':<10} | {'Vehicle ID':<12} | {'Δ Speed (m/s)':<15} | {'Δ Yaw (deg)':<15}")
    print("-" * 75)
    
    # 按困难度得分降序排序
    complex_scenarios.sort(key=lambda x: x['score'], reverse=True)
    
    for i, scene in enumerate(complex_scenarios[:10]):
        print(f"#{i+1:<3} | {scene['frame_id']:<10} | {scene['vid']:<12} | {scene['speed_change']:<15.4f} | {scene['yaw_change']:<15.4f}")
    
    if len(complex_scenarios) == 0:
        print("未找到同时满足加减速和转弯阈值的极端场景。建议降低判定阈值。")
    print("-" * 75)

if __name__ == "__main__":
    main()