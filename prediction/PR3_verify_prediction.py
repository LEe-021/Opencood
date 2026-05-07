import json
import yaml
import os
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 原始数据路径 (路侧 -1 文件夹)
INPUT_ROOT = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1/-1"

# 2. 预测结果 JSON 路径
PRED_JSON_PATH = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1_noisy/prediction_registry_imm_ukf.json"

# 3. 文件名步长
#    文件名 60 -> 62 -> 64
#    Step 1 (100ms) = 间隔 2
FILENAME_INCREMENT_PER_STEP = 1
# ===========================================

def get_gt_data(yaml_path, vehicle_id):
    """
    读取指定 YAML 文件，获取特定车辆的真实位置和航向角
    修复点：兼容 ID 为 int 的情况
    """
    if not os.path.exists(yaml_path):
        return None, None
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'vehicles' not in data or data['vehicles'] is None:
        return None, None

    # === 关键修复逻辑 ===
    # 尝试多种类型匹配，确保万无一失
    vinfo = None
    
    # 1. 尝试直接匹配 (可能是 str)
    if vehicle_id in data['vehicles']:
        vinfo = data['vehicles'][vehicle_id]
    
    # 2. 尝试转 int 匹配 (针对 YAML 解析为 int 的情况)
    elif int(vehicle_id) in data['vehicles']:
        vinfo = data['vehicles'][int(vehicle_id)]
        
    # 3. 尝试转 str 匹配 (防守性编程)
    elif str(vehicle_id) in data['vehicles']:
        vinfo = data['vehicles'][str(vehicle_id)]

    if vinfo is None:
        return None, None

    # === 数据提取 ===
    loc = vinfo['location']
    loc_np = np.array(loc[:2]) # [x, y]
    
    # 获取航向角 (Carla: angle[1] is yaw)
    yaw_deg = float(vinfo['angle'][1])
    
    return loc_np, yaw_deg

def calculate_yaw_error(pred_deg, gt_deg):
    """
    计算角度误差，处理 0/360 周期性问题
    例如：预测 359°，真值 1°，误差应该是 2°
    """
    diff = abs(pred_deg - gt_deg)
    diff = diff % 360.0 
    error = min(diff, 360.0 - diff)
    return error

def main():
    if not os.path.exists(INPUT_ROOT):
        print(f"错误: 找不到目录 {INPUT_ROOT}")
        return

    print(f"加载预测记录: {PRED_JSON_PATH}")
    with open(PRED_JSON_PATH, 'r') as f:
        predictions = json.load(f)

    # 存储误差统计结果
    stats = {} 
    valid_count = 0

    print(f"开始验证... (Step 1 -> File+{FILENAME_INCREMENT_PER_STEP})")
    
    for frame_key, vehicles_pred in tqdm(predictions.items()):
        frame_name = os.path.basename(frame_key)
        try:
            current_frame_id = int(frame_name)
        except ValueError:
            continue

        for vid, steps_data in vehicles_pred.items():
            for step_item in steps_data:
                step = step_item['step']
                
                # 获取预测值
                pred_pos = np.array([step_item['x'], step_item['y']])
                pred_yaw_rad = step_item['yaw']
                # 弧度转度
                pred_yaw_deg = np.degrees(pred_yaw_rad)
                
                # 获取真值
                target_frame_id = current_frame_id + (step * FILENAME_INCREMENT_PER_STEP)
                target_frame_name = f"{target_frame_id:06d}.yaml"
                target_yaml_path = os.path.join(INPUT_ROOT, target_frame_name)
                
                gt_pos, gt_yaw_deg = get_gt_data(target_yaml_path, vid)
                
                if gt_pos is not None:
                    # 计算误差
                    pos_err = np.linalg.norm(pred_pos - gt_pos)
                    yaw_err = calculate_yaw_error(pred_yaw_deg, gt_yaw_deg)
                    
                    if step not in stats:
                        stats[step] = {'pos': [], 'yaw': []}
                    
                    stats[step]['pos'].append(pos_err)
                    stats[step]['yaw'].append(yaw_err)
                    valid_count += 1

    if valid_count == 0:
        print("\n依然没有有效对比，请检查 prediction_registry.json 里的 ID 格式是否异常。")
        return

    # ================= 输出结果表格 (已修改为方差) =================
    print("\n" + "="*95)
    # 修改表头：Max -> Var
    print(f"{'Step':<5} {'Lat(ms)':<8} | {'Mean Pos(m)':<12} {'Var Pos':<12} | {'Mean Yaw(°)':<12} {'Var Yaw':<12} | {'Count'}")
    print("-" * 95)
    
    steps = sorted(stats.keys())
    
    for step in steps:
        pos_errors = np.array(stats[step]['pos'])
        yaw_errors = np.array(stats[step]['yaw'])
        
        time_ms = step * 50 
        
        # 修改计算逻辑：np.max -> np.var
        print(f"{step:<5} {time_ms:<8} | "
              f"{np.mean(pos_errors):.4f}        {np.var(pos_errors):.4f}        | "
              f"{np.mean(yaw_errors):.4f}        {np.var(yaw_errors):.4f}        | "
              f"{len(pos_errors)}")
    
    print("="*95)

if __name__ == "__main__":
    main()