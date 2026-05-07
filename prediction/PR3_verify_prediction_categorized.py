import json
import yaml
import os
import numpy as np
import random
import glob
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_ROOT = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1/-1"
BASE_DIR = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1_noisy"
JSON_PATHS = {
    "CV": os.path.join(BASE_DIR, "prediction_registry_cv_kf.json"),
    "CTRV": os.path.join(BASE_DIR, "prediction_registry_ctrv_ukf.json"),
    "CTRA": os.path.join(BASE_DIR, "prediction_registry_ctra_ukf.json"),
    "IMM": os.path.join(BASE_DIR, "prediction_registry_imm_ukf.json")
}

FILENAME_INCREMENT_PER_STEP = 1
SAMPLE_LIMIT = 300          # 每个类别的最大采样数量

STATIC_SPEED_THRESH = 0.5   
SPEED_CHANGE_THRESH = 0.5   
YAW_CHANGE_THRESH = 1.0     

# 固定随机种子，保证每次运行采样的300个样本是一样的，方便复现
random.seed(42) 
# ===========================================

def calculate_yaw_error(pred_deg, gt_deg):
    diff = abs(pred_deg - gt_deg)
    diff = diff % 360.0 
    return min(diff, 360.0 - diff)

def classify_scenario(gt_states):
    speeds = [s['speed'] for s in gt_states]
    yaws_rad = [np.radians(s['yaw']) for s in gt_states]
    
    if max(speeds) < STATIC_SPEED_THRESH:
        return "Stationary"
        
    speed_change = max(speeds) - min(speeds)
    if speed_change >= SPEED_CHANGE_THRESH:
        return "Accel_Decel"
        
    unwrapped_yaws = np.unwrap(yaws_rad)
    yaw_change_deg = np.degrees(max(unwrapped_yaws) - min(unwrapped_yaws))
    
    if yaw_change_deg >= YAW_CHANGE_THRESH:
        return "Turning"
        
    return "Straight"

def build_gt_cache():
    """将所有 YAML 真值一次性读入内存，突破 I/O 瓶颈"""
    print("正在构建真值内存缓存 (GT Cache)，请稍候...")
    gt_cache = {}
    yaml_files = glob.glob(os.path.join(INPUT_ROOT, "*.yaml"))
    
    for yaml_path in tqdm(yaml_files, desc="Loading YAMLs"):
        frame_name = os.path.splitext(os.path.basename(yaml_path))[0]
        try:
            frame_id = int(frame_name)
        except ValueError:
            continue
            
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        gt_cache[frame_id] = {}
        if data and 'vehicles' in data and data['vehicles']:
            for vid, vinfo in data['vehicles'].items():
                loc_np = np.array(vinfo['location'][:2])
                yaw_deg = float(vinfo['angle'][1])
                speed_mps = float(vinfo['speed']) / 3.6
                
                # 统一转为 string 类型的 ID 方便匹配
                gt_cache[frame_id][str(vid)] = {
                    'pos': loc_np, 
                    'yaw': yaw_deg, 
                    'speed': speed_mps
                }
    print(f"缓存构建完成，共加载 {len(gt_cache)} 帧数据。\n")
    return gt_cache

def main():
    gt_cache = build_gt_cache()
    
    predictions = {}
    for model_name, path in JSON_PATHS.items():
        print(f"加载 {model_name} 预测结果...")
        with open(path, 'r') as f:
            predictions[model_name] = json.load(f)

    categories = ["Straight", "Turning", "Accel_Decel"]
    models = list(JSON_PATHS.keys())
    
    # 场景池：存放各分类下所有合法的 (frame_key, vid, gt_future_states)
    scenario_pools = {cat: [] for cat in categories}

    print("\n阶段 1/2: 快速扫描场景并分类...")
    base_preds = predictions["CV"]
    
    for frame_key, vehicles_pred in tqdm(base_preds.items(), desc="Classifying"):
        frame_name = os.path.basename(frame_key)
        try:
            current_frame_id = int(frame_name)
        except ValueError:
            continue

        for vid in vehicles_pred.keys():
            gt_future_states = []
            is_valid_sequence = True
            
            # 从内存缓存中快速读取 7 帧真值
            for step in range(0, 7): 
                target_frame_id = current_frame_id + (step * FILENAME_INCREMENT_PER_STEP)
                
                if target_frame_id in gt_cache and str(vid) in gt_cache[target_frame_id]:
                    gt_future_states.append(gt_cache[target_frame_id][str(vid)])
                else:
                    is_valid_sequence = False
                    break
                
            if not is_valid_sequence:
                continue
                
            scenario_tag = classify_scenario(gt_future_states)
            
            if scenario_tag in scenario_pools:
                # 确保四个模型都有这个预测结果
                if all(str(vid) in predictions[m].get(frame_key, {}) for m in models):
                    scenario_pools[scenario_tag].append((frame_key, str(vid), gt_future_states))

    # 执行随机采样 (限制最多 SAMPLE_LIMIT 条)
    sampled_data = {cat: [] for cat in categories}
    for cat in categories:
        pool_size = len(scenario_pools[cat])
        sample_size = min(SAMPLE_LIMIT, pool_size)
        sampled_data[cat] = random.sample(scenario_pools[cat], sample_size)
        print(f"类别 [{cat:<12}] : 找到 {pool_size:<4} 条有效轨迹，抽取 {sample_size:<4} 条参与评估。")

    # 阶段 2：计算误差
    print("\n阶段 2/2: 计算误差并生成报告...")
    stats = {cat: {mod: {step: {'pos': [], 'yaw': []} for step in range(1, 7)} for mod in models} for cat in categories}
    
    for cat in categories:
        for frame_key, vid, gt_future_states in sampled_data[cat]:
            for model_name in models:
                pred_trajectory = predictions[model_name][frame_key][vid]
                for step_item in pred_trajectory:
                    step = step_item['step']
                    pred_pos = np.array([step_item['x'], step_item['y']])
                    pred_yaw_deg = np.degrees(step_item['yaw'])
                    
                    gt_pos = gt_future_states[step]['pos']
                    gt_yaw_deg = gt_future_states[step]['yaw']
                    
                    pos_err = np.linalg.norm(pred_pos - gt_pos)
                    yaw_err = calculate_yaw_error(pred_yaw_deg, gt_yaw_deg)
                    
                    stats[cat][model_name][step]['pos'].append(pos_err)
                    stats[cat][model_name][step]['yaw'].append(yaw_err)

    # 打印最终表格
    for cat in categories:
        actual_samples = len(sampled_data[cat])
        if actual_samples == 0:
            continue
            
        print(f"\n\n>>>>>>>>>> 场景: {cat} (样本数: {actual_samples}) <<<<<<<<<<")
        header = f"{'Model':<8} | " + " | ".join([f"Step {s}({s*50}ms)" for s in range(1, 7)])
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        print("[平均位置误差 Mean Pos (m)]")
        for model in models:
            row_str = f"{model:<8} | "
            for step in range(1, 7):
                pos_errs = stats[cat][model][step]['pos']
                row_str += f"{np.mean(pos_errs):.4f} m{'':<2} | "
            print(row_str)
            
        print("\n[平均航向角误差 Mean Yaw (°)]")
        for model in models:
            row_str = f"{model:<8} | "
            for step in range(1, 7):
                yaw_errs = stats[cat][model][step]['yaw']
                row_str += f"{np.mean(yaw_errs):.4f} °{'':<2} | "
            print(row_str)

if __name__ == "__main__":
    main()