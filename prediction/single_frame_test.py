import yaml
import numpy as np
import os

# === 请修改这里的路径 ===
ROOT_DIR = "/home/step/data/pcPredict_data/single_line_clean/scene/test_1/-1"
FRAME_CURRENT = "000060.yaml"
FRAME_FUTURE = "000062.yaml"  # 50ms后
# ======================

def deg2rad(deg):
    return deg * np.pi / 180.0

def analyze_prediction(sign_flip=False):
    path_cur = os.path.join(ROOT_DIR, FRAME_CURRENT)
    path_fut = os.path.join(ROOT_DIR, FRAME_FUTURE)

    with open(path_cur, 'r') as f: data_cur = yaml.safe_load(f)
    with open(path_fut, 'r') as f: data_fut = yaml.safe_load(f)

    print(f"\n=== 测试模式: Yaw符号取反 = {sign_flip} ===")
    
    # 取第一辆车做测试
    vid = list(data_cur['vehicles'].keys())[0]
    print(f"分析车辆 ID: {vid}")

    # 1. 获取真值 (Ground Truth)
    v_cur = data_cur['vehicles'][vid]
    v_fut = data_fut['vehicles'][vid]
    
    pos_cur = np.array(v_cur['location'][:2])
    pos_fut = np.array(v_fut['location'][:2])
    
    gt_delta = pos_fut - pos_cur
    gt_dist = np.linalg.norm(gt_delta)
    
    print(f"当前位置: {pos_cur}")
    print(f"未来位置: {pos_fut}")
    print(f"真实位移向量 (GT Delta): [dx={gt_delta[0]:.4f}, dy={gt_delta[1]:.4f}]")
    print(f"真实移动距离: {gt_dist:.4f} m")

    # 2. 进行预测
    raw_yaw = float(v_cur['angle'][1])
    speed_kmh = float(v_cur['speed'])
    speed_ms = speed_kmh / 3.6
    
    # === 关键点：测试不同的 Yaw 处理 ===
    if sign_flip:
        yaw_rad = -deg2rad(raw_yaw) # 你现在的代码
    else:
        yaw_rad = deg2rad(raw_yaw)  # 原始代码
        
    dt = 0.05
    pred_dx = speed_ms * np.cos(yaw_rad) * dt
    pred_dy = speed_ms * np.sin(yaw_rad) * dt
    
    pred_pos = pos_cur + np.array([pred_dx, pred_dy])
    
    # 3. 计算误差
    error_vec = pred_pos - pos_fut
    error_dist = np.linalg.norm(error_vec)
    
    print(f"预测位移向量 (Pred Delta): [dx={pred_dx:.4f}, dy={pred_dy:.4f}]")
    print(f"预测误差向量 (Error Vec): [diff_x={error_vec[0]:.4f}, diff_y={error_vec[1]:.4f}]")
    print(f"最终误差距离: {error_dist:.4f} m")
    
    if error_dist < 0.1:
        print(">>> 结论: 此配置正确！ <<<")
    else:
        print(">>> 结论: 此配置错误。 <<<")

if __name__ == "__main__":
    # 测试方案 A: 不加负号 (原汁原味)
    analyze_prediction(sign_flip=False)
    
    # 测试方案 B: 加负号 (你现在的代码)
    analyze_prediction(sign_flip=True)