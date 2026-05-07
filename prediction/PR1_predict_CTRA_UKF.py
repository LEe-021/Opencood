import os
import yaml
import numpy as np
import json
import glob
import time
from collections import deque
from tqdm import tqdm
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints

# ================= 配置 =================
INPUT_ROOT = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1_noisy/-1_noisy" # 指向加噪数据
OUTPUT_FILE = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1_noisy/prediction_registry_ctra_ukf.json"
HISTORY_LEN = 10   
PRED_STEPS = 6     
DT = 0.05          
# =======================================

def deg2rad(deg):
    return deg * np.pi / 180.0

def normalize_angle(x):
    """仅用于计算偏差或最后输出：将角度约束到 [-pi, pi]"""
    return (x + np.pi) % (2 * np.pi) - np.pi

def ctra_state_transition(x, dt):
    """
    CTRA 状态转移函数 f(x)
    State: [px, py, v, yaw, yaw_rate, a]
    允许 yaw 无限增长，保持连续性。
    """
    px, py, v, yaw, yaw_rate, a = x
    
    # 防止除以 0：当角速度极小时，退化为 CA (恒加速直线) 模型
    if abs(yaw_rate) < 0.001:
        px_new = px + (v * dt + 0.5 * a * dt**2) * np.cos(yaw)
        py_new = py + (v * dt + 0.5 * a * dt**2) * np.sin(yaw)
        v_new = v + a * dt
        yaw_new = yaw
        # yaw_rate 和 a 保持不变
    else:
        # CTRA 复杂的积分推导公式
        px_new = px + (1.0 / yaw_rate**2) * (
            (v * yaw_rate + a * yaw_rate * dt) * np.sin(yaw + yaw_rate * dt) + 
            a * np.cos(yaw + yaw_rate * dt) - 
            v * yaw_rate * np.sin(yaw) - 
            a * np.cos(yaw)
        )
        
        py_new = py + (1.0 / yaw_rate**2) * (
            -(v * yaw_rate + a * yaw_rate * dt) * np.cos(yaw + yaw_rate * dt) + 
            a * np.sin(yaw + yaw_rate * dt) + 
            v * yaw_rate * np.cos(yaw) - 
            a * np.sin(yaw)
        )
        
        v_new = v + a * dt
        yaw_new = yaw + yaw_rate * dt

    return np.array([px_new, py_new, v_new, yaw_new, yaw_rate, a])

def measurement_function(x):
    """观测位置、速度、航向角 (无法直接观测角速度和加速度)"""
    return np.array([x[0], x[1], x[2], x[3]])

class VehicleState:
    def __init__(self, x, y, yaw, speed, timestamp):
        self.x = x
        self.y = y
        self.yaw = yaw 
        self.speed = speed
        self.timestamp = timestamp

class CTRAUKFPredictor:
    def __init__(self):
        # State dim = 6
        self.points = JulierSigmaPoints(n=6, kappa=0)
        
    def predict(self, history_states: list, steps: int):
        ukf = UnscentedKalmanFilter(
            dim_x=6, dim_z=4, dt=DT, 
            fx=ctra_state_transition, hx=measurement_function, 
            points=self.points
        )
        
        # === Q 矩阵 (过程噪声) ===
        # 加速度(a) 的引入让模型很容易对噪声过度反应，
        # 所以必须对 a 和 yaw_rate 的过程噪声进行严格限制
        ukf.Q = np.diag([
            0.02**2,             # px
            0.02**2,             # py
            0.1**2,              # v
            deg2rad(0.1)**2,     # yaw 
            0.05**2,             # yaw_rate (保持稳定)
            0.5**2               # a (允许一定的加速度突变)
        ])

        # === R 矩阵 (观测噪声) ===
        # 完美对齐注入的 0.2 级别高斯噪声
        ukf.R = np.diag([
            0.2**2,              
            0.2**2,              
            0.2**2,              
            deg2rad(0.2)**2      
        ])
        
        # === P 矩阵 (初始不确定性) ===
        ukf.P = np.diag([
            0.2**2,             
            0.2**2,             
            0.2**2,             
            deg2rad(0.2)**2,    
            0.1**2,              # yaw_rate 初始不确定性
            1.0**2               # 加速度的初始不确定性给大一点，让它去收敛
        ])

        for i, state in enumerate(history_states):
            if i == 0:
                # 聪明地初始化角速度和加速度
                init_yaw_rate = 0.0
                init_accel = 0.0
                if len(history_states) > 1:
                    init_yaw_rate = normalize_angle(history_states[1].yaw - history_states[0].yaw) / DT
                    init_accel = (history_states[1].speed - history_states[0].speed) / DT
                
                ukf.x = np.array([state.x, state.y, state.speed, state.yaw, init_yaw_rate, init_accel])
            else:
                ukf.predict()
                
                # 角度解卷 (Angle Unrolling)，防止协方差爆炸
                angle_diff = normalize_angle(state.yaw - ukf.x[3])
                unrolled_obs_yaw = ukf.x[3] + angle_diff
                
                z = np.array([state.x, state.y, state.speed, unrolled_obs_yaw])
                ukf.update(z)
        
        current_state = ukf.x.copy()

        predictions = []
        for i in range(1, steps + 1):
            t_pred = i * DT
            
            # 独立预测：直接从 T0 状态推演到 T+t_pred
            pred_state = ctra_state_transition(current_state, t_pred)
            
            predictions.append({
                "step": i,
                "time_offset": round(t_pred, 3),
                "x": float(pred_state[0]),
                "y": float(pred_state[1]),
                # 输出时折叠回 [-pi, pi]
                "yaw": float(normalize_angle(pred_state[3])), 
                "speed": float(pred_state[2]) # 速度会因加速度的存在而改变
            })
            
        return predictions

def load_frame_data(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    frame_vehicles = {}
    if 'vehicles' in data and data['vehicles'] is not None:
        for vid, vinfo in data['vehicles'].items():
            loc = vinfo['location']
            angle = vinfo['angle']
            yaw_rad = deg2rad(float(angle[1]))
            speed_mps = float(vinfo['speed']) / 3.6
            frame_vehicles[vid] = VehicleState(
                x=float(loc[0]), y=float(loc[1]), yaw=yaw_rad, speed=speed_mps, timestamp=0
            )
    return frame_vehicles

def main():
    yaml_files = sorted(glob.glob(os.path.join(INPUT_ROOT, "*.yaml"), recursive=True))
    sequences = {}
    for f in yaml_files:
        dirname = os.path.dirname(f)
        if dirname not in sequences: sequences[dirname] = []
        sequences[dirname].append(f)
    
    predictor = CTRAUKFPredictor()
    registry = {}
    
    total_predict_time = 0.0
    total_predict_count = 0
    
    print(f"开始处理 (Method: CTRA + UKF)...")

    for seq_path, files in sequences.items():
        vehicle_history = {}
        files.sort()
        
        for frame_idx, yaml_path in enumerate(tqdm(files, desc=f"Processing {os.path.basename(seq_path)}", leave=False)):
            frame_name = os.path.splitext(os.path.basename(yaml_path))[0]
            frame_key = os.path.join(os.path.relpath(seq_path, INPUT_ROOT), frame_name)
            
            current_vehicles = load_frame_data(yaml_path)
            registry[frame_key] = {}
            
            for vid, state in current_vehicles.items():
                if vid not in vehicle_history:
                    vehicle_history[vid] = deque(maxlen=HISTORY_LEN)
                vehicle_history[vid].append(state)
                
                t_start = time.perf_counter()
                future_preds = predictor.predict(list(vehicle_history[vid]), PRED_STEPS)
                t_end = time.perf_counter()
                
                total_predict_time += (t_end - t_start)
                total_predict_count += 1
                
                registry[frame_key][vid] = future_preds
            
            existing_vids = set(current_vehicles.keys())
            history_vids = set(vehicle_history.keys())
            for vid in history_vids - existing_vids:
                del vehicle_history[vid]

    print(f"预测完成，正在保存至 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
        
    print("="*50)
    if total_predict_count > 0:
        avg_time_ms = (total_predict_time / total_predict_count) * 1000.0
        print(f"统计完成 (CTRA+UKF):")
        print(f"总处理车辆次数: {total_predict_count}")
        print(f"平均单车预测耗时: {avg_time_ms:.4f} ms")
    else:
        print("未检测到有效预测数据。")
    print("="*50)
    print("Done.")

if __name__ == "__main__":
    main()