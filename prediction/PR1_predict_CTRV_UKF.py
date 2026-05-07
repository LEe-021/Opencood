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
INPUT_ROOT = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1_noisy/-1_noisy"
OUTPUT_FILE = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1_noisy/prediction_registry_ctrv_ukf.json"
HISTORY_LEN = 10   # 推荐 10 帧 (500ms) 以稳定 Yaw Rate 估计
PRED_STEPS = 6     # 预测未来 6 帧 (300ms)
DT = 0.05          # 50ms
# =======================================

class VehicleState:
    def __init__(self, x, y, yaw, speed, timestamp):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.speed = speed
        self.timestamp = timestamp

# ================= UKF 核心逻辑 =================

def deg2rad(deg):
    return deg * np.pi / 180.0

def normalize_angle(x):
    """仅用于计算偏差或输出时：将角度约束到 [-pi, pi]"""
    return (x + np.pi) % (2 * np.pi) - np.pi

def ctrv_state_transition(x, dt):
    """
    【核心修复】: 去除所有 normalize_angle！
    允许 yaw (x[3]) 无限增长，保持状态空间的绝对连续性。
    """
    px, py, v, yaw, yaw_rate = x
    
    if abs(yaw_rate) < 0.001:
        px_new = px + v * np.cos(yaw) * dt
        py_new = py + v * np.sin(yaw) * dt
        yaw_new = yaw
    else:
        px_new = px + (v / yaw_rate) * (np.sin(yaw + yaw_rate * dt) - np.sin(yaw))
        py_new = py + (v / yaw_rate) * (np.cos(yaw) - np.cos(yaw + yaw_rate * dt))
        yaw_new = yaw + yaw_rate * dt
        
    return np.array([px_new, py_new, v, yaw_new, yaw_rate])

def measurement_function(x):
    return np.array([x[0], x[1], x[2], x[3]])

class CTRVUKFPredictor:
    def __init__(self):
        # 使用 JulierSigmaPoints，权重绝对为正，彻底杜绝协方差崩溃
        self.points = JulierSigmaPoints(n=5, kappa=0)
        
    def predict(self, history_states: list, steps: int):
        # 不再需要自定义 x_mean_fn 等，因为状态空间现在是连续的
        ukf = UnscentedKalmanFilter(
            dim_x=5, dim_z=4, dt=DT, 
            fx=ctrv_state_transition, hx=measurement_function, 
            points=self.points
        )
        
        # === Q 矩阵 (过程噪声) ===
        ukf.Q = np.diag([
            0.02**2,             # px
            0.02**2,             # py
            0.1**2,              # v
            deg2rad(0.1)**2,     # yaw (非常小，相信CTRV物理规律)
            0.05**2              # yaw_rate
        ])

        # === R 矩阵 (观测噪声) ===
        # 精准对齐你注入的高斯噪声
        ukf.R = np.diag([
            0.2**2,              
            0.2**2,              
            0.2**2,              
            deg2rad(0.2)**2      
        ])
        
        # === P 矩阵 (初始状态不确定性) ===
        ukf.P = np.diag([
            0.2**2,             
            0.2**2,             
            0.2**2,             
            deg2rad(0.2)**2,    
            0.1**2               # 限制初始的 yaw_rate 不确定性
        ])

        for i, state in enumerate(history_states):
            if i == 0:
                # 给滤波器一个更聪明的初始猜测，而不是盲猜 0
                init_yaw_rate = 0.0
                if len(history_states) > 1:
                    diff = normalize_angle(history_states[1].yaw - history_states[0].yaw)
                    init_yaw_rate = diff / DT
                ukf.x = np.array([state.x, state.y, state.speed, state.yaw, init_yaw_rate])
            else:
                ukf.predict()
                
                # 【最核心的魔法】：观测角度解卷 (Unrolling)
                # 计算新观测角度与当前状态角度的“最短夹角”
                angle_diff = normalize_angle(state.yaw - ukf.x[3])
                # 将这个夹角加回当前状态，使观测值与状态处于同一个欧几里得平面
                unrolled_obs_yaw = ukf.x[3] + angle_diff
                
                z = np.array([state.x, state.y, state.speed, unrolled_obs_yaw])
                ukf.update(z)
        
        current_state = ukf.x.copy()

        predictions = []
        for i in range(1, steps + 1):
            t_pred = i * DT
            pred_state = ctrv_state_transition(current_state, t_pred)
            
            predictions.append({
                "step": i,
                "time_offset": round(t_pred, 3),
                "x": float(pred_state[0]),
                "y": float(pred_state[1]),
                # 【最后一步】：仅在输出 JSON 时，把角度折叠回 [-pi, pi]
                "yaw": float(normalize_angle(pred_state[3])), 
                "speed": float(pred_state[2])
            })
            
        return predictions


def load_frame_data(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    frame_vehicles = {}
    if 'vehicles' in data and data['vehicles'] is not None:
        for vid, vinfo in data['vehicles'].items():
            loc = vinfo['location']
            angle = vinfo['angle'] # [pitch, yaw, roll]
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
    
    # 实例化预测器
    predictor = CTRVUKFPredictor()
    registry = {}
    
    # 统计变量
    total_predict_time = 0.0
    total_predict_count = 0
    
    print(f"开始处理 (Method: CTRV + UKF)...")

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
                
                # === 核心处理与计时 ===
                t_start = time.perf_counter()
                
                # 执行预测
                future_preds = predictor.predict(list(vehicle_history[vid]), PRED_STEPS)
                
                t_end = time.perf_counter()
                total_predict_time += (t_end - t_start)
                total_predict_count += 1
                # ===================
                
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
        print(f"统计完成 (CTRV+UKF):")
        print(f"总处理车辆次数: {total_predict_count}")
        print(f"平均单车预测耗时: {avg_time_ms:.4f} ms")
    else:
        print("未检测到有效预测数据。")
    print("="*50)
    print("Done.")

if __name__ == "__main__":
    main()