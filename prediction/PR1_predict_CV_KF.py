import os
import yaml
import numpy as np
import json
import glob
import time
from collections import deque
from tqdm import tqdm

# ================= 配置 =================
INPUT_ROOT = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1_noisy/-1_noisy"  # 指向加噪数据
OUTPUT_FILE = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1_noisy/prediction_registry_cv_kf.json"
HISTORY_LEN = 10  # 历史窗口长度 (帧)
PRED_STEPS = 6   # 预测未来多少帧
DT = 0.05        # 两帧之间的时间间隔 (50ms)
# =======================================

def deg2rad(deg):
    return deg * np.pi / 180.0

def normalize_angle(x):
    """将角度标准化到 [-pi, pi]"""
    return (x + np.pi) % (2 * np.pi) - np.pi

class VehicleState:
    def __init__(self, x, y, yaw, speed, timestamp):
        self.x = x
        self.y = y
        self.yaw = yaw # rad
        self.speed = speed
        self.timestamp = timestamp

class KalmanFilter:
    """
    线性卡尔曼滤波实现 (针对 CV 模型 4维状态)
    State: [x, y, vx, vy]^T
    """
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros((4, 1))
        
        # 状态转移矩阵 F (4, 4) - CV模型
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 观测矩阵 H (4, 4)
        self.H = np.eye(4)
        
        # === 【核心更新 1】精准对齐观测噪声 R ===
        # 根据注入的高斯噪声：位置 0.2m，速度 0.2m/s
        # 注意：vx 和 vy 的噪声近似由速度 0.2m/s 和微小的偏航角噪声组合而来，这里直接给 0.2
        self.R = np.diag([
            0.2**2,  # x 观测噪声
            0.2**2,  # y 观测噪声
            0.2**2,  # vx 观测噪声
            0.2**2   # vy 观测噪声
        ])
        
        # === 【核心更新 2】收紧过程噪声 Q ===
        # 我们坚信车辆在 50ms 内保持匀速直线运动，不允许速度发生剧烈突变
        self.Q = np.diag([
            0.02**2, # x 的微小模型误差
            0.02**2, # y 的微小模型误差
            0.1**2,  # vx 加速度扰动
            0.1**2   # vy 加速度扰动
        ])
        
        # === 【核心更新 3】对齐初始不确定性 P ===
        # 第一帧的状态直接来自观测，所以初始 P 矩阵应等于观测噪声 R 矩阵
        self.P = np.diag([
            0.2**2,
            0.2**2,
            0.2**2,
            0.2**2
        ])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((4, 4))
            
        self.x = self.x + K @ y
        I = np.eye(self.x.shape[0])
        self.P = (I - K @ self.H) @ self.P

class Predictor:
    def predict(self, history_states, steps):
        raise NotImplementedError

class CVKFPredictor(Predictor):
    def predict(self, history_states: list, steps: int):
        kf = KalmanFilter(dt=DT)
        
        for i, state in enumerate(history_states):
            vx = state.speed * np.cos(state.yaw)
            vy = state.speed * np.sin(state.yaw)
            z = np.array([[state.x], [state.y], [vx], [vy]])
            
            if i == 0:
                kf.x = z
                # P 矩阵已经在初始化时设置好，无需再覆盖
            else:
                kf.predict()
                kf.update(z)
        
        current_filtered_state = kf.x.copy()
        
        predictions = []
        x_0 = current_filtered_state[0, 0]
        y_0 = current_filtered_state[1, 0]
        vx_0 = current_filtered_state[2, 0]
        vy_0 = current_filtered_state[3, 0]
        
        for i in range(1, steps + 1):
            t_pred = i * DT  
            
            pred_x = x_0 + vx_0 * t_pred
            pred_y = y_0 + vy_0 * t_pred
            
            pred_vx = vx_0
            pred_vy = vy_0
            pred_speed = np.sqrt(pred_vx**2 + pred_vy**2)
            
            if pred_speed > 0.1:
                pred_yaw = np.arctan2(pred_vy, pred_vx)
            else:
                pred_yaw = history_states[-1].yaw

            predictions.append({
                "step": i,
                "time_offset": round(t_pred, 3),
                "x": float(pred_x),
                "y": float(pred_y),
                "yaw": float(normalize_angle(pred_yaw)), # 确保输出标准范围
                "speed": float(pred_speed)
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
        if dirname not in sequences:
            sequences[dirname] = []
        sequences[dirname].append(f)
    
    predictor = CVKFPredictor()
    registry = {} 

    total_predict_time = 0.0
    total_predict_count = 0
    
    print(f"开始处理 (Method: CV + Kalman Filter)...")

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
        print(f"统计完成 (CV+KF):")
        print(f"总处理车辆次数: {total_predict_count}")
        print(f"平均单车预测耗时: {avg_time_ms:.4f} ms")
    else:
        print("未检测到有效预测数据。")
    print("="*50)
    print("Done.")

if __name__ == "__main__":
    main()