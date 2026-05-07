import os
import yaml
import numpy as np
import json
import glob
import time
from collections import deque
from tqdm import tqdm
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints, IMMEstimator

# ================= 配置 =================
INPUT_ROOT = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1_noisy/-1_noisy" # 指向加噪数据
OUTPUT_FILE = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1_noisy/prediction_registry_imm_ukf.json"
HISTORY_LEN = 10   
PRED_STEPS = 6     
DT = 0.05          
# =======================================

def deg2rad(deg):
    return deg * np.pi / 180.0

def normalize_angle(x):
    """仅用于计算偏差或最后输出"""
    return (x + np.pi) % (2 * np.pi) - np.pi

# ================= 1. 定义三种模型的物理转移函数 =================
# 统一使用 6D 状态: [px, py, v, yaw, yaw_rate, a]

def cv_transition(x, dt):
    """CV模型: 匀速直线运动。无视角速度和加速度"""
    px, py, v, yaw, yaw_rate, a = x
    px_new = px + v * np.cos(yaw) * dt
    py_new = py + v * np.sin(yaw) * dt
    # 物理量透传，保证 UKF 协方差矩阵维度一致且不崩溃
    return np.array([px_new, py_new, v, yaw, yaw_rate, a])

def ctrv_transition(x, dt):
    """CTRV模型: 恒定转弯率和速度。无视加速度"""
    px, py, v, yaw, yaw_rate, a = x
    if abs(yaw_rate) < 0.001:
        px_new = px + v * np.cos(yaw) * dt
        py_new = py + v * np.sin(yaw) * dt
        yaw_new = yaw
    else:
        px_new = px + (v / yaw_rate) * (np.sin(yaw + yaw_rate * dt) - np.sin(yaw))
        py_new = py + (v / yaw_rate) * (np.cos(yaw) - np.cos(yaw + yaw_rate * dt))
        yaw_new = yaw + yaw_rate * dt
    return np.array([px_new, py_new, v, yaw_new, yaw_rate, a])

def ctra_transition(x, dt):
    """CTRA模型: 恒定转弯率和加速度"""
    px, py, v, yaw, yaw_rate, a = x
    if abs(yaw_rate) < 0.001:
        px_new = px + (v * dt + 0.5 * a * dt**2) * np.cos(yaw)
        py_new = py + (v * dt + 0.5 * a * dt**2) * np.sin(yaw)
        v_new = v + a * dt
        yaw_new = yaw
    else:
        px_new = px + (1.0 / yaw_rate**2) * (
            (v * yaw_rate + a * yaw_rate * dt) * np.sin(yaw + yaw_rate * dt) + 
            a * np.cos(yaw + yaw_rate * dt) - v * yaw_rate * np.sin(yaw) - a * np.cos(yaw)
        )
        py_new = py + (1.0 / yaw_rate**2) * (
            -(v * yaw_rate + a * yaw_rate * dt) * np.cos(yaw + yaw_rate * dt) + 
            a * np.sin(yaw + yaw_rate * dt) + v * yaw_rate * np.cos(yaw) - a * np.sin(yaw)
        )
        v_new = v + a * dt
        yaw_new = yaw + yaw_rate * dt
    return np.array([px_new, py_new, v_new, yaw_new, yaw_rate, a])

def measurement_function(x):
    """所有模型共享相同的观测映射"""
    return np.array([x[0], x[1], x[2], x[3]])

# =========================================================

class VehicleState:
    def __init__(self, x, y, yaw, speed, timestamp):
        self.x = x
        self.y = y
        self.yaw = yaw 
        self.speed = speed
        self.timestamp = timestamp

class IMMUKFPredictor:
    def __init__(self):
        # 统一的 Sigma 点采样器
        self.points = JulierSigmaPoints(n=6, kappa=0)
        
        # 共享的 R 和 P 矩阵设定
        self.R_shared = np.diag([0.2**2, 0.2**2, 0.2**2, deg2rad(0.2)**2])
        self.P_shared = np.diag([0.2**2, 0.2**2, 0.2**2, deg2rad(0.2)**2, 0.1**2, 0.5**2])

    def create_imm_estimator(self):
        """为每辆车创建一个独立的 IMM 估计器实例"""
        
        # 1. 创建 CV 滤波器 (直行专家)
        ukf_cv = UnscentedKalmanFilter(dim_x=6, dim_z=4, dt=DT, fx=cv_transition, hx=measurement_function, points=self.points)
        # CV 的 Q 矩阵极其不信任角速度和加速度，锁死在直线运动
        ukf_cv.Q = np.diag([0.02**2, 0.02**2, 0.1**2, deg2rad(0.1)**2, 1e-5, 1e-5])
        ukf_cv.R = self.R_shared.copy()
        
        # 2. 创建 CTRV 滤波器 (匀速转弯专家)
        ukf_ctrv = UnscentedKalmanFilter(dim_x=6, dim_z=4, dt=DT, fx=ctrv_transition, hx=measurement_function, points=self.points)
        # CTRV 允许角速度变化，但不允许加速度变化
        ukf_ctrv.Q = np.diag([0.02**2, 0.02**2, 0.1**2, deg2rad(0.1)**2, 0.05**2, 1e-5])
        ukf_ctrv.R = self.R_shared.copy()
        
        # 3. 创建 CTRA 滤波器 (加减速转弯专家)
        ukf_ctra = UnscentedKalmanFilter(dim_x=6, dim_z=4, dt=DT, fx=ctra_transition, hx=measurement_function, points=self.points)
        # CTRA 允许角速度和加速度同时变化
        ukf_ctra.Q = np.diag([0.02**2, 0.02**2, 0.1**2, deg2rad(0.1)**2, 0.05**2, 0.5**2])
        ukf_ctra.R = self.R_shared.copy()

        filters = [ukf_cv, ukf_ctrv, ukf_ctra]
        
        # IMM 模式初始概率：假设大家机会均等，CTRV 稍微大一点
        mu = np.array([0.3, 0.4, 0.3])
        
        # 马尔可夫状态转移矩阵 M
        # [i, j] 表示从模式 i 转移到模式 j 的概率
        # 对角线概率最高，表示车辆倾向于保持当前运动状态
        M = np.array([
            [0.90, 0.08, 0.02], # CV -> CV/CTRV/CTRA
            [0.05, 0.90, 0.05], # CTRV -> CV/CTRV/CTRA
            [0.02, 0.08, 0.90]  # CTRA -> CV/CTRV/CTRA
        ])
        
        return IMMEstimator(filters, mu, M)

    def predict(self, history_states: list, steps: int):
        imm = self.create_imm_estimator()

        # === 1. 滤波更新阶段 ===
        for i, state in enumerate(history_states):
            if i == 0:
                init_yaw_rate = 0.0
                init_accel = 0.0
                if len(history_states) > 1:
                    init_yaw_rate = normalize_angle(history_states[1].yaw - history_states[0].yaw) / DT
                    init_accel = (history_states[1].speed - history_states[0].speed) / DT
                
                init_x = np.array([state.x, state.y, state.speed, state.yaw, init_yaw_rate, init_accel])
                
                # 初始化所有子滤波器
                for f in imm.filters:
                    f.x = init_x.copy()
                    f.P = self.P_shared.copy()
                imm.x = init_x.copy()
            else:
                imm.predict()
                
                # 角度解卷，相对于 IMM 融合后的混合状态进行展开
                angle_diff = normalize_angle(state.yaw - imm.x[3])
                unrolled_obs_yaw = imm.x[3] + angle_diff
                
                z = np.array([state.x, state.y, state.speed, unrolled_obs_yaw])
                imm.update(z)
        
        current_state = imm.x.copy()
        current_mu = imm.mu.copy()

        # === 2. 多步预测阶段 ===
        predictions = []
        for i in range(1, steps + 1):
            t_pred = i * DT
            
            # 【IMM的预测魔法】: 按权重融合三种模型的未来推演
            pred_cv = cv_transition(current_state, t_pred)
            pred_ctrv = ctrv_transition(current_state, t_pred)
            pred_ctra = ctra_transition(current_state, t_pred)
            
            mixed_pred_state = (
                current_mu[0] * pred_cv + 
                current_mu[1] * pred_ctrv + 
                current_mu[2] * pred_ctra
            )
            
            predictions.append({
                "step": i,
                "time_offset": round(t_pred, 3),
                "x": float(mixed_pred_state[0]),
                "y": float(mixed_pred_state[1]),
                "yaw": float(normalize_angle(mixed_pred_state[3])), 
                "speed": float(mixed_pred_state[2])
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
    
    predictor = IMMUKFPredictor()
    registry = {}
    
    total_predict_time = 0.0
    total_predict_count = 0
    
    print(f"开始处理 (Method: IMM (CV+CTRV+CTRA) + UKF)...")

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
        print(f"统计完成 (IMM_UKF):")
        print(f"总处理车辆次数: {total_predict_count}")
        print(f"平均单车预测耗时: {avg_time_ms:.4f} ms")
    else:
        print("未检测到有效预测数据。")
    print("="*50)
    print("Done.")

if __name__ == "__main__":
    main()