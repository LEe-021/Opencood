import os
import yaml
import numpy as np
import json
import glob
from collections import deque
from tqdm import tqdm

# ================= 配置 =================
INPUT_ROOT = "/home/step/data/pcPredict_data/cross_multicar_clean/scene/test_1_noisy/-1_noisy"  # 数据根目录
OUTPUT_FILE = "/home/step/data/pcPredict_data/cross_multicar_clean/scene/test_1_noisy/prediction_registry_ctrv.json" # 结果保存路径
HISTORY_LEN = 10  # 历史窗口长度 (帧)
PRED_STEPS = 6   # 预测未来多少帧
DT = 0.05        # 两帧之间的时间间隔 (50ms)
# =======================================

def deg2rad(deg):
    return deg * np.pi / 180.0

class VehicleState:
    def __init__(self, x, y, yaw, speed, timestamp):
        self.x = x
        self.y = y
        self.yaw = yaw # rad
        self.speed = speed
        self.timestamp = timestamp

class Predictor:
    """
    预测器基类，方便后续替换不同的算法
    """
    def predict(self, history_states, steps):
        raise NotImplementedError

class CTRVPredictor(Predictor):
    """
    CTRV (Constant Turn Rate and Velocity) 模型
    假设车辆在预测期间保持恒定的线速度和角速度
    """
    def predict(self, history_states: list, steps: int):
        # 1. 获取当前状态 (最新一帧)
        current = history_states[-1]
        
        # 2. 计算角速度 (Yaw Rate)
        # 如果历史帧不够，角速度默认为0
        if len(history_states) >= 2:
            # 取最近两帧计算瞬时角速度，或者取5帧做线性拟合
            # 这里采用最近两帧的差分，对短时预测响应最快
            prev = history_states[-2]
            yaw_diff = current.yaw - prev.yaw
            
            # 处理角度跳变 (例如从 3.14 跳到 -3.14)
            while yaw_diff > np.pi: yaw_diff -= 2*np.pi
            while yaw_diff < -np.pi: yaw_diff += 2*np.pi
            
            yaw_rate = yaw_diff / DT
        else:
            yaw_rate = 0.0

        predictions = []
        
        # 3. 递推预测
        # 初始状态
        pred_x = current.x
        pred_y = current.y
        pred_yaw = current.yaw
        v = current.speed
        
        for i in range(1, steps + 1):
            t = i * DT # 预测的时间跨度 (0.05, 0.10, ...)
            
            if abs(yaw_rate) < 1e-4:
                # 近似直线运动
                pred_x += v * np.cos(pred_yaw) * DT
                pred_y += v * np.sin(pred_yaw) * DT
                # pred_yaw 保持不变
            else:
                # 曲线运动 (CTRV公式)
                # x_new = x + (v/w) * (sin(theta + w*dt) - sin(theta))
                # y_new = y + (v/w) * (cos(theta) - cos(theta + w*dt))
                new_yaw = pred_yaw + yaw_rate * DT
                pred_x += (v / yaw_rate) * (np.sin(new_yaw) - np.sin(pred_yaw))
                pred_y += (v / yaw_rate) * (np.cos(pred_yaw) - np.cos(new_yaw))
                pred_yaw = new_yaw
            
            # 存入结果字典
            predictions.append({
                "step": i,
                "time_offset": round(t, 3),
                "x": float(pred_x),
                "y": float(pred_y),
                "yaw": float(pred_yaw),
                "speed": float(v) # 假设速度恒定
            })
            
        return predictions

def load_frame_data(yaml_path):
    """解析单帧 YAML"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    frame_vehicles = {}
    if 'vehicles' in data and data['vehicles'] is not None:
        for vid, vinfo in data['vehicles'].items():
            loc = vinfo['location']
            angle = vinfo['angle'] # [pitch, yaw, roll] in deg
            
            # 注意：OpenCDA/Carla 的 Yaw 通常是 angle[1]
            yaw_rad = deg2rad(float(angle[1]))
            
            raw_speed = float(vinfo['speed'])
            speed_mps = raw_speed / 3.6  # 除以 3.6
            
            frame_vehicles[vid] = VehicleState(
                x=float(loc[0]),
                y=float(loc[1]),
                yaw=yaw_rad,
                speed=speed_mps, # 使用 m/s
                timestamp=0 # 离线处理暂时不需要绝对时间戳
            )
    return frame_vehicles

def main():
    # 1. 扫描所有序列目录
    # 假设结构是 test_1/SequenceID/Frame.yaml
    # 这里需要根据你的实际目录结构微调
    # 假设所有yaml都在 test_1/**/*.yaml
    yaml_files = sorted(glob.glob(os.path.join(INPUT_ROOT, "*.yaml"), recursive=True))
    
    # 按文件夹分组，确保只在同一个序列内使用历史信息
    sequences = {}
    for f in yaml_files:
        dirname = os.path.dirname(f)
        if dirname not in sequences:
            sequences[dirname] = []
        sequences[dirname].append(f)
    
    predictor = CTRVPredictor()
    registry = {} # 总结果
    
    print(f"开始处理")

    for seq_path, files in sequences.items():
        # 针对每个车辆维护一个历史队列
        # vehicle_history: { vid: deque([state_t-4, ..., state_t]) }
        vehicle_history = {}
        
        # 排序确保时间顺序
        files.sort()
        
        for frame_idx, yaml_path in enumerate(tqdm(files, desc=f"Processing {os.path.basename(seq_path)}", leave=False)):
            
            # 提取帧ID (假设文件名是 000060.yaml)
            frame_name = os.path.splitext(os.path.basename(yaml_path))[0]
            # 构建全局唯一的 Key: "目录/文件名"
            frame_key = os.path.join(os.path.relpath(seq_path, INPUT_ROOT), frame_name)
            
            current_vehicles = load_frame_data(yaml_path)
            
            registry[frame_key] = {}
            
            # 更新历史并预测
            for vid, state in current_vehicles.items():
                if vid not in vehicle_history:
                    vehicle_history[vid] = deque(maxlen=HISTORY_LEN)
                
                vehicle_history[vid].append(state)
                
                # 执行预测
                # 即使历史不足5帧，也可以用现有数据预测(代码内部处理了)
                future_preds = predictor.predict(list(vehicle_history[vid]), PRED_STEPS)
                
                registry[frame_key][vid] = future_preds
            
            # 清理：如果某个车这一帧消失了，是否要清除历史？
            # 简单起见，如果当前帧没有该车，就不做预测。
            # 下次该车出现时，作为新车重新积累历史。
            existing_vids = set(current_vehicles.keys())
            history_vids = set(vehicle_history.keys())
            for vid in history_vids - existing_vids:
                del vehicle_history[vid]

    # 保存结果
    print(f"预测完成，正在保存...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()