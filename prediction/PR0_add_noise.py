import os
import yaml
import numpy as np
from tqdm import tqdm
import glob

# ================= 配置区域 =================
# 1. 原始数据目录 (干净的仿真数据)
INPUT_DIR = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1/-1"

# 2. 输出数据目录 (带有噪声的副本)
OUTPUT_DIR = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1_noisy/-1_noisy"

# 3. 噪声标准差 (Standard Deviation, sigma)
STD_POS = 0.2      # 位置误差标准差 (米)
STD_YAW = 0.2      # 航向角误差标准差 (度)
STD_SPEED = 0.2    # 速度误差标准差 (米/秒)
# ===========================================

def add_noise_to_vehicle(vinfo):
    """为单辆车的数据注入高斯噪声"""
    
    # 1. 注入位置噪声 (X, Y, Z)
    loc = vinfo['location']
    loc[0] += np.random.normal(0.0, STD_POS)
    loc[1] += np.random.normal(0.0, STD_POS)
    loc[2] += np.random.normal(0.0, STD_POS)
    # 转换为原生 float 以保证 yaml 格式干净
    vinfo['location'] = [float(val) for val in loc]

    # 2. 注入航向角噪声 (angle 格式为 [pitch, yaw, roll])
    # 注意：YAML 中的角度单位是度 (deg)
    angle = vinfo['angle']
    yaw_noise = np.random.normal(0.0, STD_YAW)
    noisy_yaw = angle[1] + yaw_noise
    
    # 保证角度在合理范围内 (0~360 或 -180~180)
    # Carla通常接受 -180 到 180 或 0 到 360，这里用简单的 0~360 取模
    noisy_yaw = noisy_yaw % 360.0 
    
    angle[1] = noisy_yaw
    vinfo['angle'] = [float(val) for val in angle]

    # 3. 注入速度噪声
    # 注意：原始 YAML 速度单位是 km/h，噪声设定是 0.2 m/s
    speed_kmh = float(vinfo['speed'])
    speed_mps = speed_kmh / 3.6  # 转换到 m/s
    
    # 加入高斯噪声
    speed_mps_noisy = speed_mps + np.random.normal(0.0, STD_SPEED)
    
    # 物理约束：车辆速度不能为负数
    speed_mps_noisy = max(0.0, speed_mps_noisy)
    
    # 转回 km/h 存入字典
    vinfo['speed'] = float(speed_mps_noisy * 3.6)

def process_single_yaml(input_path, output_path):
    """读取、加噪并保存单个 YAML 文件"""
    with open(input_path, 'r') as f:
        data = yaml.safe_load(f)

    if data and 'vehicles' in data and data['vehicles']:
        for vid, vinfo in data['vehicles'].items():
            add_noise_to_vehicle(vinfo)

    # 写入新的 yaml 文件
    # default_flow_style=None 尽量保持列表和字典的层级易读性
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"错误：找不到输入目录 {INPUT_DIR}")
        return

    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    yaml_files = glob.glob(os.path.join(INPUT_DIR, "*.yaml"))
    
    if not yaml_files:
        print("未在输入目录中找到任何 .yaml 文件！")
        return

    print(f"找到 {len(yaml_files)} 个 YAML 文件，准备注入零均值高斯噪声...")
    print(f" > 位置噪声 σ = {STD_POS} m")
    print(f" > 航向噪声 σ = {STD_YAW} deg")
    print(f" > 速度噪声 σ = {STD_SPEED} m/s")

    for yaml_path in tqdm(yaml_files, desc="Adding Noise"):
        filename = os.path.basename(yaml_path)
        output_path = os.path.join(OUTPUT_DIR, filename)
        process_single_yaml(yaml_path, output_path)

    print(f"\n噪声注入完成！")
    print(f"带有噪声的副本已保存在: {OUTPUT_DIR}")
    print("你可以将后续预测脚本中的 INPUT_ROOT 指向该目录进行实验。")

if __name__ == "__main__":
    main()