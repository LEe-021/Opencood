import open3d as o3d
import numpy as np
import yaml
import json
import os
import glob
import copy
from tqdm import tqdm
import shutil

# ================= 配置区域 =================
# 1. 原始数据路径 (路侧 -1 文件夹)
INPUT_PC_ROOT = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1/-1"

# 2. 预测结果 JSON 路径
PRED_JSON_PATH = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene_noisy/test_1_noisy/prediction_registry_imm_ukf_improved.json"

# 3. 输出数据根目录 (脚本会自动在下面创建子文件夹)
#    最终结构:
#    /output_base/test_1_compensated_100ms/
#    /output_base/test_1_compensated_200ms/ ...
OUTPUT_BASE_DIR = "/home/step/data/pcPredict_data/cross_fullcar_clean/IMM_predict_improved_ori"

# 4. 要处理的 Steps 列表
#    既然间隔是 100ms，那么 [1, 2, 3, 4, 5] 对应 100ms 到 500ms
STEPS_TO_PROCESS = [1, 2, 3, 4, 5, 6]

# 5. 数据采集间隔 (用于生成文件夹命名)
TIME_INTERVAL_MS = 50 

# 6. RSU 在世界坐标系的安装位置 (lidar_pose)
RSU_POSITION = np.array([-27.0, -2.0, 6.0])
# ===========================================

def deg2rad(deg):
    return deg * np.pi / 180.0

def rsu2world(points):
    """RSU局部坐标 -> 世界坐标"""
    points_world = copy.deepcopy(points)
    #points_world[:, 1] = -points_world[:, 1] # 修复Y轴镜像
    points_world += RSU_POSITION
    return points_world

def world2rsu(points_world):
    """世界坐标 -> RSU局部坐标"""
    points_local = points_world - RSU_POSITION
    #points_local[:, 1] = -points_local[:, 1] # 恢复镜像
    return points_local

def get_rotation_matrix_z(theta_rad):
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def get_vehicle_obb_world(vinfo):
    """构建车辆的 OBB (世界坐标)"""
    loc = np.array(vinfo["location"], dtype=np.float64)
    angle = vinfo["angle"]
    # 保持与预测脚本一致的读取逻辑
    yaw_rad = deg2rad(float(angle[1])) 
    
    R = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    
    extent = np.array(vinfo["extent"], dtype=np.float64)
    full_extent = extent * 2.0 
    
    center_local = np.array(vinfo.get("center", [0,0,0]), dtype=np.float64)
    center_world = loc + R @ center_local
    
    obb = o3d.geometry.OrientedBoundingBox(center_world, R, full_extent)
    return obb, center_world, yaw_rad

def process_single_frame_all_steps(pcd_path, yaml_path, predictions, output_dirs):
    """
    读取一帧数据，处理所有 Step，并保存到对应文件夹
    """
    # 1. 读取基础数据 (只读一次 I/O)
    pcd = o3d.io.read_point_cloud(pcd_path)
    original_points_local = np.asarray(pcd.points)
    
    if len(original_points_local) == 0:
        return 
    
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
        
    # 2. 转世界坐标 (Base World Points)
    base_points_world = rsu2world(original_points_local)
    
    # 获取帧 ID 
    frame_id_str = os.path.splitext(os.path.basename(pcd_path))[0]
    file_name = os.path.basename(pcd_path)
    
    # 查找预测数据
    current_frame_preds = None
    for k, v in predictions.items():
        if k.endswith(f"/{frame_id_str}") or k == frame_id_str:
            current_frame_preds = v
            break
    
    # === 循环处理每个 Step ===
    for step in STEPS_TO_PROCESS:
        
        # 复制一份世界坐标点云，以免不同 Step 互相影响
        # 注意：这里必须是 deepcopy 或者 array copy
        points_world_step = base_points_world.copy()
        
        # 标记变换状态
        transformed = False
        
        if current_frame_preds and 'vehicles' in yaml_data and yaml_data['vehicles']:
            
            # 遍历车辆
            for vid_str, vinfo in yaml_data['vehicles'].items():
                vid_str = str(vid_str)
                
                if vid_str in current_frame_preds:
                    # 查找对应 Step 的预测
                    target_pred = None
                    for p in current_frame_preds[vid_str]:
                        if p['step'] == step:
                            target_pred = p
                            break
                    
                    if target_pred:
                        transformed = True
                        obb, curr_center, curr_yaw = get_vehicle_obb_world(vinfo)
                        
                        # 临时点云用于裁剪
                        temp_pcd = o3d.geometry.PointCloud()
                        temp_pcd.points = o3d.utility.Vector3dVector(points_world_step)
                        indices = obb.get_point_indices_within_bounding_box(temp_pcd.points)
                        
                        if len(indices) == 0: continue
                            
                        indices = np.array(indices)
                        vehicle_points = points_world_step[indices]
                        
                        # 计算变换
                        pred_x, pred_y, pred_yaw = target_pred['x'], target_pred['y'], target_pred['yaw']
                        
                        delta_pos = np.array([pred_x, pred_y, curr_center[2]]) - curr_center
                        delta_yaw = pred_yaw - curr_yaw
                        
                        # 应用变换
                        pts_centered = vehicle_points - curr_center
                        R_mat = get_rotation_matrix_z(delta_yaw)
                        pts_rotated = pts_centered @ R_mat.T
                        pts_final = pts_rotated + (curr_center + delta_pos)
                        
                        # 写回
                        points_world_step[indices] = pts_final

        # 转回 RSU 局部坐标
        points_out_local = world2rsu(points_world_step)
        
        # 保存到对应 Step 的文件夹
        target_dir = output_dirs[step]
        save_pcd_path = os.path.join(target_dir, file_name)
        
        out_pcd = o3d.geometry.PointCloud()
        out_pcd.points = o3d.utility.Vector3dVector(points_out_local)
        if pcd.has_colors():
            out_pcd.colors = pcd.colors
            
        o3d.io.write_point_cloud(save_pcd_path, out_pcd, write_ascii=True)
        
        # 复制 YAML
        shutil.copy(yaml_path, save_pcd_path.replace('.pcd', '.yaml'))

def main():
    print(f"Loading predictions from {PRED_JSON_PATH}...")
    with open(PRED_JSON_PATH, 'r') as f:
        predictions = json.load(f)
        
    # 获取源文件列表
    search_pattern = os.path.join(INPUT_PC_ROOT, "*.pcd")
    pcd_files = sorted(glob.glob(search_pattern))
    print(f"Found {len(pcd_files)} PCD files.")
    
    # === 初始化输出目录 ===
    # 格式: test_1_compensated_100ms, test_1_compensated_200ms...
    output_dirs = {}
    dataset_name = os.path.basename(os.path.dirname(INPUT_PC_ROOT)) # 提取 'test_1'
    # 如果提取不到 (比如路径以/结尾)，手动指定或微调逻辑
    if dataset_name == "-1": # 你的路径是 .../test_1/-1
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(INPUT_PC_ROOT)))
    
    print(f"Preparing output directories in {OUTPUT_BASE_DIR}...")
    for step in STEPS_TO_PROCESS:
        latency_ms = step * TIME_INTERVAL_MS
        folder_name = f"{dataset_name}_compensated_{latency_ms}ms"
        full_path = os.path.join(OUTPUT_BASE_DIR, folder_name) # 这里不包含 -1，如果需要保持结构可自行添加
        
        # 如果你想保持 OpenCDA 的结构，建议加上 /-1 子目录
        # 例如: test_1_compensated_100ms/-1/
        full_path_with_sensor = os.path.join(full_path, "-1")
        
        if not os.path.exists(full_path_with_sensor):
            os.makedirs(full_path_with_sensor)
            
        output_dirs[step] = full_path_with_sensor
        print(f"  - Step {step} ({latency_ms}ms) -> {full_path_with_sensor}")

    print("Starting batch processing...")
    for pcd_path in tqdm(pcd_files):
        yaml_path = pcd_path.replace('.pcd', '.yaml')
        if not os.path.exists(yaml_path):
            continue
        
        process_single_frame_all_steps(pcd_path, yaml_path, predictions, output_dirs)
        
    print("All done!")

if __name__ == "__main__":
    main()