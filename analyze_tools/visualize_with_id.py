# -*- coding: utf-8 -*-
# 文件位置: opencood/tools/visualize_bev_only.py

import os
import json
import yaml
import numpy as np
import open3d as o3d 
import matplotlib.pyplot as plt

# 引入 OpenCOOD 工具
from opencood.utils import box_utils

# ======================= 用户配置区域 (请修改这里) =======================
CONFIG = {
    "json_path": "/home/step/data/pcPredict_data/cross_multicar_clean/scene/test_1/analyzed_results_with_dist.json",
    "frame_id": 120,
    "ego_dir": "/home/step/data/pcPredict_data/cross_multicar_clean/scene/test_1/399",
    "rsu_dir": "/home/step/data/pcPredict_data/cross_multicar_clean/scene/test_1/-1",
    
    "bev_range": [-80, -100, 120, 100],
    "score_thresh": 0.3,

    # 【新增】坐标轴翻转控制 (解决镜像问题)
    # 如果感觉上下颠倒，将 invert_y 设为 True
    # 如果感觉左右颠倒，将 invert_x 设为 True
    # 如果想让车头朝上 (默认是朝右)，将 swap_xy 设为 True
    "invert_x": False, 
    "invert_y": False,   # <--- 默认开启这个来解决上下镜像
    "swap_xy": True    # <--- 开启这个可以让车头朝向屏幕上方
}
# =======================================================================

def load_pcd_and_pose(base_dir, frame_id):
    file_idx = f"{frame_id:06d}"
    pcd_path = os.path.join(base_dir, f"{file_idx}.pcd")
    yaml_path = os.path.join(base_dir, f"{file_idx}.yaml")

    if not os.path.exists(pcd_path) or not os.path.exists(yaml_path):
        print(f"[Error] File not found: {pcd_path} or {yaml_path}")
        return None, None

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        pose = data['lidar_pose'] 

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    
    return points, pose

def pose_to_matrix(pose_deg):
    x, y, z, roll, yaw, pitch = pose_deg
    roll, yaw, pitch = np.radians([roll, yaw, pitch])
    
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    
    c_r, s_r = np.cos(roll), np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch)
    c_y, s_y = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, c_r, -s_r], [0, s_r, c_r]])
    Ry = np.array([[c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]])
    Rz = np.array([[c_y, -s_y, 0], [s_y, c_y, 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    T[:3, :3] = R
    return T

def get_fused_pcd(config):
    fid = config['frame_id']
    print(f"正在加载第 {fid} 帧数据...")
    
    ego_pts, ego_pose = load_pcd_and_pose(config['ego_dir'], fid)
    rsu_pts, rsu_pose = load_pcd_and_pose(config['rsu_dir'], fid)

    if ego_pts is None or rsu_pts is None:
        return None

    ego_mat_world = pose_to_matrix(ego_pose)
    rsu_mat_world = pose_to_matrix(rsu_pose)
    
    T_rsu2ego = np.linalg.inv(ego_mat_world) @ rsu_mat_world
    
    N = rsu_pts.shape[0]
    rsu_hom = np.hstack((rsu_pts, np.ones((N, 1))))
    rsu_trans_hom = (T_rsu2ego @ rsu_hom.T).T
    rsu_pts_in_ego = rsu_trans_hom[:, :3]
    
    fused_pts = np.vstack((ego_pts, rsu_pts_in_ego))
    
    print(f"融合完成: Ego点数={len(ego_pts)}, RSU点数={len(rsu_pts)}, 总点数={len(fused_pts)}")
    return fused_pts

def draw_bev(points, gt_list, pred_list, config):
    print("正在绘制 BEV 图像...")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('black')
    
    # --- 0. 处理坐标轴交换 (如果想让车头朝上) ---
    # 默认: X(前后), Y(左右)。Swap后: X(左右), Y(前后)
    idx_x, idx_y = 0, 1
    if config['swap_xy']:
        idx_x, idx_y = 1, 0 # 交换绘制索引

    # --- 1. 画点云 ---
    r = config['bev_range'] 
    
    # 稍微放宽过滤范围，防止旋转后点丢了
    mask = (points[:, 0] > r[0]-50) & (points[:, 0] < r[2]+50) & \
           (points[:, 1] > r[1]-50) & (points[:, 1] < r[3]+50)
    valid_pts = points[mask]
    
    if len(valid_pts) > 20000:
        indices = np.random.choice(len(valid_pts), 20000, replace=False)
        valid_pts = valid_pts[indices]
        
    # 绘制点 (注意 idx_x, idx_y)
    ax.scatter(valid_pts[:, idx_x], valid_pts[:, idx_y], s=0.3, c='gray', alpha=0.5)
    
    # --- 2. 画框辅助函数 ---
    def draw_box_on_ax(box_7d, color, label_text=None, linewidth=1.5):
        box_np = np.array([box_7d])
        corners_3d = box_utils.boxes_to_corners_3d(box_np, order='lwh')
        corners_2d = corners_3d[0, :4, :2] 
        
        poly = np.vstack((corners_2d, corners_2d[0]))
        
        # 绘制线 (注意 idx_x, idx_y)
        ax.plot(poly[:, idx_x], poly[:, idx_y], color=color, linewidth=linewidth)
        
        if label_text is not None:
            cx, cy = box_7d[idx_x], box_7d[idx_y]
            ax.text(cx, cy, str(label_text), color='white', fontsize=9, weight='bold',
                    bbox=dict(facecolor=color, alpha=0.6, edgecolor='none', pad=1.5),
                    ha='center', va='center')

    # --- 3. 绘制框 ---
    for pred in pred_list:
        draw_box_on_ax(pred['box'], 'red', linewidth=1.0)

    for gt in gt_list:
        draw_box_on_ax(gt['box'], '#00FF00', label_text=f"ID:{gt['id']}", linewidth=2.0)
        
    # --- 4. 设置视图范围与翻转 ---
    # 根据 swap_xy 调整 limit
    if config['swap_xy']:
        ax.set_xlim(r[1], r[3])
        ax.set_ylim(r[0], r[2])
        xlabel = 'Y (Left/Right)'
        ylabel = 'X (Forward/Backward)'
    else:
        ax.set_xlim(r[0], r[2])
        ax.set_ylim(r[1], r[3])
        xlabel = 'X (Forward/Backward)'
        ylabel = 'Y (Left/Right)'

    # 【核心修复】应用翻转
    if config['invert_x']:
        ax.invert_xaxis()
        print("已应用 X 轴翻转")
    if config['invert_y']:
        ax.invert_yaxis()
        print("已应用 Y 轴翻转")

    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    title_str = f"Frame {config['frame_id']} BEV"
    if config['invert_y']: title_str += " (Y-Inverted)"
    plt.title(title_str, color='white')
    
    plt.grid(True, color='gray', linestyle='--', linewidth=0.3, alpha=0.5)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    print("绘图完成，正在显示...")
    plt.show()

def main():
    # 路径检查
    if not os.path.exists(CONFIG['json_path']):
        print(f"Error: JSON not found: {CONFIG['json_path']}")
        return
        
    with open(CONFIG['json_path'], 'r') as f:
        data = json.load(f)
    
    # 兼容 string/int key
    frame_key = str(CONFIG['frame_id'])
    if frame_key not in data:
        if int(frame_key) in data: frame_key = int(frame_key)
        else:
            print(f"Error: Frame {CONFIG['frame_id']} not found.")
            return
            
    frame_data = data[frame_key]
    
    # 提取列表
    gt_list_vis = []
    for gt in frame_data['gt_details']:
        gt_list_vis.append({'box': gt['box'], 'id': gt['gt_id']})
        
    pred_list_vis = []
    for pred in frame_data['pred_details']:
        if pred['score'] > CONFIG['score_thresh']:
            pred_list_vis.append({'box': pred['box']})

    # 获取点云
    fused_points = get_fused_pcd(CONFIG)
    if fused_points is None: return

    # 绘图
    draw_bev(fused_points, gt_list_vis, pred_list_vis, CONFIG)

if __name__ == "__main__":
    main()