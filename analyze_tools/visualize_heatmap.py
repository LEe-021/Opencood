# -*- coding: utf-8 -*-
# 文件位置: opencood/tools/visualize_bev_animation.py

import os
import json
import yaml
import time
import numpy as np
import open3d as o3d 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# 引入 OpenCOOD 工具
from opencood.utils import box_utils

# ======================= 用户配置区域 =======================
CONFIG = {
    # 1. 路径设置
    "json_path": "/home/step/data/pcPredict_data/cross_multicar_clean/scene/test_1/analyzed_results_with_dist.json",
    "ego_dir": "/home/step/data/pcPredict_data/cross_multicar_clean/scene/test_1/399",
    "rsu_dir": "/home/step/data/pcPredict_data/cross_multicar_clean/scene/test_1/-1",
    
    # 2. 视图范围
    "bev_range": [-80, -100, 120, 100],
    "invert_x": False, 
    "invert_y": False,   
    "swap_xy": True, 

    # 3. 播放控制
    "play_speed": 0.1,  # 帧间隔 (秒)
    "start_frame": 0,

    # 4. 视觉风格 (柔和对比度版)
    "bg_color": "#151515",        # 背景色 (柔和暗灰，不再是纯黑)
    "rsu_point_color": "#777777", # 路侧点云 (中灰，亮度提升)
    "ego_point_color": "#DDDDDD", # 自车点云 (青色，保持高亮)
    
    "missed_color": "#FF0033",    # 漏检颜色 (红)
    "colors_gradient": ["#FF6600", "#FFD700", "#00FF00"], # 橙 -> 黄 -> 绿
    
    "glow_layers": 5,             # 光晕层数
    "glow_sigma": 2,            # 光晕扩散
    "fill_alpha": 0.4             # 框填充透明度
}
# =======================================================================

def load_pcd_and_pose(base_dir, frame_id):
    file_idx = f"{frame_id:06d}"
    pcd_path = os.path.join(base_dir, f"{file_idx}.pcd")
    yaml_path = os.path.join(base_dir, f"{file_idx}.yaml")

    if not os.path.exists(pcd_path) or not os.path.exists(yaml_path):
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

def get_separated_points(config, fid):
    ego_pts, ego_pose = load_pcd_and_pose(config['ego_dir'], fid)
    rsu_pts, rsu_pose = load_pcd_and_pose(config['rsu_dir'], fid)

    if ego_pts is None or rsu_pts is None: return None, None

    ego_mat = pose_to_matrix(ego_pose)
    rsu_mat = pose_to_matrix(rsu_pose)
    T_rsu2ego = np.linalg.inv(ego_mat) @ rsu_mat
    
    N = rsu_pts.shape[0]
    rsu_hom = np.hstack((rsu_pts, np.ones((N, 1))))
    rsu_trans_hom = (T_rsu2ego @ rsu_hom.T).T
    rsu_pts_in_ego = rsu_trans_hom[:, :3]
    
    return ego_pts, rsu_pts_in_ego

def draw_glow_effect(ax, poly_x, poly_y, color, config):
    px = np.append(poly_x, poly_x[0])
    py = np.append(poly_y, poly_y[0])
    
    layers = config['glow_layers']
    base_width = 1.0
    
    for i in range(layers):
        width = base_width + (i * config['glow_sigma'])
        alpha = 0.25 / (i + 1)
        ax.plot(px, py, color=color, linewidth=width, alpha=alpha, zorder=4)
    
    ax.plot(px, py, color=color, linewidth=1.0, alpha=0.9, zorder=5)

def update_plot(ax, frame_id, gt_list, ego_pts, rsu_pts, config, cmap):
    """更新单帧画面"""
    ax.clear() 
    # 设置背景色 (Configurable)
    ax.set_facecolor(config['bg_color'])
    
    # 坐标映射
    idx_x, idx_y = (1, 0) if config['swap_xy'] else (0, 1)
    
    # --- 1. 画 RSU 点云 (背景) ---
    r = config['bev_range']
    mask_rsu = (rsu_pts[:, 0] > r[0]) & (rsu_pts[:, 0] < r[2]) & \
               (rsu_pts[:, 1] > r[1]) & (rsu_pts[:, 1] < r[3])
    rsu_valid = rsu_pts[mask_rsu]
    
    if len(rsu_valid) > 20000: 
        rsu_valid = rsu_valid[::int(len(rsu_valid)/20000)]
    
    # 颜色提亮，透明度增加
    ax.plot(rsu_valid[:, idx_x], rsu_valid[:, idx_y], ',', 
            color=config['rsu_point_color'], alpha=0.5, zorder=1)

    # --- 2. 画 Ego 点云 (前景) ---
    mask_ego = (ego_pts[:, 0] > r[0]) & (ego_pts[:, 0] < r[2]) & \
               (ego_pts[:, 1] > r[1]) & (ego_pts[:, 1] < r[3])
    ego_valid = ego_pts[mask_ego]
    
    if len(ego_valid) > 15000:
        ego_valid = ego_valid[::int(len(ego_valid)/15000)]
        
    ax.plot(ego_valid[:, idx_x], ego_valid[:, idx_y], '.', markersize=1.0, 
            color=config['ego_point_color'], alpha=0.8, zorder=2)

    # --- 3. 画框 (GT Heatmap) ---
    for gt in gt_list:
        box_7d = gt['box']
        box_np = np.array([box_7d])
        corners_3d = box_utils.boxes_to_corners_3d(box_np, order='lwh')
        corners_2d = corners_3d[0, :4, :2] 
        
        if not gt['is_detected']:
            color = config['missed_color'] # 红色
            # 未匹配：只有边框发光，内部少填一点
            ax.fill(corners_2d[:, idx_x], corners_2d[:, idx_y], color=color, alpha=0.1, zorder=3)
        else:
            # 匹配：橙 -> 黄 -> 绿
            iou_val = max(0.0, min(1.0, gt['iou']))
            color = cmap(iou_val)
            # 匹配：内部填充明显
            ax.fill(corners_2d[:, idx_x], corners_2d[:, idx_y], color=color, alpha=config['fill_alpha'], zorder=3)
        
        draw_glow_effect(ax, corners_2d[:, idx_x], corners_2d[:, idx_y], color, config)

    # --- 4. 视图设置 ---
    if config['swap_xy']:
        ax.set_xlim(r[1], r[3])
        ax.set_ylim(r[0], r[2])
    else:
        ax.set_xlim(r[0], r[2])
        ax.set_ylim(r[1], r[3])

    if config['invert_x']: ax.invert_xaxis()
    if config['invert_y']: ax.invert_yaxis()

    ax.set_aspect('equal')
    ax.set_title(f"Dynamic Fusion - Frame: {frame_id}", color='white', fontsize=14, weight='bold')
    ax.axis('off')

def main():
    if not os.path.exists(CONFIG['json_path']):
        print("JSON not found.")
        return
        
    print("加载 JSON 数据...")
    with open(CONFIG['json_path'], 'r') as f:
        data = json.load(f)
    
    valid_frames = []
    for k in data.keys():
        try:
            valid_frames.append(int(k))
        except:
            pass
    valid_frames.sort()
    
    start_f = CONFIG.get('start_frame', 0)
    valid_frames = [f for f in valid_frames if f >= start_f]

    print(f"共加载 {len(valid_frames)} 帧。")
    print("提示: 在绘图窗口激活时，按键盘 'Q' 或 'Esc' 键可停止播放。") # <--- 提示用户
    
    # 1. 准备色盘
    custom_cmap = LinearSegmentedColormap.from_list("OrYlGn", CONFIG['colors_gradient'], N=100)

    # 2. 初始化窗口
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title('OpenCOOD Animation (Press Q to Quit)') # 标题提示

    # --- 【新增】 键盘事件监听器 ---
    playback_control = {'running': True} # 使用字典或类属性来在闭包中修改状态

    def on_key_press(event):
        if event.key == 'q' or event.key == 'escape':
            playback_control['running'] = False
            print("\n[用户指令] 检测到按键，正在停止...")

    # 将监听器绑定到 Figure 上
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    # ---------------------------
    
    # 3. 初始化 Colorbar
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('IoU Quality', color='#AFABAB', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='#AFABAB')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#AFABAB')
    
    fig.text(0.15, 0.92, "■ Missed (FN)", color=CONFIG['missed_color'], fontsize=12, weight='bold')

    # 4. 循环播放
    try:
        for fid in valid_frames:
            # 【关键】检查是否需要停止
            if not playback_control['running']:
                break

            # 如果窗口被手动关闭了，也停止循环
            if not plt.fignum_exists(fig.number):
                print("\n窗口已关闭，停止播放。")
                break

            frame_data = data[str(fid)]
            gt_list = frame_data['gt_details']
            
            ego_pts, rsu_pts = get_separated_points(CONFIG, fid)
            if ego_pts is None: continue
                
            update_plot(ax, fid, gt_list, ego_pts, rsu_pts, CONFIG, custom_cmap)
            
            plt.pause(CONFIG['play_speed'])
            print(f"Playing Frame: {fid} (按 Q 退出)", end='\r')
            
    except KeyboardInterrupt:
        print("\n动画已停止 (Ctrl+C)。")
    except Exception as e:
        print(f"\n发生错误: {e}")

    print("\n播放结束。")
    # 如果想保留最后一帧画面不关闭，取消下面这行的注释
    plt.show()

if __name__ == "__main__":
    main()