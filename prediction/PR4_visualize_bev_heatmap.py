import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.transforms as transforms
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# ================= 【全局学术字体设置】 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
# ========================================================

# ======================= 用户配置区域 =======================
# 1. 基础路径配置
YAML_DIR = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1/96"
JSON_PATHS = {
    "Ego-only": "/home/step/data/pcPredict_data/cross_fullcar_clean/raw_detection_results/analyzed_results_ego_only.json",
    "Uncompensated V2I (150ms)": "/home/step/data/pcPredict_data/cross_fullcar_clean/raw_detection_results/analyzed_results_150ms.json",
    "IMM-Compensated V2I (150ms)": "/home/step/data/pcPredict_data/cross_fullcar_clean/raw_detection_results/analyzed_results_IMM_150ms.json"
}

# 2. 地图背景配置
ENABLE_BACKGROUND_MAP = True  # 启用背景图
MAP_IMAGE_PATH = "cross_bg.png"  # 替换为你的地图图片路径
# 物理坐标范围: [xmin, xmax, ymax, ymin] (注意: 为了兼容向下的Y轴，ymax填在前面，ymin填在后面)
MAP_EXTENT = [-82, 0, 67, -30] 

# 3. 物理与渲染参数
RSU_POS_WORLD = np.array([-27.0, -2.0, 6.0]) # RSU 世界坐标
STOP_SPEED_THRESH = 0.1  
GRID_RES = 0.5   
SIGMA_M = 4.0    
# ==========================================================

def pose_to_transformation_matrix(pose):
    """将 6 DoF 位姿转换为 4x4 变换矩阵 (主车 -> 世界)"""
    x, y, z, roll, yaw, pitch = pose
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z

    c_r, s_r = np.cos(roll), np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch)
    c_y, s_y = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0, 0], [0, c_r, -s_r, 0], [0, s_r, c_r, 0], [0, 0, 0, 1]])
    Ry = np.array([[c_p, 0, s_p, 0], [0, 1, 0, 0], [-s_p, 0, c_p, 0], [0, 0, 0, 1]])
    Rz = np.array([[c_y, -s_y, 0, 0], [s_y, c_y, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    return T @ Rz @ Ry @ Rx

def find_red_light_stop_period(yaml_dir):
    """提取主车在红灯处停留的连续时间段"""
    print("正在扫描 YAML 文件，提取主车静止时间段...")
    yaml_files = sorted([f for f in os.listdir(yaml_dir) if f.endswith('.yaml')])
    
    current_segment = []
    longest_segment = []
    
    current_stopped_pose = None
    best_stopped_pose = None  # 【新增】专门用来记录最长静止时段的初始位姿
    
    prev_pos = None
    
    for yf in yaml_files:
        frame_id = int(os.path.splitext(yf)[0])
        with open(os.path.join(yaml_dir, yf), 'r') as f:
            data = yaml.safe_load(f)
            
        if 'lidar_pose' not in data: continue
        curr_pos = np.array(data['lidar_pose'][:2])
        
        if prev_pos is not None:
            speed = np.linalg.norm(curr_pos - prev_pos) / 0.1 
            if speed < STOP_SPEED_THRESH:
                current_segment.append(frame_id)
                if current_stopped_pose is None:
                    current_stopped_pose = data['lidar_pose']
            else:
                # 当车辆重新移动时，检查并保存最长记录
                if len(current_segment) > len(longest_segment):
                    longest_segment = current_segment
                    best_stopped_pose = current_stopped_pose # 【修复】重置前把最佳位姿存下来
                
                # 重置当前状态
                current_segment = []
                current_stopped_pose = None
                
        prev_pos = curr_pos

    # 循环结束后，收尾检查最后一段是否是最长的
    if len(current_segment) > len(longest_segment):
        longest_segment = current_segment
        best_stopped_pose = current_stopped_pose
        
    print(f"提取完成！主车最长静止时段: {len(longest_segment)} 帧")
    return longest_segment, best_stopped_pose

def create_accuracy_cmap():
    """检测精度专属色带"""
    colors = [
        (1.0, 0.0, 0.0, 0.00),  # 透明
        (1.0, 0.0, 0.0, 0.50),  # 红
        (1.0, 0.8, 0.0, 0.80),  # 黄
        (0.0, 0.8, 0.2, 0.95)   # 绿
    ]
    return LinearSegmentedColormap.from_list('IoU_Accuracy', colors, N=256)

def generate_spatial_heatmap(points, extent, resolution, sigma):
    """高斯热力场渲染"""
    xmin, xmax, ymax, ymin = extent # 接收的是 MAP_EXTENT 格式 [xmin, xmax, ymax, ymin]
    
    # 强制从小到大生成坐标轴，保证矩阵和物理世界的严格映射
    real_xmin, real_xmax = min(xmin, xmax), max(xmin, xmax)
    real_ymin, real_ymax = min(ymin, ymax), max(ymin, ymax)
    
    x_coords = np.arange(real_xmin, real_xmax, resolution)
    y_coords = np.arange(real_ymin, real_ymax, resolution)
    
    xx, yy = np.meshgrid(x_coords, y_coords)
    heatmap = np.zeros_like(xx, dtype=float)
    
    for x, y, iou in points:
        dist_sq = (xx - x)**2 + (yy - y)**2
        gaussian_splat = iou * np.exp(-dist_sq / (2 * sigma**2))
        heatmap = np.maximum(heatmap, gaussian_splat)
        
    # 返回热力矩阵以及物理边界 [左, 右, 下, 上]
    return heatmap, [real_xmin, real_xmax, real_ymin, real_ymax]

def main():
    stopped_frames, ref_pose = find_red_light_stop_period(YAML_DIR)
    if len(stopped_frames) < 5:
        print("警告：提取到的静止帧数极少！")
        return
        
    # 计算 Ego 到 World 的变换矩阵
    ego_pose_rad = np.array(ref_pose, dtype=np.float32)
    ego_pose_rad[3:] = np.radians(ego_pose_rad[3:])
    ego2world_mat = pose_to_transformation_matrix(ego_pose_rad)
    
    ego_world_x, ego_world_y = ref_pose[0], ref_pose[1]
    ego_world_yaw = ref_pose[4] # 角度制，用于画车框旋转

    # 从三个 JSON 提取数据并转到世界坐标系
    datasets_points = {title: [] for title in JSON_PATHS.keys()}
    
    for title, json_path in JSON_PATHS.items():
        if not os.path.exists(json_path): continue
            
        with open(json_path, 'r') as f:
            res_data = json.load(f)
            
        for frame_id in stopped_frames:
            str_fid = str(frame_id)
            if str_fid in res_data:
                for gt in res_data[str_fid]['gt_details']:
                    local_x, local_y = gt['box'][0], gt['box'][1]
                    iou = gt['iou']
                    
                    # Local -> World 转换
                    local_pt = np.array([local_x, local_y, 0, 1.0])
                    world_pt = ego2world_mat @ local_pt
                    
                    datasets_points[title].append((world_pt[0], world_pt[1], iou))

    print("正在渲染 2D BEV 高精度全局热力图...")
    custom_cmap = create_accuracy_cmap()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), dpi=300)
    
    # 加载背景地图
    bg_img = None
    if ENABLE_BACKGROUND_MAP and os.path.exists(MAP_IMAGE_PATH):
        bg_img = mpimg.imread(MAP_IMAGE_PATH)
    
    images = []
    for i, title in enumerate(JSON_PATHS.keys()):
        ax = axes[i]
        points = datasets_points[title]
        
        # 1. 绘制底层地图
        if bg_img is not None:
            # extent 保证图片贴在正确的物理世界坐标上
            ax.imshow(bg_img, extent=MAP_EXTENT, zorder=0, alpha=0.5)
        else:
            ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
            ax.set_facecolor('#f8f9fa')
            
        # 【重要更新】：取消局部聚焦，锁定为 MAP_EXTENT 全局视角
        ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
        # Y轴直接取 MAP_EXTENT 的第三、第四个参数，实现物理向下的完美映射
        ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3]) 
        ax.set_aspect('equal')
        
        # 2. 渲染全局热力图
        if len(points) > 0:
            heatmap_matrix, hm_extent = generate_spatial_heatmap(points, MAP_EXTENT, GRID_RES, SIGMA_M)
            # origin='lower' 表示矩阵第0行放在 hm_extent[2] 的位置 (即真正的物理最底端)
            im = ax.imshow(heatmap_matrix, extent=hm_extent, origin='lower', 
                           cmap=custom_cmap, vmin=0.0, vmax=1.0, zorder=1)
            images.append(im)
            
        # 3. 绘制自车
        ego_rect = patches.Rectangle(
            (ego_world_x - 2.5, ego_world_y - 1.0), 5, 2, 
            linewidth=1, edgecolor='black', facecolor='cyan', 
            zorder=3
        )

        # 【新增】：手动创建一个绕自车中心点旋转的变换矩阵，并叠加到图表的默认坐标变换上
        t = transforms.Affine2D().rotate_deg_around(ego_world_x, ego_world_y, ego_world_yaw) + ax.transData
        ego_rect.set_transform(t)

        ax.add_patch(ego_rect)

        # 【新增】：文本框直接指向自车
        ax.annotate('Ego', xy=(ego_world_x, ego_world_y), 
                    xytext=(ego_world_x + 7, ego_world_y), # 文本放在车中心往右上角偏移15米的位置
                    fontsize=8, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, boxstyle='round,pad=0.3'),
                    zorder=5)
        
        # 4. 绘制 RSU
        ax.scatter(RSU_POS_WORLD[0], RSU_POS_WORLD[1], marker='*', s=100, color='red', 
                   edgecolor='black', linewidth=1, zorder=4)
        
        # 【新增】：文本框直接指向 RSU
        ax.annotate('RSU', xy=(RSU_POS_WORLD[0], RSU_POS_WORLD[1]), 
                    xytext=(RSU_POS_WORLD[0] + 7, RSU_POS_WORLD[1]), # 文本放在星号往右下角偏移15米的位置
                    fontsize=8, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, boxstyle='round,pad=0.3'),
                    zorder=5)

        ax.set_title(title, pad=5, fontweight='bold')
        ax.set_xlabel("Global X (m)")
        if i == 0:
            ax.set_ylabel("Global Y (m)", labelpad=5)
            
    # 5. Colorbar
    if len(images) > 0:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(images[0], cax=cbar_ax)
        cbar.set_label('Detection Accuracy (IoU)', fontsize=8, fontweight='bold', labelpad=5)
        cbar.set_ticks([0.0, 0.2, 0.5, 0.8, 1.0])
        cbar.set_ticklabels(['0.0\nMiss', '0.2', '0.5\nFair', '0.8', '1.0\nPerfect'])
        
    plt.subplots_adjust(left=0.05, right=0.9, bottom=0.1, top=0.9, wspace=0.15)
    
    out_file = "global_detection_heatmap_with_map.png"
    plt.savefig(out_file, bbox_inches='tight')
    print(f"制图成功！全局视野图片已保存至: {os.path.abspath(out_file)}")

    #plt.show()

if __name__ == "__main__":
    main()