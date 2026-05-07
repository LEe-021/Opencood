import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # 引入图片读取库
from matplotlib.lines import Line2D

# ================= 【全局字体设置】 =================
# 1. 设置所有常规文本的字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 2. 如果图表中有数学公式（例如我们画图时用的 $T_0$）
# 把公式的字体也设置为与 Times New Roman 风格一致的 STIX
plt.rcParams['mathtext.fontset'] = 'stix'

# 3. 设置全局默认字号（可选，统一设置更协调）
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8    # X/Y 轴标签字号
plt.rcParams['xtick.labelsize'] = 8   # X 轴刻度数字字号
plt.rcParams['ytick.labelsize'] = 8   # Y 轴刻度数字字号
plt.rcParams['legend.fontsize'] = 8   # 图例字号
# ==============================================================

# ================= 配置区域 =================
# 1. 目标场景设置
TARGET_FRAME_ID = 947
TARGET_VID = 133

# 2. 路径设置
INPUT_ROOT = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1/-1"
BASE_DIR = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1_noisy"
JSON_PATHS = {
    "CV": os.path.join(BASE_DIR, "prediction_registry_cv_kf.json"),
    "CTRV": os.path.join(BASE_DIR, "prediction_registry_ctrv_ukf.json"),
    "CTRA": os.path.join(BASE_DIR, "prediction_registry_ctra_ukf.json"),
    "IMM": os.path.join(BASE_DIR, "prediction_registry_imm_ukf.json")
}

HISTORY_STEPS = 5
PRED_STEPS = 6

# ================= 【新增】背景地图配置区域 =================
# 【重要】请修改这里！
# 1. 设置是否启用背景地图
ENABLE_BACKGROUND_MAP = False  # <--- 准备好图片和坐标信息后，把这里改为 True

# 2. 地图图片路径 (例如: "town03_bev.png")
MAP_IMAGE_PATH = "cross_bg.png" 

# 3. 地图图片的物理坐标范围 [xmin, xmax, ymin, ymax] (单位: 米)
# 这是最关键的一步！你需要知道这张图片覆盖了真实世界的哪个区域。
# 如果设置错误，轨迹和地图会对不上。
# 示例：假设图片中心是原点，覆盖范围是横纵各 200米 (-100m 到 +100m)
MAP_EXTENT = [-87, 0, 62, -28] 
# ==========================================================


def get_yaml_data(frame_id, vid):
    """读取指定帧和车辆的真实坐标"""
    yaml_path = os.path.join(INPUT_ROOT, f"{frame_id:06d}.yaml")
    if not os.path.exists(yaml_path):
        return None
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    if data and 'vehicles' in data and data['vehicles']:
        vinfo = None
        if vid in data['vehicles']: vinfo = data['vehicles'][vid]
        elif int(vid) in data['vehicles']: vinfo = data['vehicles'][int(vid)]
        elif str(vid) in data['vehicles']: vinfo = data['vehicles'][str(vid)]
        
        if vinfo:
            return np.array(vinfo['location'][:2])
    return None

def find_prediction_trajectory(json_data, target_frame_id, target_vid):
    """提取预测轨迹"""
    target_frame_str = f"{target_frame_id:06d}"
    
    for frame_key, vehicles in json_data.items():
        if frame_key.endswith(target_frame_str):
            if str(target_vid) in vehicles:
                traj = []
                for step_item in vehicles[str(target_vid)]:
                    traj.append([step_item['x'], step_item['y']])
                return np.array(traj)
    return None

def main():
    print(f"开始提取场景数据: Frame {TARGET_FRAME_ID}, VID {TARGET_VID}")

    # 1. 提取数据 (省略部分与之前相同，保持不变)
    history_traj = []
    for step in range(-HISTORY_STEPS, 1):
        pos = get_yaml_data(TARGET_FRAME_ID + step, TARGET_VID)
        if pos is not None:
            history_traj.append(pos)
    history_traj = np.array(history_traj)
    
    gt_future_traj = []
    for step in range(0, PRED_STEPS + 1):
        pos = get_yaml_data(TARGET_FRAME_ID + step, TARGET_VID)
        if pos is not None:
            gt_future_traj.append(pos)
    gt_future_traj = np.array(gt_future_traj)

    if len(history_traj) == 0 or len(gt_future_traj) < 2:
        print("错误：无法在 YAML 中找到足够的真实轨迹数据。")
        return

    predictions = {}
    for model_name, path in JSON_PATHS.items():
        with open(path, 'r') as f:
            j_data = json.load(f)
            pred_pts = find_prediction_trajectory(j_data, TARGET_FRAME_ID, TARGET_VID)
            if pred_pts is not None:
                t0_pos = gt_future_traj[0:1]
                predictions[model_name] = np.vstack((t0_pos, pred_pts))

    # 2. 开始绘图
    plt.figure(figsize=(3.5, 3), dpi=300)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()

    # ================= 【新增】绘制背景地图 =================
    if ENABLE_BACKGROUND_MAP:
        if os.path.exists(MAP_IMAGE_PATH):
            try:
                print(f"正在加载背景地图: {MAP_IMAGE_PATH} ...")
                # 读取图片
                img = mpimg.imread(MAP_IMAGE_PATH)
                # 将图片贴在最底层 (zorder=0)
                # extent 参数决定了图片角点对应的物理坐标
                ax.imshow(img, extent=MAP_EXTENT, zorder=0, alpha=0.7) # alpha 控制透明度
                print("背景地图加载成功！")
            except Exception as e:
                print(f"警告: 背景地图加载失败: {e}")
        else:
            print(f"警告: 找不到背景地图图片: {MAP_IMAGE_PATH}")
    else:
        # 如果不使用地图，则显示网格以便参考
        plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
    # ========================================================
    
    # ... (后面的轨迹绘制代码与之前完全一致，zorder 都大于0，保证覆盖在地图上) ...
    # [绘制元素 1] 历史轨迹
    plt.plot(history_traj[:, 0], history_traj[:, 1], color='gray', linestyle='--', linewidth=1, zorder=1)
    plt.scatter(history_traj[:-1, 0], history_traj[:-1, 1], color='gray', s=20, label='History', zorder=2)
    
    # [绘制元素 2] 真实未来轨迹 GT
    plt.plot(gt_future_traj[:, 0], gt_future_traj[:, 1], color='black', linestyle='-', linewidth=2, zorder=3)
    plt.scatter(gt_future_traj[1:, 0], gt_future_traj[1:, 1], color='black', marker='*', s=50, label='Ground Truth', zorder=4)
    
    # 标记当前时刻 T0
    t0_x, t0_y = history_traj[-1, 0], history_traj[-1, 1]
    plt.scatter(t0_x, t0_y, color='red', s=50, edgecolor='black', zorder=10)
    plt.annotate('$T_0$', (t0_x, t0_y), textcoords="offset points", xytext=(10,-15), ha='center', fontsize=12, fontweight='bold')

    # [绘制元素 3] 预测轨迹
    plot_styles = {
        "CV":   {"color": "#1f77b4", "marker": "o", "ls": "--", "lw": 1, "alpha": 0.8},
        "CTRV": {"color": "#2ca02c", "marker": "^", "ls": "-.", "lw": 1, "alpha": 0.8},
        "CTRA": {"color": "#ff7f0e", "marker": "s", "ls": ":",  "lw": 1, "alpha": 0.8},
        "IMM":  {"color": "#e377c2", "marker": "D", "ls": "-",  "lw": 2, "alpha": 1}
    }

    for model_name, style in plot_styles.items():
        if model_name in predictions:
            traj = predictions[model_name]
            plt.plot(traj[:, 0], traj[:, 1], 
                     color=style["color"], linestyle=style["ls"], linewidth=style["lw"], 
                     alpha=style["alpha"], zorder=5)
            plt.scatter(traj[1:, 0], traj[1:, 1], 
                        color=style["color"], marker=style["marker"], s=20, 
                        alpha=style["alpha"], label=model_name, zorder=6)

    # 图表修饰
    #plt.title(f"Trajectory Prediction vs. Ground Truth\n(Frame: {TARGET_FRAME_ID}, VID: {TARGET_VID})", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Global X (m)", fontweight='bold')
    plt.ylabel("Global Y (m)", fontweight='bold')
    
    # 如果加了背景图，通常可以把坐标轴刻度关掉，看起来更干净（可选）
    if ENABLE_BACKGROUND_MAP:
        plt.xticks([])
        plt.yticks([])

    leg = plt.legend(loc='upper left', frameon=True, shadow=False, edgecolor='black')
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout()

    output_filename = f"trajectory_vis_map_F{TARGET_FRAME_ID}_V{TARGET_VID}.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\n绘图成功！图片已保存至: {os.path.abspath(output_filename)}")

    plt.show()

if __name__ == "__main__":
    main()