# -*- coding: utf-8 -*-
# 文件位置: opencood/tools/plot_distance_analysis.py

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# ======================= 用户配置区域 =======================
CONFIG = {
    # 1. 结果 JSON 路径
    "json_path": "/home/step/data/pcPredict_data/cross_multicar_clean/scene/test_1/analyzed_results_with_dist.json",
    
    # 2. 颜色配置
    "missed_color": "#FF0033",  # 未检测到的颜色 (亮红)
    # IoU 从 0.0 到 1.0 的颜色渐变：橙 -> 黄 -> 绿
    "colors_gradient": ["#FF6600", "#FFD700", "#00FF00"], 
    
    # 3. 绘图样式
    "point_size": 15,    # 点的大小
    "alpha": 0.8,        # 透明度 (防止重叠看不清)
    "bg_color": "#151515", # 背景色 (深色背景对比度更好)
    
    # 4. 过滤 (可选)
    "max_dist": 120      # 只画 150米以内的点，避免离群点压缩画面
}
# ==========================================================

def main():
    if not os.path.exists(CONFIG['json_path']):
        print(f"Error: JSON file not found at {CONFIG['json_path']}")
        return

    print("正在加载数据...")
    with open(CONFIG['json_path'], 'r') as f:
        data = json.load(f)

    # --- 1. 数据提取 ---
    # 我们需要四个列表
    # Missed (未检测)
    missed_d_ego = []
    missed_d_rsu = []
    
    # Detected (已检测)
    det_d_ego = []
    det_d_rsu = []
    det_iou = []

    total_gt = 0
    valid_gt = 0

    print("正在提取距离与IoU信息...")
    for frame_id, frame_data in data.items():
        for gt in frame_data['gt_details']:
            total_gt += 1
            d_ego = gt['dist_to_ego']
            d_rsu = gt['dist_to_rsu']
            
            # 过滤掉无效数据 (比如之前的 -1) 或 超出范围的点
            if d_rsu < 0 or d_ego > CONFIG['max_dist'] or d_rsu > CONFIG['max_dist']:
                continue
            
            valid_gt += 1
            
            if not gt['is_detected']:
                missed_d_ego.append(d_ego)
                missed_d_rsu.append(d_rsu)
            else:
                det_d_ego.append(d_ego)
                det_d_rsu.append(d_rsu)
                det_iou.append(gt['iou'])

    print(f"共处理 {len(data)} 帧, {total_gt} 个目标。")
    print(f"有效绘图点数: {valid_gt} (已过滤距离 > {CONFIG['max_dist']}m 或无效点)")
    print(f"  - 未检测: {len(missed_d_ego)}")
    print(f"  - 已检测: {len(det_d_ego)}")

    # --- 2. 绘图设置 ---
    plt.style.use('dark_background') # 使用黑底，让颜色更鲜艳
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor(CONFIG['bg_color'])
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.2, color='gray')
    ax.set_axisbelow(True) # 让网格在点下面

    # --- 3. 绘制 已检测点 (Gradient Color) ---
    # 创建色盘: Orange -> Yellow -> Green
    cmap = LinearSegmentedColormap.from_list("OrYlGn", CONFIG['colors_gradient'], N=100)
    
    if len(det_d_ego) > 0:
        scatter = ax.scatter(det_d_ego, det_d_rsu, 
                             c=det_iou,          # 颜色依据 IoU
                             cmap=cmap, 
                             vmin=0.0, vmax=1.0, # 固定 IoU 范围
                             s=CONFIG['point_size'], 
                             alpha=CONFIG['alpha'], 
                             edgecolors='none',  # 无边框
                             marker='o',
                             label='Detected')

        # 添加 Colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('IoU (0.0 -> 1.0)', color='white', fontsize=10)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # --- 4. 绘制 未检测点 (Missed, Red) ---
    # 画在 Detected 上面 (zorder 更大)，这样能明显看出哪里容易漏检
    if len(missed_d_ego) > 0:
        ax.scatter(missed_d_ego, missed_d_rsu, 
                   c=CONFIG['missed_color'], 
                   s=CONFIG['point_size'], 
                   alpha=0.8,  # 未检测的点稍微不透明一点，更醒目
                   edgecolors='none', # 加个白边，更显眼
                   marker='o', # 用 X 标记
                   label='Missed')

    # --- 5. 装饰与标注 ---
    ax.set_xlabel('Distance to Ego Vehicle (m)', fontsize=12, color='white')
    ax.set_ylabel('Distance to RSU (m)', fontsize=12, color='white')
    ax.set_title('Detection Accuracy vs. Sensor Distance', fontsize=14, weight='bold', color='white')
    
    # 坐标轴范围
    ax.set_xlim(0, CONFIG['max_dist'])
    ax.set_ylim(0, CONFIG['max_dist'])
    ax.set_aspect('equal') # 保持 1:1 比例，方便对比距离

    # 添加对角线 y=x (参考线)
    ax.plot([0, CONFIG['max_dist']], [0, CONFIG['max_dist']], '--', color='gray', alpha=0.5, linewidth=1)
    ax.text(CONFIG['max_dist']*0.8, CONFIG['max_dist']*0.82, "Equal Dist", color='gray', rotation=45, fontsize=8)

    # 图例 (手动创建，因为 Scatter 的图例有时候不准)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='High IoU (Green)', markerfacecolor=CONFIG['colors_gradient'][2], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Low IoU (Orange)', markerfacecolor=CONFIG['colors_gradient'][0], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Missed (Red)', markerfacecolor=CONFIG['missed_color'], markersize=8)
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='#222222', edgecolor='none')

    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(os.path.dirname(CONFIG['json_path']), 'distance_accuracy_scatter.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"统计图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()