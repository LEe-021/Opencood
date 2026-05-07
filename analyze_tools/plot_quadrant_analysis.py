# -*- coding: utf-8 -*-
# 文件位置: opencood/tools/plot_quadrant_analysis.py

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ======================= 用户配置区域 =======================
CONFIG = {
    "json_path": "/home/step/data/pcPredict_data/cross_multicar_clean/scene/test_1/analyzed_results_with_dist.json",
    
    # 分界线设置
    "boundary": 40,      # 第一分界线 (米)
    "max_dist": 120,     # 统计的最大距离 (米)
    
    # 颜色与样式
    "missed_color": "#FF0033",
    "colors_gradient": ["#FF6600", "#FFD700", "#00FF00"],
    "bg_color": "#151515",
    "text_color": "white",
    
    # AP计算设置
    "score_threshold_step": 0.01 # 用于积分计算精度的步长 (简化版不需要)
}
# ==========================================================

def calculate_ap(gt_list, pred_list, iou_thresh):
    """
    计算特定区域内的 Average Precision (AP)
    gt_list: [{'is_detected': bool, 'iou': float, 'matched_pred_id': int}, ...]
    pred_list: [{'score': float, 'is_tp': bool}, ...] (在这个区域内的所有预测)
    
    注意：这里的逻辑是从 GT 出发反推的简化版 AP 计算。
    严谨的 AP 需要所有 Pred 按分数排序。
    """
    if len(gt_list) == 0:
        return 0.0
    if len(pred_list) == 0:
        return 0.0

    # 1. 对预测结果按分数降序排序
    pred_list.sort(key=lambda x: x['score'], reverse=True)

    tp_count = 0
    fp_count = 0
    total_gt = len(gt_list)
    
    precisions = []
    recalls = []

    # 2. 逐个累加计算 P-R 曲线点
    for pred in pred_list:
        if pred['is_tp'] and pred['iou'] >= iou_thresh:
            tp_count += 1
        else:
            fp_count += 1
        
        precision = tp_count / (tp_count + fp_count + 1e-6)
        recall = tp_count / total_gt
        
        precisions.append(precision)
        recalls.append(recall)

    # 3. 计算 AP (Area Under Curve) - 简单的 11点插值法或直接积分
    if not precisions:
        return 0.0
        
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # 平滑 P-R 曲线 (使得 Precision 单调递减)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
        
    # 计算面积 (Recall 轴上的梯形积分)
    # 在最左侧补点 (R=0, P=P_max)
    recalls = np.concatenate(([0.0], recalls))
    precisions = np.concatenate(([precisions[0]], precisions))
    
    ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    return ap

def get_region_index(d_ego, d_rsu, boundary, max_dist):
    """
    返回区域索引:
    0: Ego<60, RSU<60 (双近)
    1: Ego<60, RSU>60 (车近路远)
    2: Ego>60, RSU<60 (车远路近 - V2X优势区)
    3: Ego>60, RSU>60 (双远)
    -1: 超出范围
    """
    if d_ego > max_dist or d_rsu > max_dist:
        return -1
    
    row = 0 if d_rsu < boundary else 1 # Y轴
    col = 0 if d_ego < boundary else 1 # X轴
    
    # 映射到 0,1,2,3 (对应左下、左上、右下、右上)
    # 这里的顺序对应 matplotlib 的 quadrant
    if row == 0 and col == 0: return 0 # 左下
    if row == 1 and col == 0: return 1 # 左上
    if row == 0 and col == 1: return 2 # 右下
    if row == 1 and col == 1: return 3 # 右上
    
    return -1

def main():
    if not os.path.exists(CONFIG['json_path']):
        print("JSON not found.")
        return

    print("正在加载数据并进行分区域统计...")
    with open(CONFIG['json_path'], 'r') as f:
        data = json.load(f)

    # --- 1. 数据容器初始化 ---
    # 四个区域的数据： regions[i] = {'gts': [], 'preds': []}
    regions = [{'gts': [], 'preds': []} for _ in range(4)]
    
    # 用于画散点图的列表
    scatter_data = {
        'd_ego': [], 'd_rsu': [], 'iou': [], 'status': [] # status: 0=miss, 1=detect
    }

    # --- 2. 遍历数据 ---
    for frame_id, frame_data in data.items():
        # 处理 GT (作为分母)
        for gt in frame_data['gt_details']:
            d_ego = gt['dist_to_ego']
            d_rsu = gt['dist_to_rsu']
            
            rid = get_region_index(d_ego, d_rsu, CONFIG['boundary'], CONFIG['max_dist'])
            
            # 记录散点图数据 (即使超出范围也画出来，方便看全貌，或者这里也可以过滤)
            if rid != -1 or (d_ego <= CONFIG['max_dist'] and d_rsu <= CONFIG['max_dist']):
                scatter_data['d_ego'].append(d_ego)
                scatter_data['d_rsu'].append(d_rsu)
                scatter_data['iou'].append(gt['iou'])
                scatter_data['status'].append(1 if gt['is_detected'] else 0)

            # 记录用于 AP 计算的数据
            if rid != -1:
                regions[rid]['gts'].append(gt)

        # 处理 Pred (作为 AP 计算的分子/FP来源)
        # 注意：我们需要知道 Pred 落在哪个区域。
        # 这里用 Pred 的几何中心距离来判断区域归属。
        for pred in frame_data['pred_details']:
            d_ego = pred['dist_to_ego']
            d_rsu = pred['dist_to_rsu']
            
            rid = get_region_index(d_ego, d_rsu, CONFIG['boundary'], CONFIG['max_dist'])
            
            if rid != -1:
                # 判断这个 Pred 是否是 TP (根据 status 字段)
                # 注意：analyze_results 里我们标记了 TP/FP
                is_tp = (pred['status'] == 'TP')
                # 还需要对应的 IoU (如果是 TP)
                iou = pred['iou'] if is_tp else 0.0
                
                regions[rid]['preds'].append({
                    'score': pred['score'],
                    'is_tp': is_tp,
                    'iou': iou
                })

    # --- 3. 计算各区域 AP ---
    results = []
    region_names = [
        "Region 1\n(Both < 60m)", 
        "Region 2\n(Ego < 60, RSU > 60)", 
        "Region 3\n(Ego > 60, RSU < 60)", 
        "Region 4\n(Both > 60m)"
    ]
    
    print("\n统计结果:")
    for i in range(4):
        ap50 = calculate_ap(regions[i]['gts'], regions[i]['preds'], 0.50)
        ap70 = calculate_ap(regions[i]['gts'], regions[i]['preds'], 0.70)
        num_gt = len(regions[i]['gts'])
        
        results.append({'ap50': ap50, 'ap70': ap70, 'num_gt': num_gt})
        print(f"{region_names[i]}: AP50={ap50:.4f}, AP70={ap70:.4f}, GT_Count={num_gt}")

    # --- 4. 绘图 ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor(CONFIG['bg_color'])

    # A. 绘制散点
    d_ego = np.array(scatter_data['d_ego'])
    d_rsu = np.array(scatter_data['d_rsu'])
    iou = np.array(scatter_data['iou'])
    status = np.array(scatter_data['status'])

    # 绘制 Missed (红叉)
    mask_miss = (status == 0)
    ax.scatter(d_ego[mask_miss], d_rsu[mask_miss], c=CONFIG['missed_color'], marker='x', 
               s=20, alpha=0.6, linewidth=0.8, label='Missed', zorder=10)

    # 绘制 Detected (渐变圆点)
    mask_det = (status == 1)
    cmap = LinearSegmentedColormap.from_list("OrYlGn", CONFIG['colors_gradient'], N=100)
    sc = ax.scatter(d_ego[mask_det], d_rsu[mask_det], c=iou[mask_det], cmap=cmap, vmin=0, vmax=1,
                    s=15, alpha=0.6, marker='o', edgecolors='none', label='Detected', zorder=5)

    # B. 绘制分界线 (四宫格)
    b = CONFIG['boundary']
    m = CONFIG['max_dist']
    
    # 横线 (y=60)
    ax.plot([0, m], [b, b], linestyle='--', color='white', linewidth=1.5, alpha=0.8)
    # 竖线 (x=60)
    ax.plot([b, b], [0, m], linestyle='--', color='white', linewidth=1.5, alpha=0.8)
    # 边界框
    ax.plot([0, m], [m, m], color='gray', linewidth=0.5)
    ax.plot([m, m], [0, m], color='gray', linewidth=0.5)

    # C. 在四个区域中心标注 AP 数值
    # 坐标中心点
    centers = [
        (b/2, b/2),         # 左下
        (b/2, (b+m)/2),     # 左上
        ((b+m)/2, b/2),     # 右下
        ((b+m)/2, (b+m)/2)  # 右上
    ]
    
    # 标注框样式
    bbox_props = dict(boxstyle="round,pad=0.5", fc="#222222", ec="gray", alpha=0.9)
    
    for i in range(4):
        cx, cy = centers[i]
        res = results[i]
        text = (f"{region_names[i]}\n"
                f"AP50: {res['ap50']*100:.1f}%\n"
                f"AP70: {res['ap70']*100:.1f}%\n"
                f"(GT: {res['num_gt']})")
        
        ax.text(cx, cy, text, ha='center', va='center', color='white', 
                fontsize=11, fontweight='bold', bbox=bbox_props, zorder=20)

    # D. 装饰
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_xlabel('Distance to Ego (m)', fontsize=12)
    ax.set_ylabel('Distance to RSU (m)', fontsize=12)
    ax.set_title('Regional AP Analysis (Boundary=60m)', fontsize=16, color='white', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('IoU', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(os.path.dirname(CONFIG['json_path']), 'quadrant_ap_analysis.png')
    #plt.savefig(save_path, dpi=300)
    #print(f"统计图表已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()