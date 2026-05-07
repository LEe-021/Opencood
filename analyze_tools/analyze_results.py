# -*- coding: utf-8 -*-
# 文件位置: opencood/tools/analyze_results.py

import json
import os
import numpy as np
import yaml
from tqdm import tqdm

# 引入 OpenCOOD 基础工具
from opencood.utils import common_utils
from opencood.utils import box_utils

# ======================= 用户配置区域 =======================

RESULT_JSON_PATH = "/home/step/data/pcPredict_data/cross_fullcar_clean/raw_detection_results/raw_detection_results_IMM_150ms.json"
YAML_DIR = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene/test_1/96"
RSU_POS_WORLD = np.array([-27, -2, 6])
IOU_THRESH = 0.5

# ==========================================================

def to_numpy(data_list):
    if len(data_list) == 0:
        return np.array([])
    return np.array(data_list, dtype=np.float32)

# 【必须保留】手写矩阵变换，因为你的环境缺失 common_utils.x_to_world_transformation
def pose_to_transformation_matrix(pose):
    x, y, z, roll, yaw, pitch = pose
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z

    c_r, s_r = np.cos(roll), np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch)
    c_y, s_y = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0, 0], [0, c_r, -s_r, 0], [0, s_r, c_r, 0], [0, 0, 0, 1]])
    Ry = np.array([[c_p, 0, s_p, 0], [0, 1, 0, 0], [-s_p, 0, c_p, 0], [0, 0, 0, 1]])
    Rz = np.array([[c_y, -s_y, 0, 0], [s_y, c_y, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    # M = T * Rz * Ry * Rx
    M = T @ Rz @ Ry @ Rx
    return M

def standardize_boxes(boxes):
    """统一转换为 (N, 7) 格式"""
    if boxes.shape[0] == 0: return boxes
    if boxes.ndim == 2 and boxes.shape[1] == 7: return boxes
    
    # 兼容 8角点 和 24展平
    if boxes.ndim == 3 and boxes.shape[1] == 8 and boxes.shape[2] == 3:
        return box_utils.corner_to_center(boxes, order='lwh')
    if boxes.ndim == 2 and boxes.shape[1] == 24:
        boxes_corner = boxes.reshape(-1, 8, 3)
        return box_utils.corner_to_center(boxes_corner, order='lwh')
    return np.zeros((0, 7))

def boxes_to_bev_corners(boxes_7d):
    """(N, 7) -> (N, 4, 2)"""
    if len(boxes_7d) == 0: return []
    corners_3d = box_utils.boxes_to_corners_3d(boxes_7d, order='lwh')
    return corners_3d[:, :4, :2]

def get_rsu_pos_in_ego(ego_pose_list, rsu_pos_world):
    try:
        # 1. 确保 float 并转弧度
        ego_pose = np.array(ego_pose_list, dtype=np.float32)
        ego_pose[3:] = np.radians(ego_pose[3:])
        
        # 2. 计算矩阵
        ego2world_mat = pose_to_transformation_matrix(ego_pose)
        world2ego_mat = np.linalg.inv(ego2world_mat)
        
        # 3. 变换坐标
        rsu_world_hom = np.append(rsu_pos_world, 1.0)
        rsu_ego_hom = world2ego_mat @ rsu_world_hom
        return rsu_ego_hom[:3]
    except Exception:
        return None

def main():
    if not os.path.exists(RESULT_JSON_PATH):
        print(f"Error: 找不到结果文件: {RESULT_JSON_PATH}")
        return

    clean_yaml_dir = os.path.expanduser(YAML_DIR)
    
    print(f"Loading raw results from: {RESULT_JSON_PATH}")
    with open(RESULT_JSON_PATH, 'r') as f:
        raw_data = json.load(f)

    detailed_results = {}
    all_frames_data = list(raw_data.values())
    
    # 按 frame_id 排序
    try:
        all_frames_data.sort(key=lambda x: int(x['frame_id']))
    except:
        pass

    print(f"Processing {len(all_frames_data)} frames...")
    
    # 【新增】统计变量
    missing_yaml_count = 0

    for frame_data in tqdm(all_frames_data):
        real_frame_id = int(frame_data['frame_id'])
        
        # --- A. 计算 RSU 相对位置 ---
        rsu_pos_in_current_ego = None
        yaml_path = os.path.join(clean_yaml_dir, f"{real_frame_id:06d}.yaml")
        

        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r') as yf:
                    data = yaml.safe_load(yf)
                    if 'lidar_pose' in data:
                        rsu_pos_in_current_ego = get_rsu_pos_in_ego(data['lidar_pose'], RSU_POS_WORLD)
            except:
                pass
        else:
            # 【新增】计数
            missing_yaml_count += 1

        # --- B. 数据标准化 ---
        raw_preds = [p['box'] for p in frame_data['preds']]
        pred_scores = to_numpy([p['score'] for p in frame_data['preds']])
        pred_boxes = standardize_boxes(to_numpy(raw_preds))

        raw_gts = [g['box'] for g in frame_data['gts']]
        gt_boxes = standardize_boxes(to_numpy(raw_gts))

        # --- C. 准备 Polygon (最原始逻辑) ---
        pred_polygons = []
        pred_indices = []
        
        if len(pred_boxes) > 0:
            order = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]
            pred_indices = order
            
            # 转角点 -> 转 Polygon
            corners = boxes_to_bev_corners(pred_boxes)
            # 这里就是最原始的 convert_format，不管 shapely 警告
            pred_polygons = list(common_utils.convert_format(corners))

        gt_polygons = []
        if len(gt_boxes) > 0:
            corners = boxes_to_bev_corners(gt_boxes)
            gt_polygons = list(common_utils.convert_format(corners))
            gt_matched = [False] * len(gt_polygons)
        
        # --- D. 匹配与记录 ---
        frame_res = {"gt_details": [], "pred_details": []}

        # 记录 GT
        for i in range(len(gt_boxes)):
            gt_center = gt_boxes[i][:3]
            d_ego = float(np.linalg.norm(gt_center))
            d_rsu = -1.0
            if rsu_pos_in_current_ego is not None:
                d_rsu = float(np.linalg.norm(gt_center - rsu_pos_in_current_ego))

            frame_res["gt_details"].append({
                "gt_id": i,
                "box": gt_boxes[i].tolist(),
                "is_detected": False,
                "iou": 0.0,
                "dist_to_ego": d_ego,
                "dist_to_rsu": d_rsu
            })

        # 记录 Pred 并匹配
        for i, pred_poly in enumerate(pred_polygons):
            # 获取原始信息
            curr_id = int(pred_indices[i])
            curr_score = float(pred_scores[i])
            
            match_status = "FP"
            best_iou = 0.0
            
            p_center = pred_boxes[i][:3]
            d_ego = float(np.linalg.norm(p_center))
            d_rsu = -1.0
            if rsu_pos_in_current_ego is not None:
                d_rsu = float(np.linalg.norm(p_center - rsu_pos_in_current_ego))

            if len(gt_polygons) > 0:
                # 警告可能会在这里产生，但程序会继续运行
                ious = common_utils.compute_iou(pred_poly, gt_polygons)
                if len(ious) > 0:
                    best_iou = float(np.max(ious))
                    best_gt_idx = int(np.argmax(ious))

                    if best_iou >= IOU_THRESH:
                        if not gt_matched[best_gt_idx]:
                            match_status = "TP"
                            gt_matched[best_gt_idx] = True
                            frame_res["gt_details"][best_gt_idx].update({
                                "is_detected": True,
                                "iou": best_iou,
                                "matched_pred_id": curr_id,
                                "score": curr_score
                            })
                        else:
                            match_status = "FP (Duplicate)"
            
            frame_res["pred_details"].append({
                "pred_id": curr_id,
                "box": pred_boxes[i].tolist(),
                "score": curr_score,
                "status": match_status,
                "iou": best_iou,
                "dist_to_ego": d_ego,
                "dist_to_rsu": d_rsu
            })

        detailed_results[real_frame_id] = frame_res

    # --- 保存 ---
    save_dir = os.path.dirname(RESULT_JSON_PATH)
    out_file = os.path.join(save_dir, 'analyzed_results_with_dist.json')
    with open(out_file, 'w') as f:
        json.dump(detailed_results, f, indent=4)
        
    print("-" * 50)
    print(f"分析完成! 结果已保存至: {out_file}")
    
    # 【新增】统计输出
    if missing_yaml_count > 0:
        print(f"Warning: 有 {missing_yaml_count} 帧找不到 YAML 文件。")
    else:
        print("所有帧的 YAML 文件均已找到。")
    print("-" * 50)

if __name__ == "__main__":
    main()