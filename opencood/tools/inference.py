# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

#记得确认数据存入时的frame_id

import argparse
import os
import time
from tqdm import tqdm

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt

import json
import numpy as np


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=4,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False,
                             persistent_workers=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    full_frame_analysis = {} # 初始化存储字典
    # import glob
    # path_list = sorted(glob.glob("opv2v_data_dumping/test_culver_city/2021_08_22_09_08_29/5933/*.pcd"))
    for i, batch_data in tqdm(enumerate(data_loader)):
        # print(path_list[i])
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if opt.fusion_method == 'late':
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_late_fusion(batch_data,
                                                              model,
                                                              opencood_dataset)
                elif opt.fusion_method == 'early':
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_early_fusion(batch_data,
                                                               model,
                                                               opencood_dataset)
                elif opt.fusion_method == 'intermediate':
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_intermediate_fusion(batch_data,
                                                                      model,
                                                                      opencood_dataset)
                else:
                    raise NotImplementedError('Only early, late and intermediate'
                                              'fusion is supported.')

            '''
            # 【终极兼容版】自适应形状保存 (支持 7参数 或 8角点)
            # ==========================================
            
            # --- 1. 处理预测结果 (Predictions) ---
            if pred_box_tensor is not None and pred_box_tensor.numel() > 0:
                cur_preds = pred_box_tensor.detach().cpu().numpy() # e.g. [1, 9, 8, 3] 或 [1, 9, 7]
                cur_scores = pred_score.detach().cpu().numpy()     # e.g. [1, 9]
                
                # 智能去 Batch 维度：如果第一维是 1，且后面还有数据，就去掉第一维
                # 比如 [1, 9, 8, 3] -> [9, 8, 3]
                # 比如 [1, 5, 7] -> [5, 7]
                if cur_preds.shape[0] == 1 and cur_preds.ndim > 1:
                    cur_preds = cur_preds[0]
                    cur_scores = cur_scores[0]
                
                # 再次检查 score 是否变成了标量 (当只有1个物体时)
                cur_scores = np.atleast_1d(cur_scores)
                # 如果 box 变成了 [8, 3] 或 [7] (只有1个物体)，需要加回一个维度变成 [1, 8, 3] 或 [1, 7]
                # 判断标准：pred 的第0维长度应该等于 scores 的长度
                if len(cur_preds) != len(cur_scores):
                    # 说明被降维了，强制加一个维度
                    cur_preds = np.expand_dims(cur_preds, axis=0)

            else:
                cur_preds = np.array([])
                cur_scores = np.array([])

            # --- 2. 处理真值 (Ground Truth) ---
            if gt_box_tensor is not None and gt_box_tensor.numel() > 0:
                cur_gts = gt_box_tensor.detach().cpu().numpy()
                if cur_gts.shape[0] == 1 and cur_gts.ndim > 1:
                    cur_gts = cur_gts[0]
                
                # 同样检查单物体降维问题
                if cur_gts.ndim == 1 or (cur_gts.ndim == 2 and cur_gts.shape[1] == 3): 
                     # 这里的逻辑比较灵活，主要看是否和预期的 N 一致，简单起见：
                     if cur_gts.ndim == 1: cur_gts = np.expand_dims(cur_gts, axis=0)
            else:
                cur_gts = np.array([])

            # --- 3. 过滤 GT Padding (全0行) ---
            # 兼容 [N, 7] 和 [N, 8, 3] 的全0判断
            if cur_gts.size > 0:
                # 展平后判断每一行是否全为0
                # reshape(N, -1) 把 [N, 8, 3] 变成 [N, 24]，把 [N, 7] 变成 [N, 7]
                cur_gts_flat = cur_gts.reshape(cur_gts.shape[0], -1)
                valid_gt_mask = np.any(cur_gts_flat != 0, axis=1)
                valid_gts = cur_gts[valid_gt_mask]
            else:
                valid_gts = np.array([])

            # --- 4. 数据存入字典 ---
            frame_record = {
                "frame_id": i+169,
                "preds": [], 
                "gts": []    
            }

            def to_list(arr): return arr.tolist()

            # 保存预测框
            if len(cur_preds) > 0:
                # 确保两个数组长度对齐，取最小值防止溢出
                num_preds = min(len(cur_preds), len(cur_scores))
                for p_idx in range(num_preds):
                    frame_record["preds"].append({
                        "box": to_list(cur_preds[p_idx]), 
                        "score": float(cur_scores[p_idx]) 
                    })

            # 保存真值框
            if len(valid_gts) > 0:
                for gt_idx in range(len(valid_gts)):
                    frame_record["gts"].append({
                        "box": to_list(valid_gts[gt_idx])
                    })
            
            full_frame_analysis[i] = frame_record
            
            # ==========================================
            '''
            
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        vis_pcd,
                        mode='constant'
                        )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections)
    if opt.show_sequence:
        vis.destroy_window()

    
    # Save the detailed analysis
    #json_path = os.path.join(opt.model_dir, 'raw_detection_results.json')
    #with open(json_path, 'w') as f:
    #    json.dump(full_frame_analysis, f, indent=4)
    #print(f"原始检测数据已保存至: {json_path}")
    


if __name__ == '__main__':
    main()
