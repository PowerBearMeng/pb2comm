# -*- coding: utf-8 -*-
import argparse
import os
import torch
import math
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

def get_corners(x, y, w, l, yaw):
    """计算旋转矩形的四个角点"""
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    hw = w / 2
    hl = l / 2

    bev_corners[0, 0] = x + cos_yaw * hl - sin_yaw * hw
    bev_corners[0, 1] = y + sin_yaw * hl + cos_yaw * hw
    bev_corners[1, 0] = x - cos_yaw * hl - sin_yaw * hw
    bev_corners[1, 1] = y - sin_yaw * hl + cos_yaw * hw
    bev_corners[2, 0] = x - cos_yaw * hl + sin_yaw * hw
    bev_corners[2, 1] = y - sin_yaw * hl - cos_yaw * hw
    bev_corners[3, 0] = x + cos_yaw * hl + sin_yaw * hw
    bev_corners[3, 1] = y + sin_yaw * hl - cos_yaw * hw
    return bev_corners

def generate_gt_mask(gt_boxes, gt_mask, shape, lidar_range):
    """将真值框转换为像素级的 Binary Mask"""
    H, W = shape
    x_min, y_min, x_max, y_max = lidar_range[0], lidar_range[1], lidar_range[3], lidar_range[4]
    
    res_x = (x_max - x_min) / W
    res_y = (y_max - y_min) / H
    
    mask_img = np.zeros((H, W), dtype=np.uint8)
    
    if gt_boxes is None or len(gt_boxes) == 0:
        return mask_img

    for k in range(gt_boxes.shape[0]):
        if gt_mask is not None and gt_mask[k] == 0:
            continue
            
        x, y = gt_boxes[k, 0], gt_boxes[k, 1]
        w = gt_boxes[k, 4] 
        l = gt_boxes[k, 5]
        yaw = gt_boxes[k, 6]
        
        corners = get_corners(x, y, w, l, yaw)
        
        pts = np.zeros((4, 2), dtype=np.int32)
        pts[:, 0] = ((corners[:, 0] - x_min) / res_x).astype(np.int32)
        pts[:, 1] = ((corners[:, 1] - y_min) / res_y).astype(np.int32)
        
        cv2.fillPoly(mask_img, [pts], 1)
        
    return mask_img

def main():
    # ---------------- 配置部分 ----------------
    parser = argparse.ArgumentParser(description="Confidence Map Evaluation")
    parser.add_argument('--model_dir', type=str, default="/home/yty/mfh/code/inter/Where2comm/opencood/logs/dair_where2comm_max_multiscale_resnet_2025_12_17_18_03_32", help='Path to log dir')
    parser.add_argument('--comm_thre', type=float, default=0.01, help='Threshold to evaluate')
    opt = parser.parse_args()
    
    hypes = yaml_utils.load_yaml(None, opt)
    hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre
    
    print(f"--- Evaluating with Threshold: {opt.comm_thre} ---")

    # 构建 Dataset 和 Loader
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset, batch_size=1, num_workers=4,
                             collate_fn=opencood_dataset.collate_batch_test, shuffle=False)

    # 构建模型
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    # 获取 Lidar Range
    lidar_range = hypes['postprocess']['gt_range']

    # ---------------- 统计变量 ----------------
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_transmitted_pixels = 0 # 对应您之前的 total_pixels (TP + FP)
    total_map_area_pixels = 0    # <--- 新增: 用于统计所有帧的地图总面积
    
    valid_frames = 0

    print('Start Processing...')
    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if batch_data is None: continue

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            model_output = model(batch_data['ego'])

            if 'comm_maps' not in model_output or model_output['comm_maps'] is None:
                continue

            # 1. 获取预测图 (Prediction)
            conf_map = model_output['comm_maps'][0] 
            conf_map = conf_map.squeeze().cpu().numpy()
            if len(conf_map.shape) == 3:
                conf_map = np.max(conf_map, axis=0)
            
            # 生成二值化的预测掩码 (0 或 1)
            pred_mask = (conf_map > opt.comm_thre).astype(np.uint8)

            # 2. 获取真值 (Ground Truth)
            if 'object_bbx_center' in batch_data['ego']:
                gt_boxes = batch_data['ego']['object_bbx_center'][0].cpu().numpy()
                gt_mask_code = batch_data['ego']['object_bbx_mask'][0].cpu().numpy()
            else:
                gt_boxes = None
                gt_mask_code = None

            # 生成真值掩码 (0 或 1)
            gt_img_mask = generate_gt_mask(gt_boxes, gt_mask_code, pred_mask.shape, lidar_range)
            
            # 3. 核心计算
            tp = np.sum((pred_mask == 1) & (gt_img_mask == 1))
            fp = np.sum((pred_mask == 1) & (gt_img_mask == 0))
            fn = np.sum((pred_mask == 0) & (gt_img_mask == 1))
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # 统计传输相关
            current_transmitted = tp + fp         # 本帧传输的像素数
            total_transmitted_pixels += current_transmitted
            
            current_map_area = pred_mask.size     # <--- 新增: 本帧地图总像素 (H * W)
            total_map_area_pixels += current_map_area # <--- 新增: 累加到总面积
            
            valid_frames += 1

    # ---------------- 最终结果计算 ----------------
    if valid_frames > 0:
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        
        # 计算传输比例
        transmission_rate = total_transmitted_pixels / (total_map_area_pixels + 1e-6) # <--- 新增计算

        print("\n" + "="*40)
        print(f"Evaluation Report (Threshold > {opt.comm_thre})")
        print("="*40)
        print(f"Total Frames Processed: {valid_frames}")
        print(f"Pixel Precision (准确率): {precision:.4f}")
        print(f"Pixel Recall    (召回率): {recall:.4f}")
        print(f"Pixel F1 Score  (综合分): {f1_score:.4f}")
        print("-" * 40)
        print(f"Transmission Rate (传输比例): {transmission_rate:.2%}") # <--- 新增输出
        print("-" * 40)
        print(f"Interpretation:")
        print(f"Precision: 选出的像素中，有 {precision*100:.2f}% 确实是在车框里的。")
        print(f"Recall:    真实的车辆区域中，有 {recall*100:.2f}% 被成功选中并传输了。")
        print(f"TransRate: 平均每张地图传输了 {transmission_rate*100:.2f}% 的区域面积。") # <--- 新增解释
        print("="*40)
    else:
        print("No valid frames evaluated.")

if __name__ == '__main__':
    main()