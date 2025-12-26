# -*- coding: utf-8 -*-
# 文件位置: opencood/tools/vis_confidence.py
# 运行方法: python opencood/tools/vis_confidence.py --model_dir /path/to/your/log_folder

import argparse
import os
import time
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
# 在文件开头添加
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.box_utils import project_points_by_matrix_torch

def test_parser():
    parser = argparse.ArgumentParser(description="Confidence Map Visualization")
    parser.add_argument('--model_dir', type=str, default="/home/yty/mfh/code/inter/Where2comm/opencood/logs/dair_where2comm_max_multiscale_resnet_2025_12_23_14_07_13",
                        help='Path to the training log directory')
    parser.add_argument('--comm_thre', type=float, default=None,
                        help='Override communication threshold')
    opt = parser.parse_args()
    return opt

def get_corners(x, y, w, l, yaw):
    """
    根据中心点和长宽角计算旋转矩形的四个角点
    """
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # 车辆的一半长宽
    hw = w / 2
    hl = l / 2

    # 旋转矩阵逻辑
    # front left
    bev_corners[0, 0] = x + cos_yaw * hl - sin_yaw * hw
    bev_corners[0, 1] = y + sin_yaw * hl + cos_yaw * hw

    # rear left
    bev_corners[1, 0] = x - cos_yaw * hl - sin_yaw * hw
    bev_corners[1, 1] = y - sin_yaw * hl + cos_yaw * hw

    # rear right
    bev_corners[2, 0] = x - cos_yaw * hl + sin_yaw * hw
    bev_corners[2, 1] = y - sin_yaw * hl - cos_yaw * hw

    # front right
    bev_corners[3, 0] = x + cos_yaw * hl + sin_yaw * hw
    bev_corners[3, 1] = y + sin_yaw * hl - cos_yaw * hw

    return bev_corners

def visualize_and_save(comm_map, lidar_points, gt_boxes, gt_mask, lidar_range, save_dir, frame_id, threshold):
    """
    绘制: 左边置信度图，右边点云+真值框
    """
    plt.figure(figsize=(24, 10)) # 调宽一点
    
    # 解析 Lidar Range 用于对齐坐标 [x_min, y_min, z_min, x_max, y_max, z_max]
    x_min, y_min, x_max, y_max = lidar_range[0], lidar_range[1], lidar_range[3], lidar_range[4]

    # --- 左图：置信度热力图 ---
    plt.subplot(1, 2, 1)
    # extent参数让热力图的坐标轴变成真实的米，而不是像素
    # 这里的 comm_map 已经是 (H, W) 的 2D 数组
    plt.imshow(comm_map, cmap='jet', origin='lower', vmin=0, vmax=1, 
               extent=[x_min, x_max, y_min, y_max]) 
    plt.title(f"Confidence Map (Frame {frame_id})\nThreshold: {threshold}")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.colorbar(label='Confidence Score')

    # --- 右图：原始点云 BEV + GT Boxes ---
    ax2 = plt.subplot(1, 2, 2)
    
    # 1. 画点云
    if lidar_points is not None:
        if isinstance(lidar_points, torch.Tensor):
            lidar_points = lidar_points.cpu().numpy()
            
        # 过滤范围
        mask = (lidar_points[:, 0] > x_min) & (lidar_points[:, 0] < x_max) & \
               (lidar_points[:, 1] > y_min) & (lidar_points[:, 1] < y_max)
        valid_points = lidar_points[mask]
        
        plt.scatter(valid_points[:, 0], valid_points[:, 1], s=0.5, c='gray', alpha=0.5)
    
    # 2. 画真值框 (Ground Truth Boxes)
    if gt_boxes is not None:
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
            
        # 遍历每一个框
        for k in range(gt_boxes.shape[0]):
            # 如果 mask 是 0，说明这个位置没有物体（是填充的）
            if gt_mask[k] == 0:
                continue
                
            # gt_box 格式通常是 [x, y, z, l, w, h, yaw] 或者 [x, y, z, dx, dy, dz, yaw]
            # 这里假设是 OpenCOOD 标准格式: [x, y, z, h, w, l, yaw] 或 [x, y, z, dx, dy, dz, yaw]
            # 注意: w 和 l 的顺序可能因数据集而异，画出来如果长宽反了就调换下面代码的 idx
            x, y = gt_boxes[k, 0], gt_boxes[k, 1]
            l = gt_boxes[k, 5] # 假设 index 5 是长 (dx/l)
            w = gt_boxes[k, 4] # 假设 index 4 是宽 (dy/w)
            yaw = gt_boxes[k, 6] # index 6 是角度
            
            corners = get_corners(x, y, w, l, yaw)
            
            # 把最后一个点和第一个点连起来，形成闭环
            corners = np.vstack([corners, corners[0]])
            
            # 画线，使用亮绿色
            plt.plot(corners[:, 0], corners[:, 1], c='#00FF00', linewidth=1.5)

    # 设置样式
    plt.gca().set_facecolor('black')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("LiDAR BEV + GT Boxes (Green)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")

    # 保存
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"vis_{frame_id}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close()

def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)
    
    if opt.comm_thre is not None:
        hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre
    current_thre = hypes['model']['args']['fusion_args']['communication']['thre']

    # 获取 Lidar Range 用于对齐
    lidar_range = hypes['postprocess']['gt_range'] # e.g. [-100, -40, -3, 100, 40, 1]

    print('Building Dataset...')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=4,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model...')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f'Loading weights from {opt.model_dir}')
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    save_dir = os.path.join(opt.model_dir, 'vis_confidence_maps')
    print(f'Results will be saved to: {save_dir}')

    print('Start Processing...')
    # ... (前面的代码不变) ...
    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if batch_data is None:
            continue
            
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            model_output = model(batch_data['ego'])
            
            # -----------------------------------------------------------
            # 步骤 1: 获取路侧置信度图 (Infra Confidence Map)
            # -----------------------------------------------------------
            if 'psm_single_i' in model_output:
                # psm_single_i 的形状通常是 (1, Anchor_Num, H, W)
                infra_logits = model_output['psm_single_i']
                print(infra_logits.shape)
                infra_conf = torch.sigmoid(infra_logits)[0] # 取 Batch 0 -> (Anchor_Num, H, W)
                
                # 压缩 Anchor 维度，取最大值，得到 (H, W) 的热力图
                conf_map = torch.max(infra_conf, dim=0)[0]
                conf_map = conf_map.cpu().numpy()
            else:
                print("Warning: No psm_single_i found in output.")
                continue

            # -----------------------------------------------------------
            # 步骤 2: 获取路侧点云并转换坐标 (Points: Ego -> Infra)
            # -----------------------------------------------------------
            # batch_data['ego']['origin_lidar_i'] 是路侧点云，但处于 Ego 坐标系
            if 'origin_lidar_i' in batch_data['ego']:
                points_ego = batch_data['ego']['origin_lidar_i'][0] # (N, 4)
                
                # 获取位姿: Index 0 是车(Ego), Index 1 是路(Infra)
                # lidar_pose shape: (2, 6)
                poses = batch_data['ego']['lidar_pose']
                pose_ego = poses[0]
                pose_infra = poses[1]
                
                # --- 修改开始 ---
                # 必须先转回 CPU 并转为 numpy，因为 x1_to_x2 是 numpy 操作
                pose_ego = poses[0].cpu().numpy()
                pose_infra = poses[1].cpu().numpy()
                # --- 修改结束 ---

                # 计算变换矩阵: 从 Ego 变回 Infra
                # x1_to_x2(x1, x2) 计算的是把点从 x1 坐标系变换到 x2 坐标系的矩阵
                T_ego_to_infra = x1_to_x2(pose_ego, pose_infra)
                T_ego_to_infra = torch.from_numpy(T_ego_to_infra).to(points_ego.device).float()
                
                # 执行变换: points_infra = T_ego_to_infra * points_ego
                points_infra = project_points_by_matrix_torch(points_ego[:, :3], T_ego_to_infra)
                
                # 转为 numpy 用于画图
                raw_lidar = points_infra.cpu().numpy()
            else:
                raw_lidar = None

            # -----------------------------------------------------------
            # 步骤 3: 获取路侧视角的真值框 (GT Boxes in Infra Frame)
            # -----------------------------------------------------------
            # object_bbx_center_single_i 通常已经是路侧坐标系下的真值了
            if 'object_bbx_center_single_i' in batch_data['ego']:
                gt_boxes = batch_data['ego']['object_bbx_center_single_i'][0]
                gt_mask = batch_data['ego']['object_bbx_mask_single_i'][0]
            else:
                gt_boxes = None
                gt_mask = None

            # -----------------------------------------------------------
            # 步骤 4: 画图
            # -----------------------------------------------------------
            frame_id = f"{i:04d}_infra"
            visualize_and_save(conf_map, raw_lidar, gt_boxes, gt_mask, lidar_range, save_dir, frame_id, current_thre)

    print("Visualization Finished!")

if __name__ == '__main__':
    main()