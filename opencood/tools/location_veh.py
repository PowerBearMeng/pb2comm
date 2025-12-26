# -*- coding: utf-8 -*-
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

# --- 新增函数：绘制单个框 ---
def draw_box(ax, x, y, w, l, yaw, color, linewidth=1.5, label=None):
    corners = get_corners(x, y, w, l, yaw)
    corners = np.vstack([corners, corners[0]]) # 闭环
    ax.plot(corners[:, 0], corners[:, 1], c=color, linewidth=linewidth, label=label)
    # 画一个箭头指示方向
    arrow_len = l / 2 + 1.0
    ax.arrow(x, y, math.cos(yaw)*arrow_len, math.sin(yaw)*arrow_len, 
             head_width=1.0, head_length=1.0, fc=color, ec=color)

def visualize_and_save(comm_map, lidar_points, gt_boxes, gt_mask, ego_pose, lidar_range, save_dir, frame_id, threshold):
    """
    新增参数: ego_pose (x, y, yaw) 在路侧坐标系下的位置
    """
    plt.figure(figsize=(24, 10))
    x_min, y_min, x_max, y_max = lidar_range[0], lidar_range[1], lidar_range[3], lidar_range[4]

    # 判断 Ego 是否在视野范围内
    ego_x, ego_y, ego_yaw = ego_pose
    in_fov = (x_min < ego_x < x_max) and (y_min < ego_y < y_max)
    fov_text = "Inside FOV" if in_fov else "Outside FOV"

    # --- 左图：置信度热力图 ---
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(comm_map, cmap='jet', origin='lower', vmin=0, vmax=1, 
               extent=[x_min, x_max, y_min, y_max]) 
    
    # 在置信图上也画出 Ego 的位置（用蓝色星星表示中心）
    plt.scatter(ego_x, ego_y, s=200, c='cyan', marker='*', edgecolors='white', label='Ego Car')
    
    plt.title(f"Conf Map (Frame {frame_id})\nEgo: ({ego_x:.1f}, {ego_y:.1f}) [{fov_text}]")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend(loc='upper right') # 显示图例

    # --- 右图：原始点云 BEV + GT Boxes + Ego Box ---
    ax2 = plt.subplot(1, 2, 2)
    
    # 1. 画点云
    if lidar_points is not None:
        if isinstance(lidar_points, torch.Tensor):
            lidar_points = lidar_points.cpu().numpy()
        mask = (lidar_points[:, 0] > x_min) & (lidar_points[:, 0] < x_max) & \
               (lidar_points[:, 1] > y_min) & (lidar_points[:, 1] < y_max)
        valid_points = lidar_points[mask]
        plt.scatter(valid_points[:, 0], valid_points[:, 1], s=0.5, c='gray', alpha=0.5)
    
    # 2. 画路侧视角的 GT Boxes (绿色)
    if gt_boxes is not None:
        if isinstance(gt_boxes, torch.Tensor): gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor): gt_mask = gt_mask.cpu().numpy()
        
        for k in range(gt_boxes.shape[0]):
            if gt_mask[k] == 0: continue
            # standard format: [x, y, z, w, l, h, yaw] or similar. 
            # 假设 index 4=w, 5=l, 6=yaw (根据你的代码逻辑)
            draw_box(ax2, gt_boxes[k, 0], gt_boxes[k, 1], gt_boxes[k, 4], gt_boxes[k, 5], gt_boxes[k, 6], '#00FF00')

    # 3. 画 Ego Vehicle (蓝色)
    # 假设 Ego 尺寸: 长4.8米, 宽1.8米 (标准轿车)
    draw_box(ax2, ego_x, ego_y, 1.8, 4.8, ego_yaw, 'cyan', linewidth=2.0, label='Ego Car')
    
    # 设置样式
    plt.gca().set_facecolor('black')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("LiDAR BEV + GT(Green) + Ego(Blue)")
    plt.xlabel("X (meters)")
    plt.legend(loc='upper right')

    # 保存
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"vis_{frame_id}_location.png")
    plt.tight_layout()
    plt.savefig(save_path)
    # print(f"Saved {save_path}") # 减少打印刷屏
    plt.close()

def main():
    # ... (前面的 setup 代码保持不变) ...
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)
    if opt.comm_thre is not None:
        hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre
    current_thre = hypes['model']['args']['fusion_args']['communication']['thre']
    lidar_range = hypes['postprocess']['gt_range']

    print('Building Dataset...')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset, batch_size=1, num_workers=4,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False, pin_memory=False, drop_last=False)

    print('Creating Model...')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    save_dir = os.path.join(opt.model_dir, 'vis_confidence_maps')
    print(f'Results will be saved to: {save_dir}')

    print('Start Processing...')
    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if batch_data is None: continue
            
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            model_output = model(batch_data['ego'])
            
            # 1. 获取置信图 (同前)
            if 'psm_single_i' in model_output:
                infra_conf = torch.sigmoid(model_output['psm_single_i'])[0]
                conf_map = torch.max(infra_conf, dim=0)[0].cpu().numpy()
            else:
                continue

            # 2. 获取路侧点云 & 计算 Ego 位置
            ego_pose_infra_frame = (0, 0, 0) # 默认值
            raw_lidar = None

            if 'origin_lidar_i' in batch_data['ego']:
                points_ego = batch_data['ego']['origin_lidar_i'][0] # GPU Tensor
                poses = batch_data['ego']['lidar_pose']
                
                # --- 转 CPU Numpy ---
                pose_ego = poses[0].cpu().numpy()
                pose_infra = poses[1].cpu().numpy()
                
                # --- 计算变换矩阵 (Numpy) ---
                T_ego_to_infra_np = x1_to_x2(pose_ego, pose_infra)
                
                # A. 变换点云 (转 Tensor -> GPU -> 计算 -> CPU)
                T_tensor = torch.from_numpy(T_ego_to_infra_np).to(points_ego.device).float()
                points_infra = project_points_by_matrix_torch(points_ego[:, :3], T_tensor)
                raw_lidar = points_infra.cpu().numpy()
                
                # B. 计算 Ego 在 Infra 坐标系下的位置 (在 CPU 上计算即可)
                # Ego 在自己坐标系下是原点 (0,0,0)
                ego_center_local = np.array([0, 0, 0, 1]).reshape(4, 1)
                ego_center_infra = np.dot(T_ego_to_infra_np, ego_center_local).flatten() # [x, y, z, 1]
                
                # 计算 Ego 的 Yaw (朝向)
                # 取一个前方点 (1,0,0) 变换后减去原点变换后，得到方向向量
                ego_front_local = np.array([1, 0, 0, 1]).reshape(4, 1)
                ego_front_infra = np.dot(T_ego_to_infra_np, ego_front_local).flatten()
                
                dx = ego_front_infra[0] - ego_center_infra[0]
                dy = ego_front_infra[1] - ego_center_infra[1]
                ego_yaw_infra = np.arctan2(dy, dx)
                
                # 最终 Ego 位置: (x, y, yaw)
                ego_pose_infra_frame = (ego_center_infra[0], ego_center_infra[1], ego_yaw_infra)
                
            # 3. 获取 GT (同前)
            gt_boxes = batch_data['ego'].get('object_bbx_center_single_i', [None])[0]
            gt_mask = batch_data['ego'].get('object_bbx_mask_single_i', [None])[0]

            frame_id = f"{i:04d}_infra"
            visualize_and_save(conf_map, raw_lidar, gt_boxes, gt_mask, 
                               ego_pose_infra_frame, lidar_range, save_dir, frame_id, current_thre)

    print("Visualization Finished!")

if __name__ == '__main__':
    main()