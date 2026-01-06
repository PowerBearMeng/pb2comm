# -*- coding: utf-8 -*-
# 用法: python opencood/tools/vis_blind.py --model_dir /path/to/your/log_folder

import argparse
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.box_utils import project_points_by_matrix_torch

def test_parser():
    parser = argparse.ArgumentParser(description="Blind Spot Visualization")
    # 默认路径已保留为你之前的设置
    parser.add_argument('--model_dir', type=str, default="/home/yty/mfh/code/inter/Where2comm/opencood/logs/dair_where2comm_max_multiscale_resnet_2025_12_26_21_37_17",
                        help='Path to the training log directory')
    parser.add_argument('--comm_thre', type=float, default=None,
                        help='Override communication threshold')
    opt = parser.parse_args()
    return opt

# ==========================================
# 1. 几何辅助函数
# ==========================================

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

def draw_box(ax, x, y, w, l, yaw, color, linewidth=1.5, label=None):
    corners = get_corners(x, y, w, l, yaw)
    corners = np.vstack([corners, corners[0]])
    ax.plot(corners[:, 0], corners[:, 1], c=color, linewidth=linewidth, label=label)

# ==========================================
# 2. 盲区计算核心算法 (用于 Plot 3 对比)
# ==========================================

def points_to_occupancy_grid(points, x_range, y_range, voxel_size, ground_filter_thre=-1.5):
    points = points[points[:, 2] > ground_filter_thre]

    W = int((x_range[1] - x_range[0]) / voxel_size)
    H = int((y_range[1] - y_range[0]) / voxel_size)
    grid = np.zeros((H, W), dtype=np.uint8)
    
    mask = (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1]) & \
           (points[:, 1] > y_range[0]) & (points[:, 1] < y_range[1])
    valid_points = points[mask]
    
    if len(valid_points) == 0:
        return grid, W, H

    x_idxs = ((valid_points[:, 0] - x_range[0]) / voxel_size).astype(int)
    y_idxs = ((valid_points[:, 1] - y_range[0]) / voxel_size).astype(int)
    x_idxs = np.clip(x_idxs, 0, W - 1)
    y_idxs = np.clip(y_idxs, 0, H - 1)
    
    grid[y_idxs, x_idxs] = 1
    return grid, W, H

def compute_precise_visibility(occupancy_grid, start_pos, x_range, y_range, voxel_size):
    H, W = occupancy_grid.shape
    start_x = int((start_pos[0] - x_range[0]) / voxel_size)
    start_y = int((start_pos[1] - y_range[0]) / voxel_size)
    
    if not (0 <= start_x < W and 0 <= start_y < H):
        return np.zeros((H, W), dtype=np.uint8)

    ys, xs = np.indices((H, W))
    dy = ys - start_y
    dx = xs - start_x
    dist_map = np.sqrt(dx**2 + dy**2)
    angle_map = np.arctan2(dy, dx)
    
    obs_mask = (occupancy_grid > 0)
    if not np.any(obs_mask):
        return np.ones((H, W), dtype=np.uint8)

    obs_dists = dist_map[obs_mask]
    obs_angles = angle_map[obs_mask]
    
    # 自我遮挡剔除: 半径 3.0 米内的障碍物视为透明
    ignore_radius_pixel = 3.0 / voxel_size 
    valid_blocker_mask = obs_dists > ignore_radius_pixel
    
    valid_obs_dists = obs_dists[valid_blocker_mask]
    valid_obs_angles = obs_angles[valid_blocker_mask]
    
    num_bins = 3600
    bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    
    obs_bin_idxs = np.digitize(valid_obs_angles, bin_edges) - 1
    obs_bin_idxs = np.clip(obs_bin_idxs, 0, num_bins - 1)
    
    min_obs_dist_per_bin = np.full(num_bins, np.inf)
    if len(valid_obs_dists) > 0:
        np.minimum.at(min_obs_dist_per_bin, obs_bin_idxs, valid_obs_dists)
            
    pixel_bin_idxs = np.digitize(angle_map.ravel(), bin_edges) - 1
    pixel_bin_idxs = np.clip(pixel_bin_idxs, 0, num_bins - 1)
    
    block_dists = min_obs_dist_per_bin[pixel_bin_idxs].reshape(H, W)
    
    vis_map = np.zeros((H, W), dtype=np.uint8)
    visible_free_mask = (dist_map < (block_dists - 1.0))
    vis_map[visible_free_mask] = 1
    
    visible_obs_mask = obs_mask & (dist_map <= (block_dists + 1.0))
    vis_map[visible_obs_mask] = 2
    
    return vis_map

# ==========================================
# 3. 可视化主函数 (2x2 布局)
# ==========================================

def visualize_and_save(comm_map, real_masked_map, lidar_points, gt_boxes, gt_mask, ego_pose, lidar_range, save_dir, frame_id):
    """
    Args:
        comm_map: 原始路侧置信度图 (Plot 2)
        real_masked_map: 模型实际使用的特征图 (Plot 4)
    """
    fig = plt.figure(figsize=(16, 16), dpi=100)
    x_min, y_min, x_max, y_max = lidar_range[0], lidar_range[1], lidar_range[3], lidar_range[4]
    ego_x, ego_y, ego_yaw = ego_pose

    # --- 预计算 Plot 3 所需数据 ---
    vis_map = None
    grid = None
    if lidar_points is not None:
        voxel_size = 0.4 
        grid, W, H = points_to_occupancy_grid(lidar_points, [x_min, x_max], [y_min, y_max], voxel_size)
        vis_map = compute_precise_visibility(grid, (ego_x, ego_y), [x_min, x_max], [y_min, y_max], voxel_size)

    # ----------------------------------------------------
    # Plot 1 (左上): 路侧点云 + GT (上帝视角)
    # ----------------------------------------------------
    ax1 = plt.subplot(2, 2, 1)
    plt.gca().set_facecolor('black')
    if lidar_points is not None:
        pts = lidar_points[::2] # 降采样加速
        mask = (pts[:,0]>x_min)&(pts[:,0]<x_max)&(pts[:,1]>y_min)&(pts[:,1]<y_max)
        plt.scatter(pts[mask,0], pts[mask,1], s=0.5, c='gray', alpha=0.5)
    
    draw_box(ax1, ego_x, ego_y, 2.0, 4.8, ego_yaw, 'cyan', linewidth=2.5, label='Ego')
    
    if gt_boxes is not None:
        if isinstance(gt_boxes, torch.Tensor): gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor): gt_mask = gt_mask.cpu().numpy()
            
        for k in range(gt_boxes.shape[0]):
            if gt_mask is not None and gt_mask[k] == 0: continue
            # color: Green
            draw_box(ax1, gt_boxes[k,0], gt_boxes[k,1], gt_boxes[k,4], gt_boxes[k,5], gt_boxes[k,6], '#00FF00')
    
    plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
    plt.title("1. Roadside PointCloud & GT")

    # ----------------------------------------------------
    # Plot 2 (右上): 路侧 BEV Confidence Map (原始)
    # ----------------------------------------------------
    ax2 = plt.subplot(2, 2, 2)
    if comm_map is not None:
        plt.imshow(comm_map, cmap='jet', origin='lower', vmin=0, vmax=1, extent=[x_min, x_max, y_min, y_max])
        plt.title("2. Raw Confidence Map (Full Information)")
    else:
        plt.title("No Confidence Map")
    plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)

    # ----------------------------------------------------
    # Plot 3 (左下): 理论计算的盲区图 (Geometric Blind Spot)
    # ----------------------------------------------------
    ax3 = plt.subplot(2, 2, 3)
    if vis_map is not None:
        # 0=Blind(Gray), 1=Free(White), 2=VisibleObs(Green)
        display_img = np.zeros((H, W, 3), dtype=np.uint8)
        display_img[vis_map == 0] = [50, 50, 50]    # 盲区 (Gray)
        display_img[vis_map == 1] = [255, 255, 255] # 可见空地 (White)
        
        visible_obs_mask = (grid > 0) & (vis_map == 2)
        display_img[visible_obs_mask] = [0, 255, 0] # 可见障碍物 (Green)
        
        hidden_obs_mask = (grid > 0) & (vis_map == 0)
        display_img[hidden_obs_mask] = [255, 255, 0] # 隐藏障碍物 (Yellow)

        plt.imshow(display_img, origin='lower', extent=[x_min, x_max, y_min, y_max])
        plt.scatter(ego_x, ego_y, s=150, c='cyan', marker='*', edgecolors='black')
        plt.title("3. Geometric Blind Spot Estimation\n(Gray=Blind, White=Visible, Yellow=Hidden Obs)")
    else:
        plt.title("No Point Cloud Data")
    plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)

    # ----------------------------------------------------
    # Plot 4 (右下): 模型实际使用的特征图 (Real In-Model Features)
    # ----------------------------------------------------
    ax4 = plt.subplot(2, 2, 4)
    if real_masked_map is not None:
        # 这个图应该是模型内部计算完 Mask 后剩下的东西
        # 预期效果：非盲区(Visible)应该是全黑(0)，只有盲区(Blind)部分有热力值
        plt.imshow(real_masked_map, cmap='jet', origin='lower', vmin=0, vmax=1, extent=[x_min, x_max, y_min, y_max])
        plt.title("4. Actual Feature Used in Model\n(Transmitted = Blind Spot Only)")
    else:
        plt.title("No Masked Map Found in Model Output\n(Did you modify where2comm_mfh.py?)")
        
    plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)

    # 保存
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"vis_{frame_id}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# ==========================================
# 4. 主程序
# ==========================================

def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)
    
    if opt.comm_thre is not None:
        hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre
    
    # 必要的预处理参数
    lidar_range = hypes['postprocess']['gt_range']

    print('Building Dataset...')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=4,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False, pin_memory=False, drop_last=False)

    print('Creating Model...')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    save_dir = os.path.join(opt.model_dir, 'vis_blind_spot_validation')
    print(f'Results will be saved to: {save_dir}')

    print('Start Processing...')
    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if batch_data is None: continue
            
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            
            # --- 模型推理 ---
            model_output = model(batch_data['ego'])
            
            # 1. 原始路侧置信度图 (Plot 2)
            raw_conf_map = None
            if 'psm_single_i' in model_output:
                infra_conf = torch.sigmoid(model_output['psm_single_i'])[0]
                # Max over anchors
                raw_conf_map = torch.max(infra_conf, dim=0)[0].cpu().numpy()
            
            # 2. 模型实际 Mask 过的特征图 (Plot 4) --- [关键]
            real_masked_map = None
            masked_tensor = model_output['masked_comm_maps'][0]
            if 'masked_comm_maps' in model_output and model_output['masked_comm_maps'] is not None:
                # 获取 list 中的第一个 (Batch 0)
                masked_tensor = model_output['masked_comm_maps'][0] 
                if masked_tensor.shape[0] > 1:
                    # 取第一个邻居 (Index 1), 并去掉 Channel 维度 -> (H, W)
                    # 这里的 [1, 0] 意思是: 第2个车, 第1个通道
                    target_tensor = masked_tensor[1, 0]
                    print(target_tensor.shape)
                else:
                    # 如果场景里只有自车，那就只能画自车了
                    print("--------------------")
                    target_tensor = masked_tensor[0, 0]
                
                # 现在的 shape 可能是 (1, H, W)
                if len(target_tensor.shape) == 3:
                    target_tensor = target_tensor.squeeze(0)
                real_masked_map = target_tensor.cpu().numpy()
            
            # 3. 路侧点云转换 (用于 Plot 1 & 3)
            raw_lidar = None
            ego_pose_infra = (0, 0, 0)
            if 'origin_lidar_i' in batch_data['ego']:
                pts_ego = batch_data['ego']['origin_lidar_i'][0]
                poses = batch_data['ego']['lidar_pose']
                pose_ego = poses[0].cpu().numpy()
                pose_infra = poses[1].cpu().numpy()
                
                T_ego_to_infra = x1_to_x2(pose_ego, pose_infra)
                T_tensor = torch.from_numpy(T_ego_to_infra).to(pts_ego.device).float()
                
                raw_lidar = project_points_by_matrix_torch(pts_ego[:, :3], T_tensor).cpu().numpy()
                
                # 计算 Ego 在 Infra 坐标系下的位置
                ego_center = np.dot(T_ego_to_infra, np.array([0,0,0,1]))
                ego_front = np.dot(T_ego_to_infra, np.array([1,0,0,1]))
                yaw = np.arctan2(ego_front[1]-ego_center[1], ego_front[0]-ego_center[0])
                ego_pose_infra = (ego_center[0], ego_center[1], yaw)

            # 4. GT Boxes
            gt_boxes = batch_data['ego'].get('object_bbx_center_single_i', [None])[0]
            gt_mask = batch_data['ego'].get('object_bbx_mask_single_i', [None])[0]

            frame_id = f"{i:04d}_validation"
            visualize_and_save(raw_conf_map, real_masked_map, raw_lidar, gt_boxes, gt_mask, 
                               ego_pose_infra, lidar_range, save_dir, frame_id)
            plt.close('all')

    print("Visualization Finished!")

if __name__ == '__main__':
    main()