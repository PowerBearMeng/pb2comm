# -*- coding: utf-8 -*-
# 用法: python opencood/tools/vis_blind_spot.py --model_dir /path/to/your/log_folder

import argparse
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2  # 需要 pip install opencv-python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# OpenCOOD 依赖
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.box_utils import project_points_by_matrix_torch

def test_parser():
    parser = argparse.ArgumentParser(description="Blind Spot Visualization")
    # 请修改默认路径为你自己的模型路径
    parser.add_argument('--model_dir', type=str, default="/home/yty/mfh/code/inter/Where2comm/opencood/logs/dair_where2comm_max_multiscale_resnet_2025_12_23_14_07_13",
                        help='Path to the training log directory')
    parser.add_argument('--comm_thre', type=float, default=None,
                        help='Override communication threshold')
    opt = parser.parse_args()
    return opt

# ==========================================
# 1. 几何与绘图辅助函数
# ==========================================

def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    hw = w / 2
    hl = l / 2

    # Front Left
    bev_corners[0, 0] = x + cos_yaw * hl - sin_yaw * hw
    bev_corners[0, 1] = y + sin_yaw * hl + cos_yaw * hw
    # Rear Left
    bev_corners[1, 0] = x - cos_yaw * hl - sin_yaw * hw
    bev_corners[1, 1] = y - sin_yaw * hl + cos_yaw * hw
    # Rear Right
    bev_corners[2, 0] = x - cos_yaw * hl + sin_yaw * hw
    bev_corners[2, 1] = y - sin_yaw * hl - cos_yaw * hw
    # Front Right
    bev_corners[3, 0] = x + cos_yaw * hl + sin_yaw * hw
    bev_corners[3, 1] = y + sin_yaw * hl - cos_yaw * hw

    return bev_corners

def draw_box(ax, x, y, w, l, yaw, color, linewidth=1.5, label=None):
    corners = get_corners(x, y, w, l, yaw)
    corners = np.vstack([corners, corners[0]]) # 闭环
    ax.plot(corners[:, 0], corners[:, 1], c=color, linewidth=linewidth, label=label)
    # 箭头指示车头
    arrow_len = l / 2 + 1.0
    ax.arrow(x, y, math.cos(yaw)*arrow_len, math.sin(yaw)*arrow_len, 
             head_width=1.0, head_length=1.0, fc=color, ec=color)

# ==========================================
# 2. 盲区计算核心算法 (极坐标优化版)
# ==========================================

def points_to_occupancy_grid(points, x_range, y_range, voxel_size):
    """
    点云 -> 占据栅格 (0=Free, 1=Occupied)
    """
    # -------------------------------------------------------------
    # 【关键修改】去除地面点云
    # -------------------------------------------------------------
    # 假设地面在 -1.7m 左右，我们保留 Z > -1.5m 的点作为障碍物
    # 这个阈值请根据你刚才 check_ego_z_height.py 跑出来的结果填
    # 如果是路侧雷达（通常很高），地面可能在 -4.5m，那就填 -4.2m
    
    # 这里的例子假设是车端数据 (地面约 -1.7m)
    ground_filter_thre = -1.5 
    
    # 如果是路侧数据，请填你 check_z_height.py 测出的值 (比如 -4.0)
    # ground_filter_thre = -4.0 
    
    points = points[points[:, 2] > ground_filter_thre]
    # -------------------------------------------------------------

    W = int((x_range[1] - x_range[0]) / voxel_size)
    H = int((y_range[1] - y_range[0]) / voxel_size)
    
    grid = np.zeros((H, W), dtype=np.uint8)
    
    # ... (后续代码不变) ...
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
    
    # 注意：为了让盲区计算更严谨，这里不要膨胀(dilate)，或者只膨胀一点点
    # grid = cv2.dilate(grid, np.ones((3,3), np.uint8), iterations=1)
    
    return grid, W, H

def compute_precise_visibility(occupancy_grid, start_pos, x_range, y_range, voxel_size):
    """
    极坐标法计算可视区域
    Returns:
        vis_map: (H, W)
        0 = 盲区 (Blind)
        1 = 可见空地 (Visible Free)
        2 = 可见障碍物表面 (Visible Obstacle)
    """
    H, W = occupancy_grid.shape
    
    # 车的位置转为 Grid 索引
    start_x = int((start_pos[0] - x_range[0]) / voxel_size)
    start_y = int((start_pos[1] - y_range[0]) / voxel_size)
    
    if not (0 <= start_x < W and 0 <= start_y < H):
        # 车在地图外，简单处理为全黑或全白，这里返回全黑表示无效/盲区
        return np.zeros((H, W), dtype=np.uint8)

    # 1. 预计算所有像素相对于车的距离和角度
    ys, xs = np.indices((H, W))
    dy = ys - start_y
    dx = xs - start_x
    dist_map = np.sqrt(dx**2 + dy**2)
    angle_map = np.arctan2(dy, dx) # (-pi, pi)
    
    # 2. 提取障碍物信息
    obs_mask = (occupancy_grid > 0)
    if not np.any(obs_mask):
        return np.ones((H, W), dtype=np.uint8) # 无障碍，全可见

    obs_dists = dist_map[obs_mask]
    obs_angles = angle_map[obs_mask]
    
    # -------------------------------------------------------------
    # 【关键修改】自我遮挡剔除 (Ego Self-Masking)
    # -------------------------------------------------------------
    # 设定一个半径（比如 3.0米 / voxel_size = 像素距离），在这个范围内的障碍物
    # 不参与视线阻挡计算 (视作透明)
    
    # 3.0米对应的像素数
    ignore_radius_pixel = 3.0 / voxel_size 
    
    # 筛选：只有距离大于 3米的障碍物，才算作"阻挡视线的物体"
    valid_blocker_mask = obs_dists > ignore_radius_pixel
    
    # 只取出有效阻挡物的距离和角度
    valid_obs_dists = obs_dists[valid_blocker_mask]
    valid_obs_angles = obs_angles[valid_blocker_mask]
    # -------------------------------------------------------------
    
    # 3. 极坐标分桶 (3600 bins = 0.1度精度)
    num_bins = 3600
    bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    
    # 找到每个障碍物点属于哪个角度桶
    obs_bin_idxs = np.digitize(obs_angles, bin_edges) - 1
    obs_bin_idxs = np.clip(obs_bin_idxs, 0, num_bins - 1)
    
    # 记录每个角度桶的最小障碍物距离
    min_obs_dist_per_bin = np.full(num_bins, np.inf)
    
    # 简单的循环来填充最小值 (相比复杂的向量化，这种循环在障碍物点不多时很快)
    # 这里也可以用 np.minimum.at(min_obs_dist_per_bin, obs_bin_idxs, obs_dists)
    np.minimum.at(min_obs_dist_per_bin, obs_bin_idxs, obs_dists)
            
    # 4. 生成 Vis Map
    # 找到每个像素对应的角度桶
    pixel_bin_idxs = np.digitize(angle_map.ravel(), bin_edges) - 1
    pixel_bin_idxs = np.clip(pixel_bin_idxs, 0, num_bins - 1)
    
    # 获取阻挡距离矩阵
    block_dists = min_obs_dist_per_bin[pixel_bin_idxs].reshape(H, W)
    
    vis_map = np.zeros((H, W), dtype=np.uint8) # 默认 0 (Blind)
    
    # 逻辑判断
    # 距离 < 阻挡距离 -> 可见空地
    # 容差设为 1.0 个像素距离
    visible_free_mask = (dist_map < (block_dists - 1.0))
    vis_map[visible_free_mask] = 1
    
    # 距离 ≈ 阻挡距离 且 本身是障碍物 -> 可见障碍物表面
    visible_obs_mask = obs_mask & (dist_map <= (block_dists + 1.0))
    vis_map[visible_obs_mask] = 2
    
    return vis_map

# ==========================================
# 3. 可视化主函数
# ==========================================

def visualize_and_save(comm_map, lidar_points, gt_boxes, gt_mask, ego_pose, lidar_range, save_dir, frame_id, threshold):
    plt.figure(figsize=(30, 8)) # 宽画布
    x_min, y_min, x_max, y_max = lidar_range[0], lidar_range[1], lidar_range[3], lidar_range[4]
    
    ego_x, ego_y, ego_yaw = ego_pose
    in_fov = (x_min < ego_x < x_max) and (y_min < ego_y < y_max)
    fov_status = "In FOV" if in_fov else "Out FOV"

    # --- 图 1: 路侧置信图 ---
    plt.subplot(1, 3, 1)
    if comm_map is not None:
        plt.imshow(comm_map, cmap='jet', origin='lower', vmin=0, vmax=1, 
                   extent=[x_min, x_max, y_min, y_max])
    plt.scatter(ego_x, ego_y, s=200, c='cyan', marker='*', edgecolors='white', label='Ego')
    plt.title(f"1. Roadside Confidence Map\nEgo: ({ego_x:.1f}, {ego_y:.1f})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(loc='upper right')

    # --- 图 2: 路侧原始点云 + GT + Ego ---
    ax2 = plt.subplot(1, 3, 2)
    plt.gca().set_facecolor('black')
    
    # 画点云
    if lidar_points is not None:
        mask = (lidar_points[:, 0] > x_min) & (lidar_points[:, 0] < x_max) & \
               (lidar_points[:, 1] > y_min) & (lidar_points[:, 1] < y_max)
        valid_points = lidar_points[mask]
        plt.scatter(valid_points[:, 0], valid_points[:, 1], s=0.5, c='gray', alpha=0.5)
    
    # 画 GT 框 (绿色)
    if gt_boxes is not None:
        if isinstance(gt_boxes, torch.Tensor): gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor): gt_mask = gt_mask.cpu().numpy()
        for k in range(gt_boxes.shape[0]):
            if gt_mask[k] == 0: continue
            # 假设格式: [x, y, z, w, l, h, yaw] (根据之前讨论调整索引)
            # 这里假设 w=idx4, l=idx5, yaw=idx6
            draw_box(ax2, gt_boxes[k, 0], gt_boxes[k, 1], gt_boxes[k, 4], gt_boxes[k, 5], gt_boxes[k, 6], '#00FF00')

    # 画 Ego 框 (青色)
    draw_box(ax2, ego_x, ego_y, 1.8, 4.8, ego_yaw, 'cyan', linewidth=2.0, label='Ego')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(f"2. Raw PointCloud & GT\nEgo Status: {fov_status}")
    plt.legend(loc='upper right')

    # --- 图 3: 车辆盲区估算 ---
    plt.subplot(1, 3, 3)
    
    if lidar_points is not None:
        # 计算
        voxel_size = 0.4 # 分辨率，越小越精细但越慢
        grid, W, H = points_to_occupancy_grid(lidar_points, [x_min, x_max], [y_min, y_max], voxel_size)
        vis_map = compute_precise_visibility(grid, (ego_x, ego_y), [x_min, x_max], [y_min, y_max], voxel_size)
        
        # 绘图颜色映射
        display_img = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 0: 盲区 (深灰)
        display_img[vis_map == 0] = [40, 40, 40] 
        # 1: 可见空地 (白色)
        display_img[vis_map == 1] = [255, 255, 255]
        # 2: 可见障碍物表面 (亮红)
        display_img[vis_map == 2] = [255, 0, 0]
        # 补充: 被遮挡的障碍物内部 (暗红) -> grid是障碍物 但 vis_map是盲区
        occluded_obs_mask = (grid > 0) & (vis_map == 0)
        display_img[occluded_obs_mask] = [100, 0, 0]

        plt.imshow(display_img, origin='lower', extent=[x_min, x_max, y_min, y_max])
        plt.scatter(ego_x, ego_y, s=200, c='cyan', marker='*', edgecolors='white', label='Ego View')
        plt.title("3. Estimated Ego Blind Spot\n(White=Visible, Red=Obs Surface, Gray=Blind)")
    else:
        plt.title("No Point Cloud Data")
        
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("X (m)")

    # 保存
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"vis_{frame_id}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.close() # 如果你想看图可以注释掉这行，但批量跑建议保留

# ==========================================
# 4. 主程序
# ==========================================

def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)
    
    if opt.comm_thre is not None:
        hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre
    current_thre = hypes['model']['args']['fusion_args']['communication']['thre']
    lidar_range = hypes['postprocess']['gt_range']

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

    save_dir = os.path.join(opt.model_dir, 'vis_blind_spot')
    print(f'Results will be saved to: {save_dir}')

    print('Start Processing...')
    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if batch_data is None: continue
            
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            model_output = model(batch_data['ego'])
            
            # --- 1. 获取置信图 (路侧视角) ---
            conf_map = None
            if 'psm_single_i' in model_output:
                # psm_single_i: (B, A, H, W)
                infra_conf = torch.sigmoid(model_output['psm_single_i'])[0]
                # Max over anchors -> (H, W)
                conf_map = torch.max(infra_conf, dim=0)[0].cpu().numpy()
            
            # --- 2. 获取路侧点云 & 转换坐标 ---
            raw_lidar = None
            ego_pose_infra_frame = (0, 0, 0)

            if 'origin_lidar_i' in batch_data['ego']:
                # 原始点云 (在 Ego 坐标系)
                points_ego = batch_data['ego']['origin_lidar_i'][0] # Tensor GPU
                
                # 位姿 (在 World 坐标系)
                poses = batch_data['ego']['lidar_pose']
                pose_ego = poses[0].cpu().numpy()   # 转 CPU Numpy
                pose_infra = poses[1].cpu().numpy() # 转 CPU Numpy
                
                # 计算 Ego -> Infra 的变换矩阵
                T_ego_to_infra_np = x1_to_x2(pose_ego, pose_infra)
                
                # 变换点云: Ego Frame -> Infra Frame
                T_tensor = torch.from_numpy(T_ego_to_infra_np).to(points_ego.device).float()
                points_infra = project_points_by_matrix_torch(points_ego[:, :3], T_tensor)
                raw_lidar = points_infra.cpu().numpy() # 转回 CPU Numpy 用于绘图
                
                # 计算 Ego 自身在 Infra 坐标系下的位置
                # Ego Center (0,0,0) -> Transform
                ego_center_local = np.array([0, 0, 0, 1]).reshape(4, 1)
                ego_center_infra = np.dot(T_ego_to_infra_np, ego_center_local).flatten()
                
                # Ego Orientation (Yaw)
                # Transform forward vector (1,0,0) to get direction
                ego_front_local = np.array([1, 0, 0, 1]).reshape(4, 1)
                ego_front_infra = np.dot(T_ego_to_infra_np, ego_front_local).flatten()
                
                dx = ego_front_infra[0] - ego_center_infra[0]
                dy = ego_front_infra[1] - ego_center_infra[1]
                ego_yaw_infra = np.arctan2(dy, dx)
                
                ego_pose_infra_frame = (ego_center_infra[0], ego_center_infra[1], ego_yaw_infra)
            
            # --- 3. 获取 GT (路侧视角) ---
            # DataLoader 已经提供了转换好的 single_i GT
            gt_boxes = batch_data['ego'].get('object_bbx_center_single_i', [None])[0]
            gt_mask = batch_data['ego'].get('object_bbx_mask_single_i', [None])[0]

            # --- 4. 绘图 ---
            frame_id = f"{i:04d}_infra"
            visualize_and_save(conf_map, raw_lidar, gt_boxes, gt_mask, 
                               ego_pose_infra_frame, lidar_range, save_dir, frame_id, current_thre)
            
            # 清理图形内存，防止显存/内存溢出
            plt.close('all')

    print("Visualization Finished!")

if __name__ == '__main__':
    main()