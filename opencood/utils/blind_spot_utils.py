import numpy as np
import cv2

def points_to_occupancy_grid(points, x_range, y_range, voxel_size, ground_filter_thre=-1.5):
    # 你的原始代码逻辑
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
    # 你的原始代码逻辑 (确保可以直接复制粘贴你上面提供的代码)
    # ... (此处省略，直接使用你提供的函数内容) ...
    # 为节省篇幅，这里假设你已经把代码复制进来了
    # 注意：确保这一步返回的是 0=Blind, 1=Free, 2=VisibleObs
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

def get_blind_spot_mask(lidar_points, ego_pose, lidar_range, target_feat_shape, voxel_size=0.4):
    """
    生成用于 Mask 特征图的盲区掩码
    Returns: mask (H, W), values in [0.0, 1.0]
    """
    x_min, y_min, x_max, y_max = lidar_range[0], lidar_range[1], lidar_range[3], lidar_range[4]
    
    # 1. 生成高分辨率可视性图
    grid, W, H = points_to_occupancy_grid(lidar_points, [x_min, x_max], [y_min, y_max], voxel_size)
    vis_map = compute_precise_visibility(grid, (0, 0), [x_min, x_max], [y_min, y_max], voxel_size)
    # 注意：这里 ego_pose 传入 (0,0) 是因为通常 origin_lidar 已经在 ego 坐标系下了

    # 2. 调整到特征图尺寸
    target_h, target_w = target_feat_shape
    vis_map_resized = cv2.resize(vis_map, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    # 3. 构造传输 Mask: 0=Blind(发送), 1/2=Visible(不发送)
    # 你的逻辑：Vis_map==0 是盲区 -> Mask=1 (保留特征)
    # Vis_map!=0 是可见 -> Mask=0 (丢弃特征)
    transmission_mask = (vis_map_resized == 0).astype(np.uint8)
    
    # 4. 膨胀 (可选，建议保留以处理边缘对齐误差)
    dilation_kernel = np.ones((3, 3), np.uint8)
    expanded_mask = cv2.dilate(transmission_mask, dilation_kernel, iterations=1)
    
    return expanded_mask.astype(np.float32)