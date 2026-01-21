# -*- coding: utf-8 -*-
# Author: Your Name
# Purpose: Dedicated script for evaluating Motion Forecasting (ADE/FDE)

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

# ================= 1. 评测指标计算函数 =================
def compute_ade_fde_numpy(pred_traj, gt_traj, gt_mask):
    """
    计算 ADE 和 FDE
    Args:
        pred_traj: (B, N, T, 2)
        gt_traj:   (B, N, T, 2)
        gt_mask:   (B, N, T)
    """
    # 展平 Batch 和 N 维度，变成 (Total_Objects, T, 2)
    B, N, T, C = pred_traj.shape
    pred_traj = pred_traj.reshape(-1, T, C)
    gt_traj = gt_traj.reshape(-1, T, C)
    gt_mask = gt_mask.reshape(-1, T)

    diff = pred_traj - gt_traj
    dist = np.linalg.norm(diff, axis=-1) # (Total_Objects, T)

    # ADE: 只算 mask=1 的帧
    valid_dist = dist[gt_mask == 1]
    if len(valid_dist) == 0:
        return None, None
    ade = np.mean(valid_dist)

    # FDE: 取最后一帧 (简化版，假设所有 mask 都是连续的且最后时刻对齐)
    # 严谨做法是找每个物体最后一个 mask=1 的位置
    final_dist = dist[:, -1]
    final_mask = gt_mask[:, -1]
    
    valid_final_dist = final_dist[final_mask == 1]
    if len(valid_final_dist) == 0:
        fde = 0.0
    else:
        fde = np.mean(valid_final_dist)

    return ade, fde

# ================= 2. 可视化函数 =================
def visualize_motion(batch_data, output_dict, batch_idx, save_dir):
    """
    画出红线(预测)和绿线(真值)
    """
    # 取 Batch 中的第 0 个场景
    b_id = 0 
    
    cur_pos = batch_data['ego']['object_bbx_center'][b_id].cpu().numpy() # (N, 7)
    gt_traj = batch_data['ego']['object_traj'][b_id].cpu().numpy()       # (N, 5, 2)
    pred_traj = output_dict['traj_preds'][b_id].cpu().numpy()            # (N, 5, 2)
    mask = batch_data['ego']['object_traj_mask'][b_id].cpu().numpy()     # (N, 5)

    plt.figure(figsize=(10, 10))
    plt.plot(0, 0, 'k*', markersize=15, label='Ego') # 自车中心

    has_obj = False
    for n in range(len(cur_pos)):
        # 如果这个物体有轨迹真值 (Mask有效)
        if mask[n, 0] == 1:
            has_obj = True
            cx, cy = cur_pos[n, 0], cur_pos[n, 1]
            
            # 画当前位置
            plt.plot(cx, cy, 'bo', markersize=6, alpha=0.5)
            
            # 画真值 (绿线)
            gt_pts = gt_traj[n]
            gx = np.concatenate(([cx], gt_pts[:, 0]))
            gy = np.concatenate(([cy], gt_pts[:, 1]))
            plt.plot(gx, gy, 'g-', linewidth=2, alpha=0.7, label='GT' if not plt.gca().get_legend_handles_labels()[1].count('GT') else "")
            
            # 画预测 (红虚线)
            pred_pts = pred_traj[n]
            px = np.concatenate(([cx], pred_pts[:, 0]))
            py = np.concatenate(([cy], pred_pts[:, 1]))
            plt.plot(px, py, 'r--', linewidth=2, alpha=0.9, label='Pred' if not plt.gca().get_legend_handles_labels()[1].count('Pred') else "")

    if has_obj:
        plt.legend()
        plt.title(f"Batch {batch_idx} Motion Prediction\nGreen=GT, Red=Pred")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        save_path = os.path.join(save_dir, f"motion_vis_{batch_idx:04d}.png")
        plt.savefig(save_path)
        plt.close()

# ================= 3. 主流程 =================
def main():
    parser = argparse.ArgumentParser(description="Motion Inference")
    parser.add_argument('--model_dir', type=str, required=True, help='Training output path')
    parser.add_argument('--fusion_method', type=str, default='intermediate')
    parser.add_argument('--vis', action='store_true', help='Visualize trajectory')
    opt = parser.parse_args()

    # 1. 加载配置
    hypes = yaml_utils.load_yaml(None, opt)
    print(f"Dataset: {hypes['test_dir']}")
    
    # 2. 构建 Dataset 和 DataLoader
    # 注意：这里我们不需要 visualize=True，因为那是给检测可视化用的
    opencood_dataset = build_dataset(hypes, visualize=False, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1, # 推理通常 batch_size=1
                             num_workers=4,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    # 3. 加载模型
    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    # 4. 开始评测
    all_ade = []
    all_fde = []
    
    # 创建可视化目录
    if opt.vis:
        vis_dir = os.path.join(opt.model_dir, 'vis_motion')
        os.makedirs(vis_dir, exist_ok=True)

    print("Start Motion Inference...")
    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            
            # 【核心差异】
            # 我们不调用 inference_utils，而是直接 model(ego)
            # 这样我们能拿到完整的 output_dict，包含 traj_preds
            output_dict = model(batch_data['ego'])
            
            # 检查是否有轨迹输出
            if 'traj_preds' in output_dict and 'object_traj' in batch_data['ego']:
                
                # 获取预测和真值
                pred_traj = output_dict['traj_preds'].cpu().numpy() # (B, N, T, 2)
                gt_traj = batch_data['ego']['object_traj'].cpu().numpy()
                gt_mask = batch_data['ego']['object_traj_mask'].cpu().numpy()
                
                # 计算指标
                ade, fde = compute_ade_fde_numpy(pred_traj, gt_traj, gt_mask)
                
                if ade is not None:
                    all_ade.append(ade)
                    all_fde.append(fde)
                
                # 可视化 (每 10 帧画一次，防止太慢)
                if opt.vis and i % 10 == 0:
                    visualize_motion(batch_data, output_dict, i, vis_dir)

    # 5. 打印最终结果
    if len(all_ade) > 0:
        print("\n" + "="*40)
        print(f"Motion Evaluation Results:")
        print(f"Samples: {len(all_ade)}")
        print(f"Mean ADE: {np.mean(all_ade):.4f} meters")
        print(f"Mean FDE: {np.mean(all_fde):.4f} meters")
        print("="*40 + "\n")
        
        # 保存到 txt
        with open(os.path.join(opt.model_dir, 'motion_result.txt'), 'w') as f:
            f.write(f"Mean ADE: {np.mean(all_ade):.4f} m\n")
            f.write(f"Mean FDE: {np.mean(all_fde):.4f} m\n")
    else:
        print("No valid trajectory samples found!")

if __name__ == '__main__':
    main()