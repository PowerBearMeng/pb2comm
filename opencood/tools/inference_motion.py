# -*- coding: utf-8 -*-
# Author: OpenCOOD
# Purpose: Dedicated script for evaluating Motion Forecasting (ADE/FDE) with Static Filter

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
        pred_traj: (Total_N, T, 2) - 这里的输入已经是展平后的
        gt_traj:   (Total_N, T, 2)
        gt_mask:   (Total_N, T)
    """
    diff = pred_traj - gt_traj
    dist = np.linalg.norm(diff, axis=-1) # (Total_Objects, T)

    # ADE: 只算 mask=1 的帧
    valid_dist = dist[gt_mask == 1]
    if len(valid_dist) == 0:
        return None, None
    ade = np.mean(valid_dist)

    # FDE: 取最后一帧
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
    修复版：只画 mask 为 1 的轨迹点，防止连线连到 (0,0)
    """
    b_id = 0 
    
    cur_pos = batch_data['ego']['object_bbx_center'][b_id].cpu().numpy()
    gt_traj = batch_data['ego']['object_traj'][b_id].cpu().numpy()       
    pred_traj = output_dict['traj_preds'][b_id].cpu().numpy()            
    
    # 拿到完整的 mask (N, T)
    mask = batch_data['ego']['object_traj_mask'][b_id].cpu().numpy()     

    if np.sum(mask) == 0:
        return

    plt.figure(figsize=(10, 10))
    plt.plot(0, 0, 'k*', markersize=15, label='Ego', zorder=10)

    has_drawn = False
    for n in range(len(cur_pos)):
        # 只要有一帧有效，就尝试画
        if np.sum(mask[n]) > 0:
            cx, cy = cur_pos[n, 0], cur_pos[n, 1]
            plt.plot(cx, cy, 'bo', markersize=6, alpha=0.5)
            
            # ================= [修改核心] =================
            # 1. 筛选有效的 GT 点
            # 找到该物体所有有效的时间步
            valid_steps = mask[n] == 1
            
            if np.sum(valid_steps) > 0:
                has_drawn = True
                
                # 只取有效点
                valid_gt = gt_traj[n][valid_steps]
                
                # 连线：当前点 -> 有效GT点1 -> 有效GT点2 ...
                gx = np.concatenate(([cx], valid_gt[:, 0]))
                gy = np.concatenate(([cy], valid_gt[:, 1]))
                
                label_gt = 'GT' if 'GT' not in plt.gca().get_legend_handles_labels()[1] else ""
                plt.plot(gx, gy, 'g-', linewidth=2, alpha=0.7, label=label_gt)
                
                # 2. 画预测 (通常预测是满的，但对应着 GT 画比较好，或者画全部)
                # 这里我们画全部预测，或者只画对应 GT 有效的部分
                # 为了看清模型的完整预测，建议画全部预测 (前提是模型没预测归零)
                pred_pts = pred_traj[n] # 预测通常不需要 mask，因为模型会预测出合理值
                px = np.concatenate(([cx], pred_pts[:, 0]))
                py = np.concatenate(([cy], pred_pts[:, 1]))
                
                label_pred = 'Pred' if 'Pred' not in plt.gca().get_legend_handles_labels()[1] else ""
                plt.plot(px, py, 'r--', linewidth=2, alpha=0.9, label=label_pred)
            # ============================================

    if has_drawn:
        plt.legend(loc='upper right')
        plt.title(f"Batch {batch_idx} Motion Prediction\nGreen=GT, Red=Pred")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        save_path = os.path.join(save_dir, f"vis_{batch_idx:04d}.png")
        plt.savefig(save_path)
        print(f"Saved visualization: {save_path}")
    
    plt.close()
# ================= 3. 主流程 =================
def main():
    parser = argparse.ArgumentParser(description="Motion Inference")
    parser.add_argument('--model_dir', type=str, required=True, help='Training output path')
    parser.add_argument('--fusion_method', type=str, default='intermediate')
    parser.add_argument('--vis', action='store_true', help='Visualize trajectory')
    parser.add_argument('--static_thre', type=float, default=0.1, help='Threshold to filter static objects (meters)')
    opt = parser.parse_args()

    # 1. 加载配置
    hypes = yaml_utils.load_yaml(None, opt)
    print(f"Dataset: {hypes['test_dir']}")
    print(f"Static Threshold: {opt.static_thre} meters")
    
    # 2. 构建 Dataset 和 DataLoader
    opencood_dataset = build_dataset(hypes, visualize=False, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1, 
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
    
    # 加载 Checkpoint
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    # 4. 准备保存路径
    vis_dir = os.path.join(opt.model_dir, 'vis_motion_evaluation')
    if opt.vis:
        os.makedirs(vis_dir, exist_ok=True)

    # 5. 开始评测循环
    all_ade = []
    all_fde = []
    
    print("Start Motion Inference...")
    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            
            # 模型前向传播
            output_dict = model(batch_data['ego'])
            
            # 只有当模型输出包含轨迹，且数据集中有真值时，才进行计算
            if 'traj_preds' in output_dict and 'object_traj' in batch_data['ego']:
                
                # 获取原始数据 (B, N, T, 2)
                pred_traj = output_dict['traj_preds'].cpu().numpy()
                gt_traj = batch_data['ego']['object_traj'].cpu().numpy()
                gt_mask = batch_data['ego']['object_traj_mask'].cpu().numpy()
                
                # =============== 核心：静止物体过滤 ===============
                # 1. 计算 GT 的总位移 (最后一帧 - 第一帧)
                # cur_pos: (B, N, 2)
                cur_pos = batch_data['ego']['object_bbx_center'][..., :2].cpu().numpy()
                last_gt_pos = gt_traj[:, :, -1, :] # (B, N, 2)
                
                total_displacement = np.linalg.norm(last_gt_pos - cur_pos, axis=-1) # (B, N)
                
                # 2. 判断是否运动
                is_moving = total_displacement > opt.static_thre
                
                # 3. 更新 mask (将静止物体的时间步 mask 全部置 0)
                # is_moving[:, :, None] -> (B, N, 1) 广播到 (B, N, T)
                gt_mask = gt_mask * is_moving[:, :, np.newaxis]
                # ================================================

                # 展平数据用于计算指标
                B, N, T, C = pred_traj.shape
                flat_preds = pred_traj.reshape(-1, T, C)
                flat_gt = gt_traj.reshape(-1, T, C)
                flat_mask = gt_mask.reshape(-1, T)
                
                # 计算 ADE/FDE
                ade, fde = compute_ade_fde_numpy(flat_preds, flat_gt, flat_mask)
                
                if ade is not None:
                    all_ade.append(ade)
                    all_fde.append(fde)
                
                # 可视化 (每 20 帧保存一张，必须加 --vis 参数)
                if opt.vis and i % 20 == 0:
                    # 注意：这里传入的是更新过 mask (过滤了静止物体) 的数据
                    # 这样可视化出来的图片里，静止物体就不会画红绿线了，非常清爽
                    batch_data['ego']['object_traj_mask'] = torch.from_numpy(gt_mask).to(device)
                    visualize_motion(batch_data, output_dict, i, vis_dir)

    # 6. 打印最终结果
    if len(all_ade) > 0:
        mean_ade = np.mean(all_ade)
        mean_fde = np.mean(all_fde)
        
        print("\n" + "="*40)
        print(f"Motion Evaluation Results (Static Threshold={opt.static_thre}m):")
        print(f"Total Valid Batches: {len(all_ade)}")
        print(f"Mean ADE: {mean_ade:.4f} meters")
        print(f"Mean FDE: {mean_fde:.4f} meters")
        print("="*40 + "\n")
        
        # 保存结果到 txt
        res_file = os.path.join(opt.model_dir, 'motion_result.txt')
        with open(res_file, 'a+') as f:
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Static Threshold: {opt.static_thre}m\n")
            f.write(f"Mean ADE: {mean_ade:.4f} m\n")
            f.write(f"Mean FDE: {mean_fde:.4f} m\n")
            f.write("-" * 20 + "\n")
        print(f"Results saved to {res_file}")
    else:
        print("No valid trajectory samples found! (Maybe threshold is too high?)")

if __name__ == '__main__':
    main()