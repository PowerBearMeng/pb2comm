import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random

# 引入必要的工具
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets.intermediate_fusion_motion import IntermediateFusionMotion

def check_dataset(hypes_file):
    print(f"Loading Config: {hypes_file}")
    params = load_yaml(hypes_file)
    
    max_num = params['postprocess']['max_num']
    print(f"Config max_num: {max_num}")

    dataset = IntermediateFusionMotion(params, visualize=True, train=True)
    
    # Batch Size 可以设为 2 或 4，看你想看几个场景
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, 
                             collate_fn=dataset.collate_batch_train)

    print("开始读取 Batch...")
    
    save_dir = 'vis_scene_debug'
    os.makedirs(save_dir, exist_ok=True)

    for i, batch_data in enumerate(data_loader):
        ego_data = batch_data['ego']
        
        if 'object_traj' not in ego_data:
            print("[ERROR] 'object_traj' 缺失！")
            return

        traj = ego_data['object_traj']      # (B, Max_Num, 5, 2)
        mask = ego_data['object_traj_mask'] # (B, Max_Num, 5)
        ids = ego_data['object_ids']        # List/Array
        
        batch_size = traj.shape[0]
        print(f"\n--- Batch {i} 读取成功，包含 {batch_size} 个场景 ---")
        
        # --- 修改点 1：遍历 Batch 中的每一个场景 (Sample) ---
        for b_id in range(batch_size):
            # 统计一下这个场景里有多少条有效轨迹
            valid_count = torch.sum(mask[b_id, :, 0]).item()
            print(f"  -> 场景 {b_id}: 包含 {int(valid_count)} 条有效轨迹")
            
            if valid_count > 0:
                save_name = os.path.join(save_dir, f"batch{i}_scene{b_id}_all_trajs.png")
                
                # 调用新的画图函数：画出整个场景
                visualize_whole_scene(ego_data, b_id, ids, save_name)
        
        print(f"\n可视化完成！请查看 {save_dir} 文件夹。")
        break # 只跑一个 Batch

def get_real_id(ids, b_id, o_id):
    """鲁棒的 ID 获取函数，防止各种 list/int/numpy 报错"""
    try:
        # 情况 A: ids[b_id] 是列表/数组
        batch_container = ids[b_id]
        if hasattr(batch_container, '__len__') and not isinstance(batch_container, str):
            if o_id < len(batch_container):
                return str(batch_container[o_id])
        
        # 情况 B: ids 已经被 Flatten 或者 BatchSize=1
        elif isinstance(batch_container, (int, np.integer)):
            if b_id == 0 and o_id < len(ids):
                return str(ids[o_id])
            else:
                return f"Flat_{o_id}"
                
        # 情况 C: ID 本身就是 int (例如 ids 是 numpy array)
        if isinstance(ids, np.ndarray):
             return str(ids[b_id, o_id])

    except Exception:
        return "Err"
    return "Unknown"

def visualize_whole_scene(ego_data, batch_idx, all_ids, save_path):
    """
    修改点 2：在一张图上画出该帧所有的轨迹
    """
    plt.figure(figsize=(12, 12))
    
    # 1. 获取数据
    cur_boxes = ego_data['object_bbx_center'][batch_idx].numpy() # (Max_Num, 7)
    cur_box_mask = ego_data['object_bbx_mask'][batch_idx].numpy()
    
    future_traj = ego_data['object_traj'][batch_idx].numpy() # (Max_Num, 5, 2)
    traj_mask = ego_data['object_traj_mask'][batch_idx].numpy()
    
    max_num = cur_boxes.shape[0]

    # 2. 画自车 (黑色五角星)
    plt.plot(0, 0, 'k*', markersize=20, label='Ego Vehicle', zorder=10)
    
    # 3. 遍历所有物体，把有的全画出来
    plotted_count = 0
    for o_id in range(max_num):
        # 只有当 mask 有效时才画
        # 这里有两个检查：1. 物体本身存在 (box_mask) 2. 轨迹存在 (traj_mask)
        if cur_box_mask[o_id] == 1:
            
            # A. 画当前位置 (蓝色点)
            start_x = cur_boxes[o_id, 0]
            start_y = cur_boxes[o_id, 1]
            plt.plot(start_x, start_y, 'bo', markersize=6, alpha=0.5)
            
            # B. 获取真实 ID 并标注
            real_id = get_real_id(all_ids, batch_idx, o_id)
            plt.text(start_x, start_y, real_id, fontsize=9, color='blue', ha='right')

            # C. 画未来轨迹 (如果存在)
            if traj_mask[o_id, 0] == 1:
                valid_steps = traj_mask[o_id] == 1
                points = future_traj[o_id][valid_steps]
                
                if len(points) > 0:
                    # 连线：当前 -> 未来
                    full_x = np.concatenate(([start_x], points[:, 0]))
                    full_y = np.concatenate(([start_y], points[:, 1]))
                    
                    # 画红线
                    plt.plot(full_x, full_y, 'r.-', linewidth=1.5, markersize=4, alpha=0.8)
                    
                    # 标终点箭头
                    plt.arrow(full_x[-2], full_y[-2], full_x[-1]-full_x[-2], full_y[-1]-full_y[-2], 
                              head_width=0.5, head_length=0.5, fc='r', ec='r', alpha=0.8)
                    plotted_count += 1

    plt.title(f"Scene Visualization (Batch {batch_idx})\nTotal Trajectories: {plotted_count}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 添加一个假的图例
    plt.plot([], [], 'bo', label='Current Pos')
    plt.plot([], [], 'r.-', label='Future Traj')
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Image Saved: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 记得替换你的 YAML 路径
    parser.add_argument('--hypes_yaml', type=str, default='opencood/hypes_yaml/carla/carla_where2comm_max_multiscale_resnet.yaml')
    args = parser.parse_args()
    
    check_dataset(args.hypes_yaml)