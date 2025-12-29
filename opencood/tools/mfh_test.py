import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# 导入必要的库
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset

def draw_box(ax, x, y, h, w, l, yaw, color):
    """
    画框函数
    h, w, l: 对应 dimensions[0], [1], [2]
    通常 l 是车长(x轴方向), w 是车宽(y轴方向)
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # 车辆坐标系下四个角点 (假设中心在原点)
    # x向前(l/2), y向左(w/2)
    dx = l / 2
    dy = w / 2
    
    corners = np.array([
        [ dx,  dy],  # Front-Left
        [-dx,  dy],  # Rear-Left
        [-dx, -dy],  # Rear-Right
        [ dx, -dy]   # Front-Right
    ])
    
    # 旋转
    rot_mat = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    corners_global = corners @ rot_mat.T
    
    # 平移
    corners_global[:, 0] += x
    corners_global[:, 1] += y
    
    # 闭合用于绘图
    corners_plot = np.vstack([corners_global, corners_global[0]])
    ax.plot(corners_plot[:, 0], corners_plot[:, 1], c=color, linewidth=1.5)
    
    # 画车头箭头
    ax.arrow(x, y, cos_yaw*2, sin_yaw*2, head_width=0.5, fc=color, ec=color)

def check_dataset_loader_direct():
    parser = argparse.ArgumentParser(description="Check Dataset Alignment (Direct Mode)")
    path = 'opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet.yaml'
    path = 'opencood/hypes_yaml/v2x-seq/dair_where2comm_max_multiscale_resnet.yaml'
    parser.add_argument('--hypes_yaml', type=str, default=path, 
                        help='Path to your dataset config yaml file')
    opt = parser.parse_args()

    # 1. 加载配置
    hypes = yaml_utils.load_yaml(opt.hypes_yaml)
    
    # 2. 构建数据集 (visualize=False 避免 Open3D 报错)
    print("正在构建数据集...")
    dataset = build_dataset(hypes, visualize=False, train=True)
    
    save_dir = 'dataset_check_vis'
    os.makedirs(save_dir, exist_ok=True)
    print(f"可视化结果将保存到: {save_dir}/")

    print("开始直接检查 retrieve_base_data (前10帧)...")
    
    # 直接遍历索引，不经过 DataLoader，这样最纯净
    for i in range(len(dataset)):
        if i >= 100: break 
        
        try:
            # === 核心：直接获取原始数据 ===
            # data[0] 是 Ego, data[1] 是 Infra
            data = dataset.retrieve_base_data(i)
        except Exception as e:
            print(f"Frame {i} 加载失败: {e}")
            continue

        if data is None:
            print(f"Frame {i} 返回 None")
            continue

        # 检查路端数据 (data[1])
        if 1 not in data:
            print(f"Frame {i} 没有路端数据 (data[1])")
            continue
            
        infra_data = data[1]
        
        # 获取路端点云
        if 'lidar_np' not in infra_data:
            print(f"Frame {i} 路端没有 lidar_np")
            continue
        infra_pts = infra_data['lidar_np']
        
        # 获取路端 Boxes (这是您的代码投影过来的)
        if 'vehicles_single' not in infra_data['params']:
            print(f"Frame {i} 路端没有 vehicles 标签")
            infra_boxes = []
        else:
            infra_boxes = infra_data['params']['vehicles_single']

        # --- 绘图 ---
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor('black')
        plt.title(f"Frame {i} - Infrastructure View\n(Check if Green Boxes align with Cars)")

        # 1. 画点云 (下采样)
        if infra_pts is not None:
            # 过滤范围，只显示中心区域
            mask = (np.abs(infra_pts[:,0]) < 60) & (np.abs(infra_pts[:,1]) < 60)
            pts = infra_pts[mask]
            # 随机降采样到 1/5 
            step = 5
            plt.scatter(pts[::step, 0], pts[::step, 1], s=0.5, c='gray', alpha=0.5)

        # 2. 画投影过来的框
        for veh in infra_boxes:
            loc = veh['location']   # [x, y, z]
            dim = veh['dimensions'] # [h, w, l] (根据您的代码逻辑)
            angle = veh['angle']
            
            h = dim[0]
            w = dim[1]
            l = dim[2]
            
            draw_box(ax, loc[0], loc[1], h, w, l, angle, '#00FF00')

        plt.axis('equal')
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)
        
        out_path = os.path.join(save_dir, f"check_frame_{i:04d}_infra.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Frame {i}: 已保存 -> {out_path}")

    print("完成。")

if __name__ == '__main__':
    check_dataset_loader_direct()