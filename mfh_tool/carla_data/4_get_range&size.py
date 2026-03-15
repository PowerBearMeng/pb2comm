import os
import json
import numpy as np
import math

# ================= 辅助函数 =================
def get_corners_3d(loc, dim, yaw):
    """
    根据中心、尺寸、偏航角计算 8 个顶点的世界坐标 (通用)
    """
    x, y, z = loc['x'], loc['y'], loc['z']
    
    # 兼容不同的 dim 定义 (l,w,h)
    l = dim.get('l', dim.get('length'))
    w = dim.get('w', dim.get('width'))
    h = dim.get('h', dim.get('height'))
    
    dx = l / 2
    dy = w / 2
    dz = h / 2
    
    # 局部顶点 (8, 3)
    x_corners = [dx, dx, -dx, -dx, dx, dx, -dx, -dx]
    y_corners = [dy, -dy, -dy, dy, dy, -dy, -dy, dy]
    z_corners = [dz, dz, dz, dz, -dz, -dz, -dz, -dz]
    
    corners = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    
    # 旋转 (Yaw)
    c = math.cos(yaw)
    s = math.sin(yaw)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    rotated_corners = np.dot(R, corners)
    
    # 平移
    rotated_corners[0, :] += x
    rotated_corners[1, :] += y
    rotated_corners[2, :] += z
    
    return rotated_corners.T  # (8, 3)

def scan_dataset_stats(base_dir, town_list, target_subdirs, class_names=['Car']):
    """
    遍历多个 Town 及其下所有 seq，收集所有框的角点信息
    """
    # 超全局容器
    all_x, all_y, all_z = [], [], []
    all_l, all_w, all_h = [], [], []

    total_objects = 0
    
    print(f"目标地图列表: {town_list}")
    print(f"目标子目录: {target_subdirs}")
    print(f"目标类别: {class_names}")
    print("-" * 50)

    # --- 1. 最外层遍历 Town ---
    for town in town_list:
        root_dir = os.path.join(base_dir, town)
        
        if not os.path.exists(root_dir):
            print(f"  [跳过] 找不到地图目录: {root_dir}")
            continue
            
        seq_list = sorted([d for d in os.listdir(root_dir) if d.startswith("seq") and os.path.isdir(os.path.join(root_dir, d))])
        print(f"在 {town} 中找到 {len(seq_list)} 个序列")

        # --- 2. 遍历当前 Town 的各个 seq ---
        for seq in seq_list:
            for subdir_name in target_subdirs:
                # 拼凑路径: base/town/seqXX/vehicle/new_labels
                label_dir = os.path.join(root_dir, seq, subdir_name)
                
                if not os.path.exists(label_dir):
                    continue
                    
                files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
                for file in files:
                    try:
                        with open(os.path.join(label_dir, file), 'r') as f:
                            data = json.load(f)
                    except Exception:
                        continue
                        
                    for obj in data.get('objects', []):
                        # 类别过滤
                        obj_type = obj.get('type', obj.get('class'))
                        if obj_type in class_names:
                            loc = obj.get('3d_location')
                            dim = obj.get('3d_dimensions')
                            if not loc: loc = obj.get('location')
                            if not dim: dim = obj.get('dimensions')
                            
                            # 角度处理
                            rot = obj.get('rotation', 0.0)
                            if isinstance(rot, dict):
                                rot = math.radians(rot.get('yaw', 0.0))
                            
                            if loc and dim:
                                corners = get_corners_3d(loc, dim, float(rot))
                                all_l.append(float(dim['l']))
                                all_w.append(float(dim['w']))
                                all_h.append(float(dim['h']))
                                all_x.extend(corners[:, 0])
                                all_y.extend(corners[:, 1])
                                all_z.extend(corners[:, 2])
                                total_objects += 1

    return np.array(all_x), np.array(all_y), np.array(all_z), np.array(all_l), np.array(all_w), np.array(all_h), total_objects

def print_recommendation(X, Y, Z, count):
    if count == 0:
        print("未找到有效数据，请检查路径。")
        return

    print("\n" + "="*60)
    print(f"📊 全局数据集统计报告 (基于 {count} 个物体, {len(X)} 个顶点)")
    print("="*60)
    
    # --- Z轴 (高度) ---
    z_min, z_max = np.min(Z), np.max(Z)
    z_p01 = np.percentile(Z, 0.1)
    z_p999 = np.percentile(Z, 99.9)
    
    print(f"【Z轴 (高度)】")
    print(f"  - 实际极值: [{z_min:.2f}, {z_max:.2f}]")
    print(f"  - 99.9% 核心区: [{z_p01:.2f}, {z_p999:.2f}]")
    
    # 推荐: 向下取整到整数位 - 0.5 (留余量)
    rec_z_min = math.floor(z_p01) - 1
    rec_z_max = math.ceil(z_p999) + 1
    
    # --- XY轴 (水平范围) ---
    abs_x = np.abs(X)
    abs_y = np.abs(Y)
    
    x_max_abs = np.max(abs_x)
    x_p95 = np.percentile(abs_x, 99.0)
    x_p999 = np.percentile(abs_x, 99.9) 
    
    y_max_abs = np.max(abs_y)
    y_p95 = np.percentile(abs_y, 99.0)
    y_p999 = np.percentile(abs_y, 99.9)

    print("-" * 30)
    print(f"【X轴 (前后)】 Max: {x_max_abs:.2f} m | 95%: {x_p95:.2f} m | 99.9%: {x_p999:.2f} m")
    print(f"【Y轴 (左右)】 Max: {y_max_abs:.2f} m | 95%: {y_p95:.2f} m | 99.9%: {y_p999:.2f} m")
    
    
    def round_to_4(val):
        return math.ceil(val / 4.0) * 4
    
    rec_x = round_to_4(x_p999)
    rec_y = round_to_4(y_p999)
    
    # 防止太小 (最少给 40米)
    rec_x = max(rec_x, 40)
    rec_y = max(rec_y, 40)

    print("="*60)
    print(f"✅ 推荐 YAML 配置 (cav_lidar_range):")
    print(f"   说明: 基于 99.9% 覆盖率，并确保适配 Backbone 对齐要求 (8倍数)")
    print("-" * 30)
    print(f"cav_lidar_range: [{-rec_x}, {-rec_y}, {rec_z_min}, {rec_x}, {rec_y}, {rec_z_max}]")
    print("-" * 30)
    print(f"   (X覆盖: ±{rec_x}m, Y覆盖: ±{rec_y}m)")
    print(f"   (Z覆盖: {rec_z_min}m 到 {rec_z_max}m)")

def print_dim_stats(name, arr):
    print(f"【{name}】")
    print(f"  - min / max : {arr.min():.2f} / {arr.max():.2f}")
    print(f"  - mean      : {arr.mean():.2f}")
    print(f"  - 95%       : {np.percentile(arr, 95):.2f}")
    print(f"  - 99%       : {np.percentile(arr, 99):.2f}")

if __name__ == "__main__":
    # ================= 配置区域 =================
    BASE_DIR = "/home/yty/mfh/carla_data"
    TOWN_LIST = ["Town12_t_0", "Town12_t_1"] 
    
    TARGET_SUBDIRS = [
        "vehicle/new_labels", 
        # "roadside0/new_labels_virt"
    ]
    
    CLASS_NAMES = ['Car'] # 统计所有车
    # ===========================================
    
    X, Y, Z, L, W, H, count = scan_dataset_stats(BASE_DIR, TOWN_LIST, TARGET_SUBDIRS, CLASS_NAMES)
    
    if count > 0:
        print_recommendation(X, Y, Z, count)
        print("\n📦 3D Box 尺度统计 (全局单位: 米)")
        print("-" * 50)
        print_dim_stats("Length (l)", L)
        print_dim_stats("Width  (w)", W)
        print_dim_stats("Height (h)", H)
    else:
        print("\n⚠️ 警告: 没有扫描到任何有效数据，请检查 BASE_DIR 和 TOWN_LIST 是否正确。")