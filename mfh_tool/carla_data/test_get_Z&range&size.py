import os
import json
import math
import numpy as np

def get_corners_3d(loc, dim, yaw):
    """ 计算 3D 框的 8 个顶点 """
    x, y, z = loc['x'], loc['y'], loc['z']
    h, w, l = dim['h'], dim['w'], dim['l']
    
    dx = l / 2
    dy = w / 2
    dz = h / 2
    
    # 局部顶点 (8, 3)
    x_corners = [dx, dx, -dx, -dx, dx, dx, -dx, -dx]
    y_corners = [dy, -dy, -dy, dy, dy, -dy, -dy, dy]
    z_corners = [dz, dz, dz, dz, -dz, -dz, -dz, -dz]
    
    corners = np.vstack([x_corners, y_corners, z_corners])
    
    # 旋转 (Yaw)
    c = math.cos(yaw)
    s = math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    rotated_corners = np.dot(R, corners)
    
    # 平移
    rotated_corners[0, :] += x
    rotated_corners[1, :] += y
    rotated_corners[2, :] += z
    
    return rotated_corners.T

def round_to_4(val):
    """ 向上取整为 4 的倍数 (适配 OpenCOOD Voxel 对齐) """
    return math.ceil(val / 4.0) * 4

def analyze_directory(label_dir, title, class_name='Car'):
    print(f"\n{'='*20} 正在分析: {title} {'='*20}")
    print(f"目录: {label_dir}")
    
    if not os.path.exists(label_dir):
        print("Error: 目录不存在！")
        return

    # 数据容器
    all_x, all_y, all_z = [], [], [] # 存储所有角点坐标
    all_h, all_w, all_l = [], [], [] # 存储所有物体尺寸
    count = 0

    files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    
    for file in files:
        with open(os.path.join(label_dir, file), 'r') as f:
            try:
                data = json.load(f)
            except:
                continue
        
        objects = data.get('objects', [])
        # 有些数据集可能把 objects 放在 key 比如 'vehicles' 里，这里假设是标准的 objects
        if not objects and 'vehicles' in data:
            objects = data['vehicles']

        for obj in objects:
            # 类别过滤
            obj_type = obj.get('type', obj.get('class'))
            if obj_type != class_name:
                continue

            # 读取位置和尺寸
            loc = obj.get('3d_location')
            dim = obj.get('3d_dimensions')
            rot = obj.get('rotation')

            # 兼容性处理
            if not loc: loc = obj.get('location')
            if not dim: dim = obj.get('dimensions')
            
            # 确保数据完整
            if loc and dim and rot is not None:
                # 角度处理 (确保是弧度)
                if isinstance(rot, dict):
                    yaw = math.radians(rot.get('yaw', 0.0))
                else:
                    yaw = float(rot) # 假设已经是弧度

                # 尺寸处理 (h, w, l)
                h = dim.get('h', dim.get('height'))
                w = dim.get('w', dim.get('width'))
                l = dim.get('l', dim.get('length'))

                all_h.append(h)
                all_w.append(w)
                all_l.append(l)

                # 范围处理 (计算角点)
                # 构造标准字典传给 get_corners_3d
                loc_dict = {'x': loc.get('x', loc[0] if isinstance(loc, list) else 0),
                            'y': loc.get('y', loc[1] if isinstance(loc, list) else 0),
                            'z': loc.get('z', loc[2] if isinstance(loc, list) else 0)}
                dim_dict = {'h': h, 'w': w, 'l': l}
                
                corners = get_corners_3d(loc_dict, dim_dict, yaw)
                all_x.extend(corners[:, 0])
                all_y.extend(corners[:, 1])
                all_z.extend(corners[:, 2])
                count += 1

    if count == 0:
        print("未找到有效目标。")
        return

    # --- 统计结果输出 ---
    X, Y, Z = np.array(all_x), np.array(all_y), np.array(all_z)
    
    print(f"\n[1] 尺寸统计 (基于 {count} 个 {class_name}):")
    mean_h, mean_w, mean_l = np.mean(all_h), np.mean(all_w), np.mean(all_l)
    print(f"  > 平均尺寸 (l, w, h): [{mean_l:.2f}, {mean_w:.2f}, {mean_h:.2f}]")
    print(f"  > 推荐 Anchor Size: l={mean_l:.2f}, w={mean_w:.2f}, h={mean_h:.2f}")

    print(f"\n[2] Z轴分布 (高度):")
    z_p01 = np.percentile(Z, 0.1)  # 0.1% 分位 (地面附近)
    z_p99 = np.percentile(Z, 99.9) # 99.9% 分位 (最高点)
    print(f"  > 覆盖范围 (99.9%): [{z_p01:.2f}, {z_p99:.2f}]")
    rec_z_min = math.floor(z_p01) - 1
    rec_z_max = math.ceil(z_p99) + 1
    print(f"  > 推荐 Z 范围: [{rec_z_min}, {rec_z_max}]")

    print(f"\n[3] Lidar Range (水平覆盖):")
    abs_x, abs_y = np.abs(X), np.abs(Y)
    x_999 = np.percentile(abs_x, 99.9)
    y_999 = np.percentile(abs_y, 99.9)
    
    rec_x = round_to_4(x_999)
    rec_y = round_to_4(y_999)
    # 限制最小范围
    rec_x = max(rec_x, 48) 
    rec_y = max(rec_y, 48)

    print(f"  > X轴 99.9% 覆盖: ±{x_999:.2f} m -> 推荐: ±{rec_x} m")
    print(f"  > Y轴 99.9% 覆盖: ±{y_999:.2f} m -> 推荐: ±{rec_y} m")
    
    print(f"\n✅ 最终推荐配置 ({title}):")
    print(f"cav_lidar_range: [{-rec_x}, {-rec_y}, {rec_z_min}, {rec_x}, {rec_y}, {rec_z_max}]")
    print(f"anchor_args -> l={mean_l:.2f}, w={mean_w:.2f}, h={mean_h:.2f}")

if __name__ == "__main__":
    # 指向你刚刚生成好的数据集根目录
    DATASET_ROOT = "/home/yty/mfh/record/cooperative-vehicle-infrastructure"
    
    # 定义车端和路侧的路径
    # 根据标准 OpenCOOD 结构和你的转移脚本逻辑
    veh_label_path = os.path.join(DATASET_ROOT, "vehicle-side/label/lidar")
    infra_label_path = os.path.join(DATASET_ROOT, "infrastructure-side/label/virtuallidar")
    
    analyze_directory(veh_label_path, "Vehicle Side (车端)", class_name='Car')
    
    analyze_directory(infra_label_path, "Infrastructure Side (路侧)", class_name='Car')