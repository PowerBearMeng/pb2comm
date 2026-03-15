import json
import os
import math
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
import shutil  

# ================= 配置区域 =================
IS_ROTATION_IN_DEGREES = True 
# ===========================================
def reset_dir(dir_path):
    """如果目录存在则清空重建，否则直接创建"""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def get_dynamic_transform_params(infra_pose, veh_pose):
    """
    计算动态变换参数：
    1. 旋转：根据 Infra 的 Pitch/Roll 进行回正。
    2. 高度：根据 (Infra Z - Vehicle Z) 计算相对高度差。
    """
    # --- 1. 旋转处理 (只看 Infra) ---
    curr_pitch = infra_pose.get('pitch', 0.0)
    curr_roll  = infra_pose.get('roll', 0.0)
    
    # 构造逆向旋转矩阵，把歪的转正
    # 这一步是为了让路侧雷达"变平"，平行于 xy 平面
    r = R.from_euler('yx', [-curr_pitch, -curr_roll], degrees=IS_ROTATION_IN_DEGREES)
    rot_matrix = r.as_matrix()
    
    infra_z = infra_pose.get('z', 0.0)
    veh_z = veh_pose.get('z', 0.0)
    
    z_offset = infra_z - veh_z
    
    print(f"    [Height Align] Infra Z: {infra_z:.2f} | Veh Z: {veh_z:.2f} | Delta Offset: {z_offset:.2f}")
    
    return rot_matrix, z_offset

def process_single_frame(label_path, output_label_dir, points_dir, output_points_dir, vehicle_label_dir):
    filename = os.path.basename(label_path)
    file_id = os.path.splitext(filename)[0]
    
    # 1. 读取 Infra Label
    with open(label_path, 'r', encoding='utf-8') as f:
        infra_data = json.load(f)
    
    if 'sensor_pose' not in infra_data:
        print(f"Skipping {filename}: No sensor_pose")
        return

    # 2. 读取对应的 Vehicle Label (为了获取车的高度)
    veh_json_path = os.path.join(vehicle_label_dir, filename)
    if not os.path.exists(veh_json_path):
        print(f"Warning: Corresponding vehicle label not found: {veh_json_path}")
        # 如果找不到对应的车，只能由 fallback 方案 (比如默认高度差 4.0米，或者跳过)
        # 这里为了演示，假设必须找到
        return
        
    with open(veh_json_path, 'r', encoding='utf-8') as f:
        veh_data = json.load(f)
    
    infra_pose = infra_data['sensor_pose']
    veh_pose = veh_data['sensor_pose'] # 假设车端也有 sensor_pose 字段
    
    # --- 计算变换矩阵 (核心修改) ---
    rot_matrix, z_offset = get_dynamic_transform_params(infra_pose, veh_pose)

    # --- 处理点云 ---
    bin_filename = file_id + ".bin"
    src_bin_path = os.path.join(points_dir, bin_filename)
    dst_bin_path = os.path.join(output_points_dir, bin_filename)
    
    if os.path.exists(src_bin_path):
        points = np.fromfile(src_bin_path, dtype=np.float32).reshape(-1, 4)
        
        # 1. 旋转 (把路侧雷达转正)
        # points[:, :3] 是 (N, 3)，rot_matrix 是 (3, 3)
        # 公式: P_new = R * P_old^T -> (P_old * R^T)
        points[:, :3] = points[:, :3] @ rot_matrix.T
        
        # 2. 平移 (高度对齐到车端雷达高度)
        points[:, 2] += z_offset
        
        points.astype(np.float32).tofile(dst_bin_path)
    else:
        print(f"Warning: Point cloud not found for {filename}")

    # --- 处理 Label ---
    new_data = copy.deepcopy(infra_data)
    new_objects_list = []
    
    if 'objects' in new_data:
        for old_obj in new_data['objects']:
            new_obj = {} 
            new_obj['type'] = old_obj.get('class', 'Car') 
            
            # --- 坐标变换 ---
            # 假设 location 是相对于 Infra Sensor 的坐标 (如果是绝对坐标，处理方式不同)
            # 根据你之前的代码逻辑，这里似乎是在处理相对坐标
            raw_loc = old_obj.get('location', {})
            loc_vec = np.array([raw_loc.get('x', 0), raw_loc.get('y', 0), raw_loc.get('z', 0)])
            
            # 1. 旋转位置
            new_loc_vec = rot_matrix @ loc_vec
            # 2. 平移位置 (Z轴补偿)
            new_loc_vec[2] += z_offset 
            
            new_obj['3d_location'] = {
                'x': float(new_loc_vec[0]),
                'y': float(new_loc_vec[1]),
                'z': float(new_loc_vec[2])
            }
            
            # 尺寸
            raw_dim = old_obj.get('dimensions', {})
            new_obj['3d_dimensions'] = {
                'h': float(raw_dim.get('h', raw_dim.get('height', 0))),
                'w': float(raw_dim.get('w', raw_dim.get('width', 0))),
                'l': float(raw_dim.get('l', raw_dim.get('length', 0)))
            }
            
            # --- 旋转角度修正 (Yaw) ---
            raw_rot = old_obj.get('rotation', {})
            if isinstance(raw_rot, dict):
                old_yaw = math.radians(raw_rot.get('yaw', 0.0))
            else:
                old_yaw = float(raw_rot)

            # 向量法旋转 Yaw
            heading_vec = np.array([math.cos(old_yaw), math.sin(old_yaw), 0.0])
            new_heading_vec = rot_matrix @ heading_vec
            new_yaw = math.atan2(new_heading_vec[1], new_heading_vec[0])
            
            new_obj['rotation'] = float(new_yaw)
            
            # 其他字段
            if 'id' in old_obj: new_obj['id'] = old_obj['id']
            new_obj['occluded'] = old_obj.get('occluded', 0)
            new_obj['truncated'] = old_obj.get('truncated', 0)
            new_obj['alpha'] = old_obj.get('alpha', 0)
            new_objects_list.append(new_obj)

    new_data['objects'] = new_objects_list
      
    new_sensor_pose = new_data['sensor_pose']
    new_sensor_pose['pitch'] = 0.0 
    new_sensor_pose['roll'] = 0.0
    new_sensor_pose['z'] = veh_pose['z'] 

    dst_json_path = os.path.join(output_label_dir, filename)
    with open(dst_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)

# def main():
#     ROOT_DIR = "/home/yty/mfh/carla_data/Town12_t_0"
#     DEST_DIR = os.path.join(ROOT_DIR, "cooperative-vehicle-infrastructure")

#     # ================= 1. 初始化目标目录 (只执行一次) =================
#     # 路侧数据一般放在 infrastructure-side 文件夹下
#     # Label 放在 label/virtuallidar (表示转换过虚拟雷达坐标系)
#     # Point 放在 velodyne
#     dest_infra_label_dir = os.path.join(DEST_DIR, "infrastructure-side", "label", "virtuallidar")
#     dest_infra_point_dir = os.path.join(DEST_DIR, "infrastructure-side", "velodyne")

#     print(f">>> 初始化目标目录: {dest_infra_label_dir}")
#     reset_dir(dest_infra_label_dir)
#     reset_dir(dest_infra_point_dir)
#     # ===============================================================

#     # 安全地获取 sequence 列表 (过滤掉非 seq 文件夹，比如 dest_dir 本身)
#     seq_list = sorted([d for d in os.listdir(ROOT_DIR) if d.startswith("seq")])
    
#     for seq in seq_list:
#         if seq.endswith(".txt"):
#             continue
#         if not seq.startswith("seq"):
#             continue
#         print(f"Processing sequence: {seq}")
        
#         infra_root = os.path.join(ROOT_DIR, seq, "roadside0") 
#         veh_root = os.path.join(ROOT_DIR, seq, "vehicle")    
        
#         labels_dir = os.path.join(infra_root, "labels")       
#         points_dir = os.path.join(infra_root, "points")     
#         veh_labels_dir = os.path.join(veh_root, "labels")
        
#         # 临时生成目录
#         new_labels_dir = os.path.join(infra_root, "new_labels_virt") 
#         new_points_dir = os.path.join(infra_root, "new_points_virt")
        
#         # 每次循环可以重新创建临时的 new 目录，或者直接用，最后转移完可以不管
#         reset_dir(new_labels_dir)
#         reset_dir(new_points_dir)

#         # 1. 执行你的处理逻辑 (生成新的 Json 和 Bin 到 new_xxx 目录)
#         if os.path.exists(labels_dir):
#             files = [f for f in os.listdir(labels_dir) if f.endswith('.json')]
#             print(f"  - Generating Virtual Lidar Data ({len(files)} frames)...")
#             for f in files:
#                 process_single_frame(
#                     os.path.join(labels_dir, f), 
#                     new_labels_dir, 
#                     points_dir, 
#                     new_points_dir,
#                     veh_labels_dir
#                 )
#         else:
#             print(f"  [Warning] No labels found in {labels_dir}")
#             continue

#         # ================= 2. 转移文件到 DEST_DIR =================
#         print(f"  - Transferring files to {DEST_DIR}...")
        
#         # 转移 Label
#         if os.path.exists(new_labels_dir):
#             files = [f for f in os.listdir(new_labels_dir) if f.endswith('.json')]
#             for fname in files:
#                 src = os.path.join(new_labels_dir, fname)
#                 new_name = f"{seq}_{fname}"
#                 dst = os.path.join(dest_infra_label_dir, new_name)
#                 shutil.copy2(src, dst)
        
#         # 转移 Point Cloud
#         if os.path.exists(new_points_dir):
#             files = [f for f in os.listdir(new_points_dir) if f.endswith('.bin')]
#             for fname in files:
#                 src = os.path.join(new_points_dir, fname)
#                 new_name = f"{seq}_{fname}"
#                 dst = os.path.join(dest_infra_point_dir, new_name)
#                 shutil.copy2(src, dst)
#         # ==========================================================

#     print("\nAll Done! Infrastructure data is ready.")

def main():
    # ================= 1. 定义基础路径和目标 =================
    BASE_DIR = "/home/yty/mfh/carla_data"
    town_list = ["Town12_t_0", "Town12_t_1"] 
    
    # 统一的全局输出目录，放在 BASE_DIR 下
    DEST_DIR = os.path.join(BASE_DIR, "cooperative-vehicle-infrastructure")

    # ================= 2. 初始化目标目录 (全局只执行一次！) =================
    # 路侧数据一般放在 infrastructure-side 文件夹下
    # Label 放在 label/virtuallidar (表示转换过虚拟雷达坐标系)
    # Point 放在 velodyne
    dest_infra_label_dir = os.path.join(DEST_DIR, "infrastructure-side", "label", "virtuallidar")
    dest_infra_point_dir = os.path.join(DEST_DIR, "infrastructure-side", "velodyne")

    print(f">>> 初始化全局目标目录: {dest_infra_label_dir}")
    reset_dir(dest_infra_label_dir)
    reset_dir(dest_infra_point_dir)
    # ===============================================================

    # ================= 3. 开始外层遍历不同的 Town =================
    for town in town_list:
        ROOT_DIR = os.path.join(BASE_DIR, town)
        
        if not os.path.exists(ROOT_DIR):
            print(f"跳过: 找不到目录 {ROOT_DIR}")
            continue
            
        print(f"\n================ 开始处理: {town} ================")
        
        # 安全地获取 sequence 列表 (过滤掉非 seq 文件夹)
        seq_list = sorted([d for d in os.listdir(ROOT_DIR) if d.startswith("seq") and os.path.isdir(os.path.join(ROOT_DIR, d))])
        
        for seq in seq_list:
            print(f"Processing sequence: {seq}")
            
            infra_root = os.path.join(ROOT_DIR, seq, "roadside0") 
            veh_root = os.path.join(ROOT_DIR, seq, "vehicle")    
            
            labels_dir = os.path.join(infra_root, "labels")       
            points_dir = os.path.join(infra_root, "points")     
            veh_labels_dir = os.path.join(veh_root, "labels")
            
            # 临时生成目录
            new_labels_dir = os.path.join(infra_root, "new_labels_virt") 
            new_points_dir = os.path.join(infra_root, "new_points_virt")
            
            # 每次循环可以重新创建临时的 new 目录
            reset_dir(new_labels_dir)
            reset_dir(new_points_dir)

            # 1. 执行你的处理逻辑 (生成新的 Json 和 Bin 到 new_xxx 目录)
            if os.path.exists(labels_dir):
                files = [f for f in os.listdir(labels_dir) if f.endswith('.json')]
                print(f"  - Generating Virtual Lidar Data ({len(files)} frames)...")
                for f in files:
                    process_single_frame(
                        os.path.join(labels_dir, f), 
                        new_labels_dir, 
                        points_dir, 
                        new_points_dir,
                        veh_labels_dir
                    )
            else:
                print(f"  [Warning] No labels found in {labels_dir}")
                continue

            # ================= 4. 转移文件到全局 DEST_DIR =================
            print(f"  - Transferring files to {DEST_DIR}...")
            
            # 转移 Label
            if os.path.exists(new_labels_dir):
                files = [f for f in os.listdir(new_labels_dir) if f.endswith('.json')]
                for fname in files:
                    src = os.path.join(new_labels_dir, fname)
                    # 【核心修改】文件名加上 town 前缀防止冲突
                    new_name = f"{town}_{seq}_{fname}"
                    dst = os.path.join(dest_infra_label_dir, new_name)
                    shutil.copy2(src, dst)
            
            # 转移 Point Cloud
            if os.path.exists(new_points_dir):
                files = [f for f in os.listdir(new_points_dir) if f.endswith('.bin')]
                for fname in files:
                    src = os.path.join(new_points_dir, fname)
                    # 【核心修改】文件名加上 town 前缀防止冲突
                    new_name = f"{town}_{seq}_{fname}"
                    dst = os.path.join(dest_infra_point_dir, new_name)
                    shutil.copy2(src, dst)
            # ==========================================================

    print("\nAll Done! Infrastructure data is ready.")


if __name__ == "__main__":
    main()