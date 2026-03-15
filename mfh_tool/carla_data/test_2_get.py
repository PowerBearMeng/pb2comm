import json
import os
import math
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
import shutil  
import sys

# ================= 配置区域 =================
IS_ROTATION_IN_DEGREES = True 
# ===========================================

def reset_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def calculate_seq_calibration_mean(file_list, infra_label_dir, veh_label_dir):
    """
    【修改版】计算序列校准参数：
    1. 旋转矩阵：依然使用第一帧的参数（假设路侧单元安装后不晃动）。
    2. 高度偏移：遍历整个序列，计算 (Infra_Z - Veh_Z) 的【平均值】。
    3. 基准车高：计算整个序列的 Vehicle_Z 的【平均值】，作为最终 Sensor Pose 的 Z。
    """
    total_z_offset = 0.0
    total_veh_z = 0.0
    count = 0
    
    first_rot_matrix = None
    
    print(f"  [Calibration] Scanning {len(file_list)} frames to calculate MEAN offset...")

    for i, filename in enumerate(file_list):
        infra_path = os.path.join(infra_label_dir, filename)
        veh_path = os.path.join(veh_label_dir, filename)
        
        if not os.path.exists(infra_path) or not os.path.exists(veh_path):
            continue
            
        try:
            with open(infra_path, 'r', encoding='utf-8') as f:
                infra_data = json.load(f)
            with open(veh_path, 'r', encoding='utf-8') as f:
                veh_data = json.load(f)
            
            infra_pose = infra_data.get('sensor_pose', {})
            veh_pose = veh_data.get('sensor_pose', {})
            
            i_z = infra_pose.get('z', 0.0)
            v_z = veh_pose.get('z', 0.0)
            
            # 累加差异 (Infra - Veh)
            total_z_offset += (i_z - v_z)
            total_veh_z += v_z
            count += 1
            
            # 只在第一帧计算旋转矩阵 (路侧是固定的，不需要平均旋转，否则容易引入计算误差)
            if first_rot_matrix is None:
                curr_pitch = infra_pose.get('pitch', 0.0)
                curr_roll  = infra_pose.get('roll', 0.0)
                r = R.from_euler('yx', [-curr_pitch, -curr_roll], degrees=IS_ROTATION_IN_DEGREES)
                first_rot_matrix = r.as_matrix()
                print(f"    - First Frame Rotation Locked: Pitch={curr_pitch}, Roll={curr_roll}")

        except Exception as e:
            print(f"    - Warning: Error reading {filename}: {e}")
            continue
    
    if count == 0:
        print("    [Error] No valid pairs found in sequence!")
        return None, None, None

    # 计算均值
    mean_z_offset = total_z_offset / count
    mean_veh_z = total_veh_z / count
    
    print(f"  [Calibration Result] Valid Frames: {count}")
    print(f"    - Mean Height Offset (Infra - Veh): {mean_z_offset:.4f} m")
    print(f"    - Mean Vehicle Height: {mean_veh_z:.4f} m")
    
    return first_rot_matrix, mean_z_offset, mean_veh_z

def process_single_frame(label_path, output_label_dir, points_dir, output_points_dir, 
                         fixed_rot_matrix, fixed_z_offset, fixed_target_z):
    filename = os.path.basename(label_path)
    file_id = os.path.splitext(filename)[0]
    
    with open(label_path, 'r', encoding='utf-8') as f:
        infra_data = json.load(f)
    
    if 'sensor_pose' not in infra_data:
        return

    # --- 处理点云 ---
    bin_filename = file_id + ".bin"
    src_bin_path = os.path.join(points_dir, bin_filename)
    dst_bin_path = os.path.join(output_points_dir, bin_filename)
    
    if os.path.exists(src_bin_path):
        points = np.fromfile(src_bin_path, dtype=np.float32).reshape(-1, 4)
        
        # 1. 旋转：应用固定的"回正"矩阵
        points[:, :3] = points[:, :3] @ fixed_rot_matrix.T
        
        # 2. 平移：应用固定的 Mean Offset
        # 逻辑：Infra_Z(高) - Veh_Z(低) = 正数 offset
        # 地面点(负值) + offset = 地面点变高(接近0)
        points[:, 2] += fixed_z_offset
        
        points.astype(np.float32).tofile(dst_bin_path)

    # --- 处理 Label ---
    new_data = copy.deepcopy(infra_data)
    new_objects_list = []
    
    if 'objects' in new_data:
        for old_obj in new_data['objects']:
            new_obj = {} 
            new_obj['type'] = old_obj.get('class', 'Car') 
            
            raw_loc = old_obj.get('location', {})
            loc_vec = np.array([raw_loc.get('x', 0), raw_loc.get('y', 0), raw_loc.get('z', 0)])
            
            # 坐标变换
            new_loc_vec = fixed_rot_matrix @ loc_vec
            new_loc_vec[2] += fixed_z_offset 
            
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
            
            # 角度变换
            raw_rot = old_obj.get('rotation', {})
            if isinstance(raw_rot, dict):
                old_yaw = math.radians(raw_rot.get('yaw', 0.0))
            else:
                old_yaw = float(raw_rot)

            heading_vec = np.array([math.cos(old_yaw), math.sin(old_yaw), 0.0])
            new_heading_vec = fixed_rot_matrix @ heading_vec
            new_yaw = math.atan2(new_heading_vec[1], new_heading_vec[0])
            
            new_obj['rotation'] = float(new_yaw)
            
            if 'id' in old_obj: new_obj['id'] = old_obj['id']
            new_obj['occluded'] = old_obj.get('occluded', 0)
            new_obj['truncated'] = old_obj.get('truncated', 0)
            new_obj['alpha'] = old_obj.get('alpha', 0)
            new_objects_list.append(new_obj)

    new_data['objects'] = new_objects_list
      
    # 更新 Sensor Pose
    new_sensor_pose = new_data['sensor_pose']
    new_sensor_pose['pitch'] = 0.0 
    new_sensor_pose['roll'] = 0.0
    # 【关键】 Z 设为序列的平均车高 (Mean Vehicle Height)
    # 这样整个序列的虚拟雷达高度是完全恒定的
    new_sensor_pose['z'] = fixed_target_z 

    dst_json_path = os.path.join(output_label_dir, filename)
    with open(dst_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)

def main():
    ROOT_DIR = "/home/yty/mfh/carla_data/town12_t_6"
    DEST_DIR = os.path.join(ROOT_DIR, "cooperative-vehicle-infrastructure")

    dest_infra_label_dir = os.path.join(DEST_DIR, "infrastructure-side", "label", "virtuallidar")
    dest_infra_point_dir = os.path.join(DEST_DIR, "infrastructure-side", "velodyne")

    print(f">>> 初始化: {dest_infra_label_dir}")
    reset_dir(dest_infra_label_dir)
    reset_dir(dest_infra_point_dir)

    seq_list = sorted([d for d in os.listdir(ROOT_DIR) if d.startswith("seq")])
    
    for seq in seq_list:
        if seq.endswith(".txt"):
            continue
        print(f"\nProcessing sequence: {seq}")
        
        infra_root = os.path.join(ROOT_DIR, seq, "roadside0") 
        veh_root = os.path.join(ROOT_DIR, seq, "vehicle")    
        
        labels_dir = os.path.join(infra_root, "labels")       
        points_dir = os.path.join(infra_root, "points")     
        veh_labels_dir = os.path.join(veh_root, "labels")
        
        new_labels_dir = os.path.join(infra_root, "new_labels_virt") 
        new_points_dir = os.path.join(infra_root, "new_points_virt")
        
        reset_dir(new_labels_dir)
        reset_dir(new_points_dir)

        if os.path.exists(labels_dir):
            files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.json')])
            if len(files) == 0: continue

            # === 1. 计算序列的 MEAN 参数 ===
            # 传入所有文件列表，计算均值
            seq_rot_matrix, seq_mean_offset, seq_mean_veh_z = calculate_seq_calibration_mean(
                files, labels_dir, veh_labels_dir
            )
            
            if seq_rot_matrix is None: continue
            
            # === 2. 用固定的均值参数处理所有帧 ===
            print(f"  - Transforming {len(files)} frames...")
            for f in files:
                process_single_frame(
                    os.path.join(labels_dir, f), 
                    new_labels_dir, 
                    points_dir, 
                    new_points_dir,
                    seq_rot_matrix, 
                    seq_mean_offset, # 使用均值 offset
                    seq_mean_veh_z   # 使用均值车高作为目标 pose.z
                )
        
        # 转移文件...
        if os.path.exists(new_labels_dir):
            files = [f for f in os.listdir(new_labels_dir) if f.endswith('.json')]
            for fname in files:
                shutil.copy2(os.path.join(new_labels_dir, fname), os.path.join(dest_infra_label_dir, f"{seq}_{fname}"))
        
        if os.path.exists(new_points_dir):
            files = [f for f in os.listdir(new_points_dir) if f.endswith('.bin')]
            for fname in files:
                shutil.copy2(os.path.join(new_points_dir, fname), os.path.join(dest_infra_point_dir, f"{seq}_{fname}"))
                
        # 清理临时目录
        # if os.path.exists(new_labels_dir): shutil.rmtree(new_labels_dir)
        # if os.path.exists(new_points_dir): shutil.rmtree(new_points_dir)

    print("\nAll Done.")

if __name__ == "__main__":
    main()