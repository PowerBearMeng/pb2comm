import os
import json
import numpy as np
from tqdm import tqdm

# ================= 配置区 =================
DATA_DIR = "/home/yty/mfh/carla_data/cooperative-vehicle-infrastructure"
# 过滤阈值：一个框内至少包含几个雷达点才保留它？(通常建议 3~5 个点)
MIN_POINTS_THRESHOLD = 10 
# ==========================================

def load_bin(bin_path):
    """加载 .bin 点云，返回 (N, 3) 坐标"""
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]

def eul2rot(r, p, y):
    """欧拉角转旋转矩阵"""
    R_x = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    R_y = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    R_z = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return R_z @ R_y @ R_x

def get_lidar_to_world_matrix(pose):
    """组装 4x4 变换矩阵"""
    x, y, z, roll, yaw, pitch = pose
    roll, pitch, yaw = np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)
    T = np.eye(4)
    T[:3, :3] = eul2rot(roll, pitch, yaw)
    T[:3, 3] = [x, y, z]
    return T

def project_points(points, matrix):
    """点云投影"""
    points_hom = np.ones((points.shape[0], 4))
    points_hom[:, :3] = points
    return np.dot(points_hom, matrix.T)[:, :3]

def count_points_in_box_2d(points, cx, cy, l, w, yaw):
    """
    利用矩阵旋转快速判断点云中有多少个点落在一个 2D 框内
    :param points: 点云集合 (N, 3)
    :param cx, cy: 框中心 X, Y
    :param l, w: 框的长, 宽
    :param yaw: 偏航角 (弧度)
    :return: 落在框内的点云数量
    """
    if len(points) == 0:
        return 0
        
    # 平移点云到框的中心
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    
    # 将点云反向旋转 yaw 角，对齐到框的局部坐标系
    c, s = np.cos(yaw), np.sin(yaw)
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    
    # 在局部坐标系下判断是否在长宽范围内 (这里稍微给一点0.1m的容差)
    mask = (np.abs(local_x) <= (l / 2.0 + 0.1)) & (np.abs(local_y) <= (w / 2.0 + 0.1))
    
    return np.sum(mask)

def main():
    world_label_dir = os.path.join(DATA_DIR, "cooperative/label_world")
    json_files = [f for f in os.listdir(world_label_dir) if f.endswith('.json')]
    
    print(f"开始清洗数据集... 共找到 {len(json_files)} 个标签文件。")
    print(f"过滤规则：融合点云(Ego+Infra)数量 < {MIN_POINTS_THRESHOLD} 的幽灵框将被永久删除。")
    
    removed_total = 0
    kept_total = 0

    for filename in tqdm(json_files):
        frame_id = filename.replace('.json', '')
        
        # 1. 拼凑所有需要的文件路径
        veh_bin = os.path.join(DATA_DIR, f"vehicle-side/velodyne/{frame_id}.bin")
        veh_pose_file = os.path.join(DATA_DIR, f"vehicle-side/label/lidar/{frame_id}.json")
        inf_bin = os.path.join(DATA_DIR, f"infrastructure-side/velodyne/{frame_id}.bin")
        inf_pose_file = os.path.join(DATA_DIR, f"infrastructure-side/label/virtuallidar/{frame_id}.json")
        world_label_path = os.path.join(world_label_dir, filename)

        if not all([os.path.exists(veh_bin), os.path.exists(inf_bin), os.path.exists(world_label_path)]):
            continue

        # 2. 读取车端点云、路端点云及位姿
        points_v = load_bin(veh_bin)
        with open(veh_pose_file, 'r') as f:
            v_pose_data = json.load(f)['sensor_pose']
            v_pose = [v_pose_data['x'], v_pose_data['y'], v_pose_data['z'], v_pose_data['roll'], v_pose_data['yaw'], v_pose_data['pitch']]
        T_world_ego = get_lidar_to_world_matrix(v_pose)

        points_i_raw = load_bin(inf_bin)
        with open(inf_pose_file, 'r') as f:
            i_pose_data = json.load(f)['sensor_pose']
            i_pose = [i_pose_data['x'], i_pose_data['y'], i_pose_data['z'], i_pose_data['roll'], i_pose_data['yaw'], i_pose_data['pitch']]
        T_world_infra = get_lidar_to_world_matrix(i_pose)
        
        # 3. 把路端点云投影过来，拼成一团完整的融合点云
        T_ego_infra = np.dot(np.linalg.inv(T_world_ego), T_world_infra)
        points_i_proj = project_points(points_i_raw, T_ego_infra)
        
        # 过滤掉地面的点，防止误判
        mask_v = points_v[:, 2] > -2.0
        mask_i = points_i_proj[:, 2] > -2.0
        fused_points = np.vstack([points_v[mask_v], points_i_proj[mask_i]])

        # 4. 读取 JSON 并逐个过滤框
        with open(world_label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        
        objects = label_data.get('objects', [])
        valid_objects = []

        for obj in objects:
            loc = obj['3d_location']
            dim = obj['3d_dimensions']
            yaw = obj['rotation']
            
            # 计算落在该框内的点云数量
            pts_count = count_points_in_box_2d(
                fused_points, 
                loc['x'], loc['y'], 
                dim['l'], dim['w'], 
                yaw
            )
            
            if pts_count >= MIN_POINTS_THRESHOLD:
                valid_objects.append(obj)
                kept_total += 1
            else:
                removed_total += 1

        # 5. 用只包含有效框的数据覆盖原 JSON 文件
        label_data['objects'] = valid_objects
        with open(world_label_path, 'w', encoding='utf-8') as f:
            json.dump(label_data, f, indent=2)

    print("\n清洗完成！")
    print(f"✅ 保留了 {kept_total} 个有效框。")
    print(f"🗑️ 删除了 {removed_total} 个无点云的幽灵框。")
    print("现在你可以重新运行之前的可视化脚本，看看世界是不是清静了！")

if __name__ == "__main__":
    main()