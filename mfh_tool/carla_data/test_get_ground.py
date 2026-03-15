import os
import numpy as np
import struct
import math

# ================= 配置区域 =================
ROOT_DIR = "/home/yty/mfh/record"
# 目标文件夹名称
LIDAR_FOLDER_NAME = "new_points_virt" 
# 每个序列抽样多少帧进行计算 (取前 5 帧通常足够，因为地面由标定决定，不会变)
SAMPLE_NUM_PER_SEQ = 5 

# RANSAC 参数
RANSAC_ITERATIONS = 100
RANSAC_THRESH = 0.1  # 点到平面的距离阈值 (米)，越小越严苛

Z_FILTER_MIN = -3.0
Z_FILTER_MAX = 4.0   # 稍微降低上限，避免拟合到车顶，只关注地面
# ===========================================

def read_bin(file_path):
    """
    读取 .bin 文件 (x, y, z, intensity)
    """
    try:
        # 假设是 float32 格式
        points = np.fromfile(file_path, dtype=np.float32)
        # 常见的 bin 是 N * 4 (x, y, z, i)
        points = points.reshape(-1, 4)
        return points[:, :3] # 只取 x, y, z
    except Exception as e:
        print(f"读取 BIN 失败 {file_path}: {e}")
        return np.array([])

def fit_plane_ransac(points, iterations=100, thresh=0.1):
    """
    使用 RANSAC 拟合平面 ax + by + cz + d = 0
    返回: (a, b, c, d), inlier_count
    """
    best_plane = None
    max_inliers = 0
    n_points = points.shape[0]
    
    if n_points < 50:
        return None, 0

    # 预先生成随机索引
    # 为了速度，如果点太多，先随机降采样一下参与 RANSAC
    if n_points > 10000:
        sample_idxs = np.random.choice(n_points, 10000, replace=False)
        work_points = points[sample_idxs]
    else:
        work_points = points

    n_work = work_points.shape[0]
    rand_idxs = np.random.randint(0, n_work, size=(iterations, 3))

    for i in range(iterations):
        # 1. 随机选3个点
        idxs = rand_idxs[i]
        p1, p2, p3 = work_points[idxs]

        # 2. 计算法向量
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        
        if norm < 1e-6: continue
        
        a, b, c = normal / norm
        
        # 3. 约束：地面必须大概水平
        # 法向量 (a,b,c) 应该接近 (0,0,1) 或 (0,0,-1)
        # 即 |c| 应该接近 1。如果 |c| < 0.7 (倾角 > 45度)，认为是墙不是地
        if abs(c) < 0.8: 
            continue

        # 4. 计算 d: ax + by + cz + d = 0
        d = -np.dot([a, b, c], p1)

        # 5. 验证内点 (使用全量 points 还是 work_points 都可以，为了速度用 work_points)
        distances = np.abs(np.dot(work_points, np.array([a, b, c])) + d)
        inliers_count = np.sum(distances < thresh)

        if inliers_count > max_inliers:
            max_inliers = inliers_count
            best_plane = (a, b, c, d)

    return best_plane, max_inliers

def get_ground_z_intercept(plane_params):
    """
    计算平面与 Z 轴的交点 (即 x=0, y=0 时的 z)
    这代表了雷达正下方的地面高度。
    ax + by + cz + d = 0  =>  cz = -d  => z = -d/c
    """
    if plane_params is None: return None
    a, b, c, d = plane_params
    if abs(c) < 1e-4: return None
    return -d / c

if __name__ == "__main__":
    # 自动扫描 seq 开头的文件夹
    if not os.path.exists(ROOT_DIR):
        print(f"错误: 根目录不存在 {ROOT_DIR}")
        exit()

    seq_list = sorted([d for d in os.listdir(ROOT_DIR) if d.startswith("seq")])
    
    GLOBAL_GROUND_Z = []
    
    print(f"{'='*60}")
    print(f" 地面平面一致性检查 (Ground Z Consistency Check)")
    print(f"{'='*60}")
    print(f"目标目录: .../roadside0/{LIDAR_FOLDER_NAME}/*.bin")
    print(f"过滤范围: Z in [{Z_FILTER_MIN}, {Z_FILTER_MAX}]")
    print(f"说明: 计算出的 Ground Z 代表【坐标原点正下方的地面 Z 值】")
    print(f"      如果数据是 virt (车系)，这里显示的应该是约 -1.7m (主车高度)")
    print(f"{'-'*60}\n")

    for seq in seq_list:
        print(f"正在处理序列: {seq} ...")
        
        lidar_dir = os.path.join(ROOT_DIR, seq, "roadside0", LIDAR_FOLDER_NAME)
        if not os.path.exists(lidar_dir):
            print(f"  [跳过] 找不到目录: {lidar_dir}")
            continue
            
        files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.bin')])
        if not files:
            print(f"  [跳过] 无 .bin 文件")
            continue
            
        # 抽样
        sample_files = files[:min(len(files), SAMPLE_NUM_PER_SEQ)]
        current_seq_zs = []
        
        for bin_file in sample_files:
            file_path = os.path.join(lidar_dir, bin_file)
            points = read_bin(file_path)
            
            if len(points) == 0: continue
            
            # --- 核心步骤: 粗过滤 ---
            # 1. Z轴过滤
            mask_z = (points[:, 2] > Z_FILTER_MIN) & (points[:, 2] < Z_FILTER_MAX)
            # 2. 半径过滤 (只取近处，远处的地面可能不平或拟合误差大)
            mask_r = (points[:, 0]**2 + points[:, 1]**2) < 30**2 
            
            roi_points = points[mask_z & mask_r]
            
            if len(roi_points) < 100:
                continue
                
            # --- RANSAC ---
            plane, inliers = fit_plane_ransac(roi_points, 
                                            iterations=RANSAC_ITERATIONS, 
                                            thresh=RANSAC_THRESH)
            
            if plane:
                # 计算 Ground Z (雷达原点下方的地面高度)
                z_ground = get_ground_z_intercept(plane)
                if z_ground is not None:
                    current_seq_zs.append(z_ground)

        # 统计当前序列
        if current_seq_zs:
            avg_z = np.mean(current_seq_zs)
            std_z = np.std(current_seq_zs)
            GLOBAL_GROUND_Z.append(avg_z)
            print(f"  -> Ground Z Mean: {avg_z:.4f} m (Std: {std_z:.3f})")
        else:
            print(f"  -> [警告] 未检测到有效地面 (可能过滤范围不对)")

    print(f"\n{'='*60}")
    if GLOBAL_GROUND_Z:
        total_avg = np.mean(GLOBAL_GROUND_Z)
        total_std = np.std(GLOBAL_GROUND_Z)
        print(f"【所有序列汇总】")
        print(f"平均地面高度 (Ground Z): {total_avg:.4f} m")
        print(f"标准差 (波动):          {total_std:.4f} m")
        
        print(f"\n[结果分析]:")
        if abs(total_avg - (-1.7)) < 0.5:
            print(f"  > 结果接近 -1.7m，说明 'new_point_virt' 已经正确对齐到车端坐标系。")
            print(f"  > 此时 Ground Z 反映的是【主车 LiDAR 离地高度】。")
        elif abs(total_avg) > 5.0:
            print(f"  > 结果 ({total_avg:.2f}m) 较大，说明数据可能仍处于路侧坐标系（或 Z 轴未对齐）。")
        
        # 检查异常序列
        print(f"\n[异常值检查]:")
        outliers = []
        for i, seq in enumerate([s for s in seq_list if i < len(GLOBAL_GROUND_Z)]): # 简化对应逻辑，实际使用可能需要更严谨的 mapping
             # 这里的 mapping 只是简写，假设上面的 loop 没有 continue 导致错位。
             # 严谨写法是把 seq name 存进 tuple
             pass 
        
        # 简单打印偏离均值 > 0.3m 的
        for z in GLOBAL_GROUND_Z:
             if abs(z - total_avg) > 0.3:
                 print(f"  警告: 存在序列的地面 Z 值 ({z:.2f}m) 与总均值偏差较大，请检查标定！")
    else:
        print("未获取到任何数据。")