import json
import os
import math
import copy
import shutil


def process_format_with_conversion(input_dir, output_dir):
    """
    CARLA -> DAIR-V2X 转换模式:
    1. 字段重命名: class->type, location->3d_location, dimensions->3d_dimensions
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"找到 {len(file_list)} 个文件，准备处理 (包含坐标转换)...")
    start_z = 0  # 可根据需要调整起始高度偏移
    for filename in file_list:
        file_path = os.path.join(input_dir, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        new_data = copy.deepcopy(original_data)
        if 'objects' in new_data:
            new_objects = []
            for obj in new_data['objects']:
                new_obj = {}

                # --- 1. class -> type ---
                new_obj['type'] = obj.get('class', 'Car')
                raw_loc = obj.get('location', {})
                new_obj['3d_location'] = {
                    'x': raw_loc.get('x', 0),
                    'y': raw_loc.get('y', 0), 
                    'z': raw_loc.get('z', 0)
                }
                
                new_obj['3d_dimensions'] = obj.get('dimensions', {})
                # 兼容缺字段
                raw_dim = obj.get('dimensions', {})
                l = raw_dim.get('l', raw_dim.get('length', 0))
                w = raw_dim.get('w', raw_dim.get('width', 0))
                h = raw_dim.get('h', raw_dim.get('height', 0))
                new_obj['3d_dimensions'] = {
                'h': h,
                'w': w,
                'l': l
                }
                # --- 4. rotation -> rotation (提取yaw, 转弧度, 取反) ---
                raw_rot = obj.get('rotation', {})
                
                if isinstance(raw_rot, dict):
                    # CARLA 的 yaw 是度数
                    raw_yaw_deg = raw_rot.get('yaw', 0.0)
                    
                    # 1. 转弧度
                    yaw_rad = math.radians(raw_yaw_deg)
                    final_yaw = yaw_rad
                    
                    new_obj['rotation'] = final_yaw
                else:
                    # 如果原数据已经是单值，视情况是否需要处理
                    new_obj['rotation'] = raw_rot

                # 保留 id
                if 'id' in obj:
                    new_obj['id'] = obj['id']
                
                # 补充: DAIR-V2X 格式有时还需要 'occluded', 'truncated', 'alpha' 等字段
                # 如果没有，可以设为默认值
                new_obj['occluded'] = 0
                new_obj['truncated'] = 0
                new_obj['alpha'] = 0

                new_objects.append(new_obj)
            
            new_data['objects'] = new_objects

        save_path = os.path.join(output_dir, filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2)
            
    print(f"转换完成！保存至: {output_dir}")
def reset_dir(dir_path):
    """
    如果目录存在：删掉整个目录
    然后重新创建一个空目录
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

# if __name__ == "__main__":
#     ROOT_DIR = "/home/yty/mfh/carla_data/Town12_t_0"
#     seq_list = sorted(os.listdir(ROOT_DIR))
    
#     # 跳过非文件夹项，防止报错
#     seq_list = [s for s in seq_list if os.path.isdir(os.path.join(ROOT_DIR, s))]
    
#     side_list = ["vehicle", "world"]
#     DEST_DIR = os.path.join(ROOT_DIR, "cooperative-vehicle-infrastructure")

#     # ================= 1. 先清空/创建目标根目录 (只做一次！) =================
#     # 定义好所有需要的目标子目录
#     dest_world_label_dir = os.path.join(DEST_DIR, "cooperative", "label_world")
#     dest_veh_label_dir   = os.path.join(DEST_DIR, "vehicle-side", "label", "lidar")
#     dest_veh_point_dir   = os.path.join(DEST_DIR, "vehicle-side", "velodyne")
#     # 如果你也需要路侧点云，这里定义一下，比如：
#     dest_world_point_dir = os.path.join(DEST_DIR, "infrastructure-side", "velodyne")

#     # 初始化目录 (清空旧数据)
#     print("正在初始化目标目录...")
#     reset_dir(dest_world_label_dir)
#     reset_dir(dest_veh_label_dir)
#     reset_dir(dest_veh_point_dir)
#     reset_dir(dest_world_point_dir) # 如果需要路侧点云
#     # ====================================================================

#     # 开始遍历
#     for seq in seq_list: 
#         # 简单过滤一下，只处理 seq 开头的文件夹
#         if not seq.startswith("seq"): 
#             continue 

#         for side in side_list:
#             print(f"Processing sequence: {seq}, side: {side}")
            
#             INPUT_DIR = os.path.join(ROOT_DIR, seq, side, "labels")
#             OUTPUT_DIR = os.path.join(ROOT_DIR, seq, side, "new_labels")
            
#             # 1. 转换 Label 格式
#             if os.path.exists(INPUT_DIR):
#                 process_format_with_conversion(INPUT_DIR, OUTPUT_DIR)
#             else:
#                 print(f"警告: 找不到输入目录 {INPUT_DIR}")
#                 continue

#             # 2. 移动文件 (Copy & Rename)
#             if side == "world":
#                 # --- 移动 Label ---
#                 if os.path.exists(OUTPUT_DIR):
#                     print(f"Moving world labels from {OUTPUT_DIR} to {dest_world_label_dir} ...")
#                     for fname in os.listdir(OUTPUT_DIR):
#                         if not fname.endswith(".json"): continue
#                         src = os.path.join(OUTPUT_DIR, fname)
#                         new_name = f"{seq}_{fname}"
#                         dst = os.path.join(dest_world_label_dir, new_name)
#                         shutil.copy2(src, dst)
#                 # world下没有点云

#             elif side == "vehicle":
#                 # --- 移动 Label ---
#                 if os.path.exists(OUTPUT_DIR):
#                     for fname in os.listdir(OUTPUT_DIR):
#                         if not fname.endswith(".json"): continue
#                         src = os.path.join(OUTPUT_DIR, fname)
#                         new_name = f"{seq}_{fname}"
#                         dst = os.path.join(dest_veh_label_dir, new_name)
#                         shutil.copy2(src, dst)
                
#                 # --- 移动 Point Cloud (车侧) ---
#                 points_src_dir = os.path.join(ROOT_DIR, seq, side, "points")
#                 if os.path.exists(points_src_dir):
#                     for fname in os.listdir(points_src_dir):
#                         if not fname.endswith(".bin"): continue
#                         src = os.path.join(points_src_dir, fname)
#                         new_name = f"{seq}_{fname}"
#                         dst = os.path.join(dest_veh_point_dir, new_name)
#                         shutil.copy2(src, dst)
                    

if __name__ == "__main__":
    # ================= 1. 定义基础路径和目标 =================
    BASE_DIR = "/home/yty/mfh/carla_data"
    
    # 你想要处理的 Town 列表，可以随时增加
    town_list = ["Town12_t_0", "Town12_t_1"] 
    
    side_list = ["vehicle", "world"]
    
    # 统一的全局输出目录，放在 BASE_DIR 下
    DEST_DIR = os.path.join(BASE_DIR, "cooperative-vehicle-infrastructure")

    # 定义所有需要的目标子目录
    dest_world_label_dir = os.path.join(DEST_DIR, "cooperative", "label_world")
    dest_veh_label_dir   = os.path.join(DEST_DIR, "vehicle-side", "label", "lidar")
    dest_veh_point_dir   = os.path.join(DEST_DIR, "vehicle-side", "velodyne")
    dest_world_point_dir = os.path.join(DEST_DIR, "infrastructure-side", "velodyne")

    # ================= 2. 初始化目标目录 (全局只做一次！) =================
    print("正在初始化全局目标目录...")
    reset_dir(dest_world_label_dir)
    reset_dir(dest_veh_label_dir)
    reset_dir(dest_veh_point_dir)
    reset_dir(dest_world_point_dir) # 如果需要路侧点云
    # ====================================================================

    # ================= 3. 开始外层遍历不同的 Town =================
    for town in town_list:
        ROOT_DIR = os.path.join(BASE_DIR, town)
        
        # 检查该 Town 文件夹是否存在
        if not os.path.exists(ROOT_DIR):
            print(f"跳过: 找不到目录 {ROOT_DIR}")
            continue
            
        print(f"\n================ 开始处理: {town} ================")
        
        seq_list = sorted(os.listdir(ROOT_DIR))
        # 跳过非文件夹项，防止报错
        seq_list = [s for s in seq_list if os.path.isdir(os.path.join(ROOT_DIR, s))]

        for seq in seq_list: 
            # 简单过滤一下，只处理 seq 开头的文件夹
            if not seq.startswith("seq"): 
                continue 

            for side in side_list:
                print(f"Processing - Town: {town}, Sequence: {seq}, Side: {side}")
                
                INPUT_DIR = os.path.join(ROOT_DIR, seq, side, "labels")
                OUTPUT_DIR = os.path.join(ROOT_DIR, seq, side, "new_labels")
                
                # 1. 转换 Label 格式
                if os.path.exists(INPUT_DIR):
                    process_format_with_conversion(INPUT_DIR, OUTPUT_DIR)
                else:
                    print(f"警告: 找不到输入目录 {INPUT_DIR}")
                    continue

                # 2. 移动文件 (Copy & Rename)
                if side == "world":
                    # --- 移动 Label ---
                    if os.path.exists(OUTPUT_DIR):
                        for fname in os.listdir(OUTPUT_DIR):
                            if not fname.endswith(".json"): continue
                            src = os.path.join(OUTPUT_DIR, fname)
                            # 【核心修改】文件名加上 town 前缀防止冲突
                            new_name = f"{town}_{seq}_{fname}" 
                            dst = os.path.join(dest_world_label_dir, new_name)
                            shutil.copy2(src, dst)
                    # world下没有点云

                elif side == "vehicle":
                    # --- 移动 Label ---
                    if os.path.exists(OUTPUT_DIR):
                        for fname in os.listdir(OUTPUT_DIR):
                            if not fname.endswith(".json"): continue
                            src = os.path.join(OUTPUT_DIR, fname)
                            # 【核心修改】文件名加上 town 前缀防止冲突
                            new_name = f"{town}_{seq}_{fname}"
                            dst = os.path.join(dest_veh_label_dir, new_name)
                            shutil.copy2(src, dst)
                    
                    # --- 移动 Point Cloud (车侧) ---
                    points_src_dir = os.path.join(ROOT_DIR, seq, side, "points")
                    if os.path.exists(points_src_dir):
                        for fname in os.listdir(points_src_dir):
                            if not fname.endswith(".bin"): continue
                            src = os.path.join(points_src_dir, fname)
                            # 【核心修改】文件名加上 town 前缀防止冲突
                            new_name = f"{town}_{seq}_{fname}"
                            dst = os.path.join(dest_veh_point_dir, new_name)
                            shutil.copy2(src, dst)