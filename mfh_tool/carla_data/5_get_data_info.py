# import os
# import json
# import glob
# from collections import defaultdict

# def get_filename_ids(folder_path, suffix):
#     """
#     获取指定文件夹下所有指定后缀文件的 ID (不包含扩展名)
#     返回一个 set 集合
#     """
#     if not os.path.exists(folder_path):
#         print(f"\033[91mWarning: Directory not found: {folder_path}\033[0m")
#         return set()
    
#     files = glob.glob(os.path.join(folder_path, f"*{suffix}"))
#     ids = {os.path.basename(f).replace(suffix, "") for f in files}
#     return ids

# def get_seq_name(frame_id):
#     # "seq01_000123" -> "seq01"
#     return frame_id.split("_")[0]

# def generate_data_info(root_dir, save_dir="cooperative"):
#     # 1. 定义关键路径
#     veh_bin_dir = os.path.join(root_dir, "vehicle-side/velodyne")
#     inf_bin_dir = os.path.join(root_dir, "infrastructure-side/velodyne")
    
#     # 标签路径
#     coop_lbl_dir = os.path.join(root_dir, "cooperative/label_world")
#     veh_lbl_dir  = os.path.join(root_dir, "vehicle-side/label/lidar")
#     inf_lbl_dir  = os.path.join(root_dir, "infrastructure-side/label/virtuallidar")
    
#     # 2. 获取各自文件夹下的 ID 集合
#     print("-" * 30)
#     print("Scanning directories...")
    
#     veh_ids = get_filename_ids(veh_bin_dir, ".bin")
#     inf_ids = get_filename_ids(inf_bin_dir, ".bin")
    
#     coop_lbl_ids = get_filename_ids(coop_lbl_dir, ".json")
#     veh_lbl_ids  = get_filename_ids(veh_lbl_dir, ".json")
#     inf_lbl_ids  = get_filename_ids(inf_lbl_dir, ".json")
    
#     print(f"Vehicle PC Frames:   {len(veh_ids)}")
#     print(f"Infra PC Frames:     {len(inf_ids)}")
#     print(f"Coop Label Frames:   {len(coop_lbl_ids)}")
#     print(f"Vehicle Label Frames:{len(veh_lbl_ids)}")
#     print(f"Infra Label Frames:  {len(inf_lbl_ids)}")

#     # 3. 取五者交集
#     valid_ids = (
#         veh_ids
#         .intersection(inf_ids)
#         .intersection(coop_lbl_ids)
#         .intersection(veh_lbl_ids)
#         .intersection(inf_lbl_ids)
#     )
    
#     sorted_ids = sorted(list(valid_ids))
    
#     if len(sorted_ids) == 0:
#         print("\033[91mError: No common frames found across ALL 5 folders!\033[0m")
#         return

#     print("-" * 30)
#     print(f"Found {len(sorted_ids)} common frames.")

#     # 4. 生成 data_info (详细信息)
#     data_info = []
#     for frame_id in sorted_ids:
#         frame_dict = {
#             "infrastructure_image_path": f"infrastructure-side/image/{frame_id}.jpg",
#             "infrastructure_pointcloud_path": f"infrastructure-side/velodyne/{frame_id}.bin",
#             "vehicle_image_path": f"vehicle-side/image/{frame_id}.jpg",
#             "vehicle_pointcloud_path": f"vehicle-side/velodyne/{frame_id}.bin",

#             "cooperative_label_path": f"cooperative/label_world/{frame_id}.json",
#             "vehicle_label_path": f"vehicle-side/label/lidar/{frame_id}.json",
#             "infrastructure_label_path": f"infrastructure-side/label/virtuallidar/{frame_id}.json",
            
#             "system_error_offset": {"delta_x": 0.0, "delta_y": 0.0}
#         }
#         data_info.append(frame_dict)

#     # 5.1 按 seq 分组 frame_id
#     seq_to_ids = defaultdict(list)
#     for fid in sorted_ids:
#         seq_name = get_seq_name(fid)
#         seq_to_ids[seq_name].append(fid)

#     num_seqs = len(seq_to_ids)
#     total_frames = len(sorted_ids)

#     # ================= 核心修改区域：基于帧数的贪心划分 =================
#     # 计算我们期望的训练集和测试集的帧数目标 (7:3)
#     target_train_frames = total_frames * 0.7
#     target_test_frames = total_frames * 0.3

#     # 将 seq 按照包含的帧数从大到小排序 (Greedy 策略的核心)
#     seq_lengths = [(seq, len(fids)) for seq, fids in seq_to_ids.items()]
#     seq_lengths.sort(key=lambda x: x[1], reverse=True)

#     train_seqs = []
#     test_seqs = []
#     train_frames_count = 0
#     test_frames_count = 0

#     # 遍历排序后的序列，优先分配给当前距离目标差距更大的集合
#     for seq, length in seq_lengths:
#         # 计算当前集合缺少的帧数
#         deficit_train = target_train_frames - train_frames_count
#         deficit_test = target_test_frames - test_frames_count

#         if deficit_train >= deficit_test:
#             train_seqs.append(seq)
#             train_frames_count += length
#         else:
#             test_seqs.append(seq)
#             test_frames_count += length

#     # 为了打印和后续处理好看，对选出来的 seq 名字重新按字母排序
#     train_seqs.sort()
#     test_seqs.sort()
#     # ====================================================================

#     # 5.3 根据分配好的 seq 展开成最终的 frame_id 列表
#     train_ids = []
#     test_ids = []
#     for seq in train_seqs:
#         train_ids.extend(seq_to_ids[seq])
#     for seq in test_seqs:
#         test_ids.extend(seq_to_ids[seq])

#     # 打印划分结果，验证比例是否接近 7:3
#     print("-" * 30)
#     print(f"Total seqs: {num_seqs}")
#     print(f"Train seqs ({len(train_seqs)}): {train_seqs}")
#     print(f"Test seqs  ({len(test_seqs)}): {test_seqs}")
    
#     actual_train_ratio = len(train_ids) / total_frames
#     actual_test_ratio = len(test_ids) / total_frames
#     print(f"Train frames: {len(train_ids)} ({actual_train_ratio:.1%}) | Test frames: {len(test_ids)} ({actual_test_ratio:.1%})")

#     # 6.1 保存完整的 data_info.json (包含路径字典)
#     coop_dir = os.path.join(root_dir, save_dir)
#     os.makedirs(coop_dir, exist_ok=True)
#     with open(os.path.join(coop_dir, "data_info.json"), "w") as f:
#         json.dump(data_info, f, indent=2)
        
#     # 6.2 保存 train.json
#     with open(os.path.join(root_dir, "train.json"), "w") as f:
#         json.dump(train_ids, f, indent=2)
        
#     # 6.3 保存 test.json
#     with open(os.path.join(root_dir, "test.json"), "w") as f:
#         json.dump(test_ids, f, indent=2)
    
#     print(f"Successfully generated files in {root_dir}")

# if __name__ == "__main__":
#     my_data_root = "/home/yty/mfh/carla_data/Town12_t_0/cooperative-vehicle-infrastructure"
#     generate_data_info(my_data_root)

import os
import json
import glob
from collections import defaultdict

def get_filename_ids(folder_path, suffix):
    """
    获取指定文件夹下所有指定后缀文件的 ID (不包含扩展名)
    返回一个 set 集合
    """
    if not os.path.exists(folder_path):
        print(f"\033[91mWarning: Directory not found: {folder_path}\033[0m")
        return set()
    
    files = glob.glob(os.path.join(folder_path, f"*{suffix}"))
    ids = {os.path.basename(f).replace(suffix, "") for f in files}
    return ids

def get_seq_name(frame_id):
    # 【核心修改 1】适配带 Town 前缀的新命名规则
    # 新格式: Town12_t_0_seq_0_0000
    # 最后一个 "_" 后面是具体的帧号(0000)，前面的全都是序列名标识
    # 我们用 '_' 分割，然后把除了最后一部分之外的所有部分再拼起来
    parts = frame_id.split("_")
    seq_name = "_".join(parts[:-1]) 
    # 返回结果如: "Town12_t_0_seq_0"
    return seq_name

def generate_data_info(root_dir, save_dir="cooperative"):
    # 1. 定义关键路径
    veh_bin_dir = os.path.join(root_dir, "vehicle-side/velodyne")
    inf_bin_dir = os.path.join(root_dir, "infrastructure-side/velodyne")
    
    # 标签路径
    coop_lbl_dir = os.path.join(root_dir, "cooperative/label_world")
    veh_lbl_dir  = os.path.join(root_dir, "vehicle-side/label/lidar")
    inf_lbl_dir  = os.path.join(root_dir, "infrastructure-side/label/virtuallidar")
    
    # 2. 获取各自文件夹下的 ID 集合
    print("-" * 30)
    print("Scanning directories...")
    
    veh_ids = get_filename_ids(veh_bin_dir, ".bin")
    inf_ids = get_filename_ids(inf_bin_dir, ".bin")
    
    coop_lbl_ids = get_filename_ids(coop_lbl_dir, ".json")
    veh_lbl_ids  = get_filename_ids(veh_lbl_dir, ".json")
    inf_lbl_ids  = get_filename_ids(inf_lbl_dir, ".json")
    
    print(f"Vehicle PC Frames:   {len(veh_ids)}")
    print(f"Infra PC Frames:     {len(inf_ids)}")
    print(f"Coop Label Frames:   {len(coop_lbl_ids)}")
    print(f"Vehicle Label Frames:{len(veh_lbl_ids)}")
    print(f"Infra Label Frames:  {len(inf_lbl_ids)}")

    # 3. 取五者交集
    valid_ids = (
        veh_ids
        .intersection(inf_ids)
        .intersection(coop_lbl_ids)
        .intersection(veh_lbl_ids)
        .intersection(inf_lbl_ids)
    )
    
    sorted_ids = sorted(list(valid_ids))
    
    if len(sorted_ids) == 0:
        print("\033[91mError: No common frames found across ALL 5 folders!\033[0m")
        return

    print("-" * 30)
    print(f"Found {len(sorted_ids)} common frames.")

    # 4. 生成 data_info (详细信息)
    data_info = []
    for frame_id in sorted_ids:
        frame_dict = {
            "infrastructure_image_path": f"infrastructure-side/image/{frame_id}.jpg",
            "infrastructure_pointcloud_path": f"infrastructure-side/velodyne/{frame_id}.bin",
            "vehicle_image_path": f"vehicle-side/image/{frame_id}.jpg",
            "vehicle_pointcloud_path": f"vehicle-side/velodyne/{frame_id}.bin",

            "cooperative_label_path": f"cooperative/label_world/{frame_id}.json",
            "vehicle_label_path": f"vehicle-side/label/lidar/{frame_id}.json",
            "infrastructure_label_path": f"infrastructure-side/label/virtuallidar/{frame_id}.json",
            
            "system_error_offset": {"delta_x": 0.0, "delta_y": 0.0}
        }
        data_info.append(frame_dict)

    # 5.1 按 seq 分组 frame_id
    seq_to_ids = defaultdict(list)
    for fid in sorted_ids:
        seq_name = get_seq_name(fid)
        seq_to_ids[seq_name].append(fid)

    num_seqs = len(seq_to_ids)
    total_frames = len(sorted_ids)

    # ================= 核心修改区域：基于帧数的贪心划分 =================
    # 计算我们期望的训练集和测试集的帧数目标 (7:3)
    target_train_frames = total_frames * 0.7
    target_test_frames = total_frames * 0.3

    # 将 seq 按照包含的帧数从大到小排序 (Greedy 策略的核心)
    seq_lengths = [(seq, len(fids)) for seq, fids in seq_to_ids.items()]
    seq_lengths.sort(key=lambda x: x[1], reverse=True)

    train_seqs = []
    test_seqs = []
    train_frames_count = 0
    test_frames_count = 0

    # 遍历排序后的序列，优先分配给当前距离目标差距更大的集合
    for seq, length in seq_lengths:
        # 计算当前集合缺少的帧数
        deficit_train = target_train_frames - train_frames_count
        deficit_test = target_test_frames - test_frames_count

        if deficit_train >= deficit_test:
            train_seqs.append(seq)
            train_frames_count += length
        else:
            test_seqs.append(seq)
            test_frames_count += length

    # 为了打印和后续处理好看，对选出来的 seq 名字重新按字母排序
    train_seqs.sort()
    test_seqs.sort()
    # ====================================================================

    # 5.3 根据分配好的 seq 展开成最终的 frame_id 列表
    train_ids = []
    test_ids = []
    for seq in train_seqs:
        train_ids.extend(seq_to_ids[seq])
    for seq in test_seqs:
        test_ids.extend(seq_to_ids[seq])

    # 打印划分结果，验证比例是否接近 7:3
    print("-" * 30)
    print(f"Total seqs: {num_seqs}")
    print(f"Train seqs ({len(train_seqs)}): {train_seqs}")
    print(f"Test seqs  ({len(test_seqs)}): {test_seqs}")
    
    actual_train_ratio = len(train_ids) / total_frames if total_frames > 0 else 0
    actual_test_ratio = len(test_ids) / total_frames if total_frames > 0 else 0
    print(f"Train frames: {len(train_ids)} ({actual_train_ratio:.1%}) | Test frames: {len(test_ids)} ({actual_test_ratio:.1%})")

    # 6.1 保存完整的 data_info.json (包含路径字典)
    coop_dir = os.path.join(root_dir, save_dir)
    os.makedirs(coop_dir, exist_ok=True)
    with open(os.path.join(coop_dir, "data_info.json"), "w") as f:
        json.dump(data_info, f, indent=2)
        
    # 6.2 保存 train.json
    with open(os.path.join(root_dir, "train.json"), "w") as f:
        json.dump(train_ids, f, indent=2)
        
    # 6.3 保存 test.json
    with open(os.path.join(root_dir, "test.json"), "w") as f:
        json.dump(test_ids, f, indent=2)
    
    print(f"Successfully generated files in {root_dir}")

if __name__ == "__main__":
    # 【核心修改 2】修改为我们刚才生成的全局多地图合并路径
    my_data_root = "/home/yty/mfh/carla_data/cooperative-vehicle-infrastructure"
    generate_data_info(my_data_root)