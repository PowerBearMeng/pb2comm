import os
import json
import copy
import random
from collections import defaultdict

# ================= 配置区域 =================
# 【核心修改 1】数据的全局根目录
ROOT_DIR = "/home/yty/mfh/carla_data/cooperative-vehicle-infrastructure"

# 输入文件 (由你之前的 generate_data_info 脚本生成，只读)
DATA_INFO_PATH = os.path.join(ROOT_DIR, "cooperative/data_info.json")
TRAIN_SPLIT_PATH = os.path.join(ROOT_DIR, "train.json")
TEST_SPLIT_PATH  = os.path.join(ROOT_DIR, "test.json")

# 输出目录
OUTPUT_DIR = os.path.join(ROOT_DIR, "flow_data_jsons") # 建议换个名字以示区别
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 训练时允许的随机延迟范围 (帧数, 1帧=100ms)
# [1, 2] 代表训练时网络会随机遇到 100ms, 200ms 的延迟情况
TRAIN_DELAY_CHOICES = [1, 2]

# 验证时需要生成的固定延迟列表
VAL_DELAY_LIST = [1, 2, 3] 


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {len(data)} samples to: {path}")

def get_frame_id_from_path(path):
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0]

def build_seq_map(raw_data_info):
    """
    预处理：建立 ID 索引 和 序列分组
    """
    id_to_info = {}
    seq_map = defaultdict(list)
    
    for item in raw_data_info:
        # 假设 ID 在 vehicle_pointcloud_path 文件名里
        fid = get_frame_id_from_path(item['vehicle_pointcloud_path'])
        id_to_info[fid] = item
        
        # 【核心修改 2】解析序列名，适配多地图前缀
        # 比如把 "Town12_t_0_seq_0_0000" 变成 "Town12_t_0_seq_0"
        parts = fid.split('_')
        seq_name = "_".join(parts[:-1])
        seq_map[seq_name].append(fid)
    
    # 确保序列内按时间排序 (字符串排序对于固定位数数字是安全的)
    for seq in seq_map:
        seq_map[seq].sort()
        
    return id_to_info, seq_map

def generate_aligned_flow_data(target_ids, id_to_info, seq_map, mode='val', fixed_delay=0):
    """
    生成严格对齐的时序数据
    逻辑：Target(T2) 必须是 target_ids 里存在的帧，保证标签存在。
          Input(T1) 根据 delay 反推。
    """
    flow_list = []
    target_ids_set = set(target_ids)
    
    for seq_name, frames in seq_map.items():
        # frames: [Town12_t_0_seq_0_0000, Town12_t_0_seq_0_0001, ...]
        
        for i, current_fid in enumerate(frames):
            # 1. 只有当前帧在 split (train/test) 中，才把它当作 Target (T2)
            # 这样保证了 T2 拥有对应的 GT Label
            if current_fid not in target_ids_set:
                continue
            
            # 2. 确定延迟 (Delay k)
            if mode == 'train':
                # 训练模式：随机选择一种延迟，让模型学会处理不同情况
                delay_k = random.choice(TRAIN_DELAY_CHOICES)
            else:
                # 验证模式：使用固定延迟
                delay_k = fixed_delay
                
            # 3. 反推索引 (Back-tracing)
            # T2 (Target/GT) = i
            # T1 (Input)     = i - delay_k
            # T0 (History)   = T1 - 1
            
            idx_t2 = i
            idx_t1 = i - delay_k
            idx_t0 = idx_t1 - 1
            
            # 4. 边界检查
            # 必须保证 T0 存在且属于同一个 Sequence (idx >= 0)
            if idx_t0 < 0:
                continue
                
            # 获取 ID
            fid_t2 = frames[idx_t2]
            fid_t1 = frames[idx_t1]
            fid_t0 = frames[idx_t0]
            
            # 获取原始信息字典
            info_t2 = id_to_info[fid_t2]
            info_t1 = id_to_info[fid_t1]
            info_t0 = id_to_info[fid_t0]
            
            # 5. 构建 FFNet 样本
            # 复制 T2 的信息，这样该样本的 Label 就是 T2 的 Label
            new_item = copy.deepcopy(info_t2)
            
            # --- 写入关键路径 ---
            # T2 (Target)
            if os.path.exists(os.path.join(ROOT_DIR, info_t2['infrastructure_pointcloud_path'])):
                new_item['infrastructure_pointcloud_bin_path_t_2'] = info_t2['infrastructure_pointcloud_path']
            else:
                raise FileNotFoundError(f"T2 infrastructure pointcloud not found: {info_t2['infrastructure_pointcloud_path']}")
            # T1 (Input)
            if os.path.exists(os.path.join(ROOT_DIR, info_t1['infrastructure_pointcloud_path'])):
                new_item['infrastructure_pointcloud_bin_path_t_1'] = info_t1['infrastructure_pointcloud_path']
            else:
                raise FileNotFoundError(f"T1 infrastructure pointcloud not found: {info_t1['infrastructure_pointcloud_path']}")
            # T0 (History)
            if os.path.exists(os.path.join(ROOT_DIR, info_t0['infrastructure_pointcloud_path'])):
                new_item['infrastructure_pointcloud_bin_path_t_0'] = info_t0['infrastructure_pointcloud_path']
            else:
                raise FileNotFoundError(f"T0 infrastructure pointcloud not found: {info_t0['infrastructure_pointcloud_path']}")
            
            # --- 写入时间差参数 (Delta T) ---
            new_item['infrastructure_t_0_1'] = 1.0          # T0 到 T1 永远差 1 帧
            new_item['infrastructure_t_1_2'] = float(delay_k) # T1 到 T2 差 k 帧
            
            new_item['debug_info'] = {
                'delay_frames': delay_k,
                't2_id': fid_t2,
                't1_id': fid_t1
            }
            
            flow_list.append(new_item)
            
    return flow_list

def filter_split_by_flow(flow_data, original_split_ids):
    """
    根据 flow 数据中实际出现的 T2 帧，过滤 split
    """
    used_t2_ids = set(
        item['debug_info']['t2_id'] for item in flow_data
    )
    
    original_set = set(original_split_ids)
    filtered_ids = sorted(original_set & used_t2_ids)
    
    return filtered_ids


if __name__ == "__main__":
    # 设置随机种子，保证每次生成的训练集是一样的 (可复现性)
    random.seed(2026)
    
    print(f"Loading metadata...")
    raw_data_info = load_json(DATA_INFO_PATH)
    train_ids = load_json(TRAIN_SPLIT_PATH)
    test_ids  = load_json(TEST_SPLIT_PATH)
    
    # 建立索引
    id_to_info, seq_map = build_seq_map(raw_data_info)
    
    print("=" * 50)
    print(f"Generating Training Data (Random Delay: {TRAIN_DELAY_CHOICES})...")
    # 生成训练集
    train_data = generate_aligned_flow_data(train_ids, id_to_info, seq_map, mode='train')
    save_json(train_data, os.path.join(OUTPUT_DIR, "flow_train.json"))
    
    print("\n" + "=" * 50)
    print("Generating Validation Data (Fixed Delays)...")
    # 生成多个验证集
    for k in VAL_DELAY_LIST:
        val_data = generate_aligned_flow_data(test_ids, id_to_info, seq_map, mode='val', fixed_delay=k)
        
        fname = f"flow_val_delay_{k}.json"
        save_json(val_data, os.path.join(OUTPUT_DIR, fname))
        
    filtered_train_ids = filter_split_by_flow(train_data, train_ids)
    save_json(
        filtered_train_ids,
        os.path.join(ROOT_DIR, "train.json")
    )

    MAX_POSSIBLE_DELAY = 3  
    
    print(f"\nFiltering split based on MAX delay = {MAX_POSSIBLE_DELAY}...")

    val_data_ref = generate_aligned_flow_data(
        test_ids, id_to_info, seq_map, mode='val', fixed_delay=MAX_POSSIBLE_DELAY
    )

    filtered_test_ids = filter_split_by_flow(val_data_ref, test_ids)
    save_json(
        filtered_test_ids,
        os.path.join(ROOT_DIR, "test.json")
    )

    print("\nAll done! Output directory:", OUTPUT_DIR)