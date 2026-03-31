import json

# 1. 加载包含完整信息的车端 data_info.json
input_json_path = '/home/yty/mfh/DAIR-V2X/V2X-C/cooperative-vehicle-infrastructure/vehicle-side/data_info.json'  # 替换成你的文件路径
output_split_path = 'my_temporal_split.json'

with open(input_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 按 batch_id 将数据分组
batches = {}
for item in data:
    b_id = item['batch_id']
    if b_id not in batches:
        batches[b_id] = []
    batches[b_id].append(item)

# 3. 确定每个 batch 的时间顺序
batch_time_info = []
for b_id, frames in batches.items():
    # 确保单个 batch 内部的数据按时间排好序
    frames_sorted = sorted(frames, key=lambda x: int(x['image_timestamp']))
    batches[b_id] = frames_sorted
    
    # 提取这个 batch 第一帧的时间戳，作为该 batch 的“起始时间”
    batch_start_time = int(frames_sorted[0]['image_timestamp'])
    batch_time_info.append({
        'batch_id': b_id,
        'start_time': batch_start_time
    })

# 4. 核心：将所有的 batch 按照“起始时间”从早到晚进行排序！
batch_time_info_sorted = sorted(batch_time_info, key=lambda x: x['start_time'])
sorted_batch_ids = [item['batch_id'] for item in batch_time_info_sorted]

# 5. 按比例划分 Batch (例如：70% Train, 15% Val, 15% Test)
total_batches = len(sorted_batch_ids)
train_end = int(total_batches * 0.7)
val_end = int(total_batches * 0.85)

train_batches = sorted_batch_ids[:train_end]
val_batches = sorted_batch_ids[train_end:val_end]
test_batches = sorted_batch_ids[val_end:]

# 6. 提取 DAIR-V2X 框架需要的纯帧 ID (例如从 "image/000000.jpg" 提取 "000000")
def extract_frame_ids(batch_id_list):
    frame_ids = []
    for b_id in batch_id_list:
        for frame in batches[b_id]:
            # 获取类似于 "000000" 的纯 ID
            frame_id = frame['image_path'].split('/')[-1].split('.')[0]
            frame_ids.append(frame_id)
    return frame_ids

dair_v2x_split = {
    "train": extract_frame_ids(train_batches),
    "val": extract_frame_ids(val_batches),
    "test": extract_frame_ids(test_batches)
}

# 7. 保存结果
with open(output_split_path, 'w', encoding='utf-8') as f:
    json.dump(dair_v2x_split, f, indent=4)

# 打印最终统计信息
print("✅ 完美的时序划分完成！")
print(f"总计 Batches: {total_batches}")
print(f"Train: {len(train_batches)} batches, {len(dair_v2x_split['train'])} frames (过去的时间)")
print(f"Val:   {len(val_batches)} batches, {len(dair_v2x_split['val'])} frames (中间的时间)")
print(f"Test:  {len(test_batches)} batches, {len(dair_v2x_split['test'])} frames (未来的时间)")