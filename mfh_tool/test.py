import json
data_dir = "/home/yty/mfh/DAIR-V2X/V2X-C/cooperative-vehicle-infrastructure/cooperative/data_info.json"

root_dir = "/home/yty/mfh/DAIR-V2X/V2X-C/cooperative-vehicle-infrastructure/train.json"
test_dir = "/home/yty/mfh/DAIR-V2X/V2X-C/cooperative-vehicle-infrastructure/val.json"

with open(data_dir, 'r') as f:
    data = json.load(f)
print(f"Total samples in data_json: {len(data)}")
with open(root_dir, 'r') as f:
    data = json.load(f)
print(f"Total samples in training set: {len(data)}")

with open(test_dir, 'r') as f:
    data = json.load(f)
print(f"Total samples in test set: {len(data)}")


split_path = "/home/yty/mfh/code/inter/Where2comm/mfh_tool/cooperative-split-data.json"
with open(split_path, 'r') as f:
    split_data = json.load(f)
print("111")
