import os
import sys
import json
import numpy as np

# 确保能导入 opencood
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from opencood.utils.transformation_utils import x_to_world
except ImportError:
    sys.path.append(os.getcwd())
    from opencood.utils.transformation_utils import x_to_world

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

def generate_traj_json(root_dir, split_json_path, pred_len=5, save_name='train_traj_coop.json'):
    print(f"\n{'='*50}")
    print(f"开始生成轨迹标签: {save_name}")
    print(f"预测未来: {pred_len} 帧")
    print(f"{'='*50}")

    # 1. 加载 Split 列表
    if not os.path.exists(split_json_path):
        print(f"错误: 找不到拆分文件 {split_json_path}")
        return
        
    split_info = load_json(split_json_path)
    total_frames = len(split_info)
    
    database = {}
    valid_count = 0
    
    COOP_LABEL_DIR = os.path.join(root_dir, 'cooperative/label_world') 
    VEH_LABEL_DIR = os.path.join(root_dir, 'vehicle-side/label/lidar')

    for idx, current_frame_id in enumerate(split_info):
        if idx % 100 == 0:
            print(f"进度: {idx}/{total_frames}")

        # --- A. 准备当前帧 ---
        coop_json_path = os.path.join(COOP_LABEL_DIR, f'{current_frame_id}.json')
        if not os.path.exists(coop_json_path):
            continue
        coop_data = load_json(coop_json_path)
        
        veh_json_path = os.path.join(VEH_LABEL_DIR, f'{current_frame_id}.json')
        if not os.path.exists(veh_json_path):
            continue
        veh_data = load_json(veh_json_path)

        ego_pose = veh_data['sensor_pose'] 
        ego_pose_list = [ego_pose['x'], ego_pose['y'], ego_pose['z'], 
                         ego_pose['roll'], ego_pose['yaw'], ego_pose['pitch']]
        
        T_world_curr = x_to_world(ego_pose_list)
        T_curr_world = np.linalg.inv(T_world_curr)

        current_objects = coop_data.get('objects', [])
        current_obj_ids = [obj['id'] for obj in current_objects]
        num_objs = len(current_obj_ids)
        
        trajs = np.zeros((num_objs, pred_len, 2))
        masks = np.zeros((num_objs, pred_len))

        # --- B. 推算未来帧 ---
        # 【核心修改】适配带多重下划线的新名字 Town12_t_0_seq_0_0000
        # 我们需要切开最后一个下划线，把前面的作为 prefix，后面的 0000 作为编号
        if '_' in current_frame_id:
            # rsplit('_', 1) 意味着从右边开始切 1 次
            prefix, frame_num_str = current_frame_id.rsplit('_', 1)
        else:
            prefix, frame_num_str = "", current_frame_id

        try:
            current_id_int = int(frame_num_str)
            zfill_len = len(frame_num_str)
        except ValueError:
            print(f"  [警告] 无法解析帧号: {current_frame_id}")
            continue

        for t in range(1, pred_len + 1):
            # 1. 算下一帧文件名
            next_id_int = current_id_int + t
            next_num_str = str(next_id_int).zfill(zfill_len)
            next_frame_id = f"{prefix}_{next_num_str}" if prefix else next_num_str

            # 2. 检查未来文件是否存在
            future_coop_path = os.path.join(COOP_LABEL_DIR, f'{next_frame_id}.json')
            future_veh_path = os.path.join(VEH_LABEL_DIR, f'{next_frame_id}.json')
            
            if not os.path.exists(future_coop_path) or not os.path.exists(future_veh_path):
                continue
            
            # 3. 加载未来数据
            f_coop_data = load_json(future_coop_path)
            f_veh_data = load_json(future_veh_path)
            
            # 4. 未来的 Ego Pose
            fp = f_veh_data['sensor_pose']
            fp_list = [fp['x'], fp['y'], fp['z'], fp['roll'], fp['yaw'], fp['pitch']]
            T_world_future = x_to_world(fp_list)

            # 5. 构建查找表
            future_obj_dict = {}
            for obj in f_coop_data.get('objects', []):
                future_obj_dict[obj['id']] = obj['3d_location']

            # 6. 匹配与变换
            for k, obj_id in enumerate(current_obj_ids):
                if obj_id in future_obj_dict:
                    loc = future_obj_dict[obj_id]
                    pt_local = np.array([loc['x'], loc['y'], loc['z'], 1.0])
                    
                    pt_world = T_world_future @ pt_local
                    pt_curr = T_curr_world @ pt_world
                    
                    trajs[k, t-1, :] = pt_curr[:2]
                    masks[k, t-1] = 1.0

        # --- C. 存入字典 ---
        database[current_frame_id] = {
            'gt_ids': current_obj_ids,
            'gt_traj': trajs.tolist(),
            'gt_traj_mask': masks.tolist()
        }
        valid_count += 1

    # 保存文件 (放在 ROOT_DIR 下)
    save_path = os.path.join(root_dir, save_name)
    with open(save_path, 'w') as f:
        json.dump(database, f, indent=None)
    
    print(f"✅ 写入完成: {save_path} (有效帧数: {valid_count})")

if __name__ == '__main__':
    # ================= 配置区域 =================
    # 全局数据根目录
    ROOT_DIR = "/home/yty/mfh/carla_data/cooperative-vehicle-infrastructure" 
    
    # 预测未来 5 帧
    PRED_LEN = 5 
    
    # 定义需要处理的拆分文件及其对应的输出名字
    TASKS = [
        {"split": "train.json", "output": "train_traj_coop.json"},
        {"split": "test.json",  "output": "test_traj_coop.json"}
    ]
    # ============================================

    for task in TASKS:
        split_path = os.path.join(ROOT_DIR, task["split"])
        generate_traj_json(
            root_dir=ROOT_DIR, 
            split_json_path=split_path, 
            pred_len=PRED_LEN, 
            save_name=task["output"]
        )
    
    print("\n🎉 所有轨迹标签生成完毕！")