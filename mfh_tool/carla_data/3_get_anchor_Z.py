import os
import json
import numpy as np

def collect_z_from_dir(dir_path, class_name='Car'):
    """
    从指定目录读取所有 json 文件，提取 Z 值并返回一个列表。
    """
    z_values = []
    if not os.path.exists(dir_path):
        return z_values

    files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
    for file in files:
        file_path = os.path.join(dir_path, file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception:
            continue

        objects = data.get('objects', [])
        for obj in objects:
            # 兼容 type 和 class 字段
            obj_type = obj.get('type', obj.get('class'))
            if obj_type == class_name:
                # 获取 Z 值 (兼容不同格式)
                loc = obj.get('3d_location')
                if not loc: loc = obj.get('location')
                
                z = None
                if isinstance(loc, dict):
                    z = loc.get('z')
                elif isinstance(loc, (list, tuple)) and len(loc) > 2:
                    z = loc[2]
                
                # 简单过滤无效值
                if z is not None and -10.0 < z < 10.0:
                    z_values.append(z)
    return z_values

def print_stats(title, z_data):
    """
    打印一组 Z 值的统计信息
    """
    if len(z_data) > 0:
        mean_z = np.mean(z_data)
        std_z = np.std(z_data)
        min_z = np.min(z_data)
        max_z = np.max(z_data)
        print(f"  > {title}:")
        print(f"    - 数量: {len(z_data)}")
        print(f"    - 均值 (Mean): {mean_z:.4f} m")
        print(f"    - 方差 (Std):  {std_z:.4f} m")
        # print(f"    - 范围: [{min_z:.2f}, {max_z:.2f}]")
        return mean_z
    else:
        print(f"  > {title}: 无有效数据")
        return None

if __name__ == "__main__":
    # ================= 配置区域 =================
    BASE_DIR = "/home/yty/mfh/carla_data"
    town_list = ["Town12_t_0", "Town12_t_1"] 
    TARGET_CLASS = 'Car'
    # ===========================================

    # 超全局容器，用于存储所有 Town 中所有序列的数据
    SUPER_GLOBAL_VEH_Z = []
    SUPER_GLOBAL_INFRA_Z = []

    # --- 1. 最外层遍历 Town ---
    for town in town_list:
        ROOT_DIR = os.path.join(BASE_DIR, town)
        if not os.path.exists(ROOT_DIR):
            print(f"跳过: 找不到目录 {ROOT_DIR}")
            continue
            
        print(f"\n================ 开始扫描地图: {town} ================")
        
        # 自动扫描当前 Town 下所有以 seq 开头的文件夹
        seq_list = sorted([d for d in os.listdir(ROOT_DIR) if d.startswith("seq") and os.path.isdir(os.path.join(ROOT_DIR, d))])
        
        if not seq_list:
            print(f"  [警告] {town} 下没有找到 seq 文件夹")
            continue
            
        print(f"找到序列: {seq_list}\n")

        # --- 2. 遍历当前 Town 的每个序列收集数据 ---
        for seq in seq_list:
            print(f"正在处理序列: {town}/{seq} ...")
            
            # 注意：这里读取的是你刚刚用脚本2生成的临时目录 (new_labels 和 new_labels_virt)
            veh_dir = os.path.join(ROOT_DIR, seq, "vehicle", "new_labels")
            infra_dir = os.path.join(ROOT_DIR, seq, "roadside0", "new_labels_virt")
            
            # 收集当前序列的数据
            cur_veh_z = collect_z_from_dir(veh_dir, TARGET_CLASS)
            cur_infra_z = collect_z_from_dir(infra_dir, TARGET_CLASS)
            
            # 存入超全局列表
            SUPER_GLOBAL_VEH_Z.extend(cur_veh_z)
            SUPER_GLOBAL_INFRA_Z.extend(cur_infra_z)

            # (可选) 打印当前序列的简单对比，确认单个序列是否正常
            if cur_veh_z and cur_infra_z:
                diff = np.mean(cur_veh_z) - np.mean(cur_infra_z)
                print(f"  [单个检查] Veh均值: {np.mean(cur_veh_z):.3f} | Infra均值: {np.mean(cur_infra_z):.3f} | 差值: {diff:.3f}")
            else:
                print(f"  [警告] {town}/{seq} 缺少数据 (Veh: {len(cur_veh_z)}, Infra: {len(cur_infra_z)})")

    # --- 3. 打印所有地图、所有序列的最终汇总结果 ---
    print("\n" + "="*60)
    print("      所有地图 & 所有序列汇总结果 (Super Global Summary)")
    print("="*60)

    print(f"目标类别: {TARGET_CLASS}")
    
    mean_veh = print_stats("Vehicle (车端基准 - 全部)", SUPER_GLOBAL_VEH_Z)
    print("-" * 40)
    mean_infra = print_stats("Infra   (路侧虚拟 - 全部)", SUPER_GLOBAL_INFRA_Z)
    
    print("="*60)
    
    # --- 4. 最终结论 ---
    if mean_veh is not None and mean_infra is not None:
        final_diff = abs(mean_veh - mean_infra)
        print(f"最终全局对齐误差 (均值差): {final_diff:.4f} m")
        
        if final_diff < 0.1:
            print(">>> 结论: 完美对齐！(误差 < 10cm)")
        elif final_diff < 0.3:
            print(">>> 结论: 对齐良好。(误差 < 30cm)")
        else:
            print(">>> 结论: 存在明显偏差，请检查 z_offset 计算逻辑或数据源。")
            
        print("\n" + "*"*60)
        print(f"👉 后续操作提示 (Where2comm):")
        print(f"请打开文件: opencood/data_utils/post_processor/voxel_postprocessor.py")
        print(f"定位到第 64 行左右，修改 Z 值基准。")
        print(f"建议修改为: cz = np.ones_like(cx) * {mean_veh:.2f}")
        print("*"*60)
    else:
        print(">>> 无法计算偏差，因为某一侧没有数据。")