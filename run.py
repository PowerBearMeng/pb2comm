import os
import yaml
import copy
import time

tasks = [
    # --- 组 1: Flow Motion 模型 ---
    {
        "name": "flow_no_detach",
        "script": "opencood/tools/train_flow_motion.py",
        "yaml": "opencood/hypes_yaml/carla/carla_flow_motion.yaml",
        "detach": False
    },
    {
        "name": "flow_with_detach",
        "script": "opencood/tools/train_flow_motion.py",
        "yaml": "opencood/hypes_yaml/carla/carla_flow_motion.yaml",
        "detach": True
    },
    
    # --- 组 2: 普通 Motion 模型 (Where2comm) ---
    {
        "name": "base_motion_no_detach",
        "script": "opencood/tools/train_motion.py",
        "yaml": "opencood/hypes_yaml/carla/carla_where2comm_motion.yaml",
        "detach": False
    },
    {
        "name": "base_motion_with_detach",
        "script": "opencood/tools/train_motion.py",
        "yaml": "opencood/hypes_yaml/carla/carla_where2comm_motion.yaml",
        "detach": True
    }
]
# ====================================================

def run_all():
    final_report = []
    start_time_total = time.time()

    print(f"🚀 准备开始执行 {len(tasks)} 个实验...\n")

    for i, task in enumerate(tasks):
        exp_name = task['name']
        script_path = task['script']
        yaml_path = task['yaml']
        detach_bool = task['detach']
        
        print(f"\n{'='*60}")
        print(f"进度 [{i+1}/{len(tasks)}]: 正在运行 {exp_name}")
        print(f"脚本: {script_path}")
        print(f"配置: {yaml_path}")
        print(f"设置: detach_motion = {detach_bool}")
        print(f"{'='*60}")

        # 1. 读取原始 YAML
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"❌ 找不到配置文件: {yaml_path}")
            final_report.append((exp_name, "Config Missing ❌"))
            continue

        # 2. 修改配置 (detach_motion)
        config_copy = copy.deepcopy(config)
        
        # 兼容性处理：尝试在 model->args 里找，如果找不到就假定在根目录
        if 'model' in config_copy and 'args' in config_copy['model']:
            config_copy['model']['args']['detach_motion'] = detach_bool
        else:
            config_copy['detach_motion'] = detach_bool

        # 3. 保存临时 YAML
        # 为了避免文件名冲突，加上实验名
        temp_yaml_path = f"opencood/hypes_yaml/carla/temp_{exp_name}.yaml"
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(config_copy, f)

        # 4. 指定独立的 Log 目录
        # 这一步非常重要！防止实验结果互相覆盖
        # 结果会保存在 opencood/logs/flow_no_detach 等文件夹下
        # model_dir = f"opencood/logs/{exp_name}"

        # 5. 构造运行命令
        cmd = f"python {script_path} --hypes_yaml {temp_yaml_path}"
        
        # 6. 执行
        print(f"Executing: {cmd} ...")
        exit_code = os.system(cmd)

        # 7. 记录结果
        if exit_code != 0:
            print(f"❌ 实验 {exp_name} 失败！(Exit Code: {exit_code})")
            final_report.append((exp_name, "Failed ❌"))
        else:
            print(f"✅ 实验 {exp_name} 成功完成！")
            final_report.append((exp_name, "Success ✅"))
        
        # 删除临时文件 (可选，保留方便debug)
        # if os.path.exists(temp_yaml_path):
        #     os.remove(temp_yaml_path)

    # ================= 最终总结报告 =================
    print("\n\n")
    print("╔════════════════════════════════════════╗")
    print("║           所有实验运行总结             ║")
    print("╠════════════════════════════════════════╣")
    for name, status in final_report:
        # 格式化输出，让列对齐
        print(f"║ {name.ljust(30)} : {status.ljust(8)}║")
    print("╚════════════════════════════════════════╝")
    
    duration = (time.time() - start_time_total) / 3600
    print(f"\n总耗时: {duration:.2f} 小时")

if __name__ == '__main__':
    run_all()