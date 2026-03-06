import os
import yaml
import copy
import time
import numpy as np

# ================= 配置区域 =================
# 1. 基础配置文件路径
BASE_YAML = "opencood/hypes_yaml/carla/carla_pb2comm.yaml"
# 2. 训练脚本路径
TRAIN_SCRIPT = "opencood/tools/train_flow_motion.py"

# 3. 自动生成 0.1 到 0.9 的阈值列表
# np.arange(0.1, 1.0, 0.1) 会生成 [0.1, 0.2, ..., 0.9]
# 使用 round(x, 1) 是为了防止出现 0.3000000004 这种情况
threshold_list = [round(x, 1) for x in np.arange(0.4, 0.6, 0.1)]

tasks = []
for t in threshold_list:
    tasks.append({
        "name": f"flow_risk_{t}",  # 实验名称，例如 flow_risk_0.1
        "risk_threshold": t         # 当前任务的阈值
    })

# ===========================================

def run_all():
    final_report = []
    start_time_total = time.time()

    print(f"🚀 准备开始执行 {len(tasks)} 个实验...")
    print(f"📋 阈值列表: {threshold_list}\n")

    for i, task in enumerate(tasks):
        exp_name = task['name']
        risk_val = task['risk_threshold']
        
        print(f"\n{'='*60}")
        print(f"进度 [{i+1}/{len(tasks)}]: 正在运行 {exp_name}")
        print(f"🎯 目标 Risk Threshold: {risk_val}")
        print(f"{'='*60}")

        # 1. 读取原始 YAML
        try:
            with open(BASE_YAML, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"❌ 找不到基础配置文件: {BASE_YAML}")
            return

        # 2. 修改配置
        config_copy = copy.deepcopy(config)
        
        # 2.1 强制关闭 detach_motion
        # 假设 yaml 结构是 model -> args -> detach_motion
        if 'model' in config_copy and 'args' in config_copy['model']:
            config_copy['model']['args']['detach_motion'] = False
            
            # 2.2 【核心】修改 risk_threshold
            # 路径通常是: model -> args -> fusion_args -> communication -> risk_threshold
            # 我们需要由外向里一层层检查，确保路径存在
            model_args = config_copy['model']['args']
            
            if 'fusion_args' in model_args and 'communication' in model_args['fusion_args']:
                model_args['fusion_args']['communication']['risk_threshold'] = risk_val
                print(f"✅ 已修改配置文件: risk_threshold -> {risk_val}")
            else:
                print("❌ 警告: 在 YAML 中找不到 fusion_args.communication 配置项！")
                print("   请检查你的 yaml 文件结构是否正确。")
                continue
        else:
            print("❌ 警告: YAML 结构不符合预期 (缺少 model.args)")
            continue

        # 3. 保存临时 YAML
        # 文件名如: opencood/hypes_yaml/carla/temp_flow_risk_0.1.yaml
        temp_yaml_path = f"opencood/hypes_yaml/carla/temp_{exp_name}.yaml"
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(config_copy, f)

        # 4. 指定 Log 文件夹 (可选但推荐)
        # 加上 --model_dir 可以强制指定保存路径，方便你以后找
        # 比如 saved_models/flow_risk_0.1
        # save_dir = f"opencood/logs/{exp_name}"
        
        # 5. 构造运行命令
        # 注意：这里我加了 --model_dir 参数，如果你想让它自动生成时间戳目录，可以去掉这个参数
        cmd = f"python {TRAIN_SCRIPT} --hypes_yaml {temp_yaml_path}"
        
        # 6. 执行
        print(f"Executing: {cmd} ...")
        
        # 为了防止显存没释放，有时候可以在这里加个 sleep
        # time.sleep(5) 
        
        exit_code = os.system(cmd)

        # 7. 记录结果
        if exit_code != 0:
            print(f"❌ 实验 {exp_name} 失败！(Exit Code: {exit_code})")
            final_report.append((exp_name, "Failed ❌"))
        else:
            print(f"✅ 实验 {exp_name} 成功完成！")
            final_report.append((exp_name, "Success ✅"))
        
        # 删除临时文件
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

    # ================= 最终总结报告 =================
    print("\n\n")
    print("╔════════════════════════════════════════╗")
    print("║          所有实验运行总结              ║")
    print("╠════════════════════════════════════════╣")
    for name, status in final_report:
        print(f"║ {name.ljust(30)} : {status.ljust(8)}║")
    print("╚════════════════════════════════════════╝")
    
    duration = (time.time() - start_time_total) / 3600
    print(f"\n总耗时: {duration:.2f} 小时")

if __name__ == '__main__':
    run_all()