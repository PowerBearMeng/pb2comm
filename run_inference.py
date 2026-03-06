import subprocess

if __name__ == '__main__':
    motion_infer_path = "opencood/tools/inference_motion.py"

    task_list = [
        'opencood/logs/carla_pb2comm_2026_02_27_15_40_11',
        'opencood/logs/carla_pb2comm_2026_02_27_20_57_24',
        'opencood/logs/carla_pb2comm_2026_02_28_02_15_44',
        'opencood/logs/carla_pb2comm_2026_02_28_07_33_51',
        'opencood/logs/carla_pb2comm_2026_02_28_12_51_24',
    ]

    for t in task_list:
        cmd = [
            "python",
            motion_infer_path,
            "--model_dir",
            t
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd)