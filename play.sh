#!/bin/bash
# 带宽推理
# python opencood/tools/mfh/band_inference_3d.py --model_dir opencood/logs/carla_pb2comm_2026_03_05_16_59_18
# python opencood/tools/mfh/band_inference_3d.py --model_dir opencood/logs/carla_where2comm_2026_03_04_21_30_13
# 推理
# python opencood/tools/inference.py --model_dir opencood/logs/carla_where2comm_2026_03_12_17_43_06  --fusion_method late  
# python opencood/tools/inference.py --model_dir opencood/logs/carla_single_late_2026_03_12_19_11_43 --fusion_method late  

# 训练
# python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/carla/carla_pb2comm.yaml --fusion_method intermediate_with_comm
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/carla/carla_where2comm.yaml --fusion_method intermediate_with_comm
# python opencood/tools/train_flow_motion.py --hypes_yaml opencood/hypes_yaml/carla/carla_single.yaml --fusion_method late
