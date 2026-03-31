#!/bin/bash
# 带宽推理
# python opencood/tools/mfh/band_inference_3d.py --model_dir opencood/logs/carla_pb2comm_2026_03_05_16_59_18
# python opencood/tools/mfh/band_inference_3d.py --model_dir opencood/logs/carla_where2comm_2026_03_04_21_30_13
# 推理
# python opencood/tools/inference.py --model_dir opencood/logs/carla_where2comm_2026_03_12_17_43_06  --fusion_method late  
# python opencood/tools/inference.py --model_dir opencood/logs/carla_single_late_2026_03_12_19_11_43 --fusion_method late  

# 训练
# python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/carla/carla_pb2comm.yaml --fusion_method intermediate_with_comm
python opencood/tools/train_flow_motion.py --hypes_yaml opencood/hypes_yaml/carla/carla_pb2comm.yaml --fusion_method intermediate_with_comm
# python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/carla/carla_single.yaml --fusion_method late

v2x-vit cyk baseline 
# carla_v2xvit_2026_03_16_15_01_04 
# python opencood/tools/inference.py --model_dir opencood/logs/carla_v2xvit_2026_03_16_15_01_04  --fusion_method intermediate
#

single： yty baseline # 
python opencood/tools/inference.py --model_dir opencood/logs/baseline_single  --fusion_method late
python opencood/tools/mfh/band_single_3d.py  --model_dir opencood/logs/baseline_single 

where2comm max cyk
# opencood/logs/carla_where2comm_2026_03_16_22_51_28
python opencood/tools/inference.py --model_dir opencood/logs/carla_where2comm_2026_03_16_22_51_28  --fusion_method intermediate_with_comm
# python opencood/tools/inference.py --model_dir opencood/logs/dair_v2vnet_2026_03_17_02_58_59 --fusion_method late 

where2comm atten baseline cyk # 
python opencood/tools/inference.py --model_dir opencood/logs/carla_where2comm_2026_03_16_10_54_45 --fusion_method intermediate_with_comm
0.7973 0.6888 0.0532


# pb2comm: PB cyk 
# opencood/logs/carla_pb2comm_2026_03_15_17_40_16
# python opencood/tools/inference.py --model_dir opencood/logs/carla_pb2comm_2026_03_15_17_40_16 --fusion_method intermediate_with_comm
# 0.7704 0.6865 0.001135

# pb2comm PB yty 
# opencood/logs/carla_pb2comm_2026_03_15_11_28_11
# python opencood/tools/inference.py --model_dir opencood/logs/carla_pb2comm_2026_03_15_11_28_11  --fusion_method intermediate_with_comm


pb2comm: atten yty best  # 
opencood/logs/b_pb2comm_atten
python opencood/tools/inference.py --model_dir opencood/logs/b_pb2comm_atten --fusion_method intermediate_with_comm
python opencood/tools/mfh/band_inference_3d.py  --model_dir opencood/logs/b_pb2comm_atten --trace_csv /home/yty/mfh/code/inter/Where2comm/4G.xlsx
Epoch: 21 | AP @0.3: 0.8139 | AP @0.5: 0.8010 | AP @0.7: 0.7197 | comm_rate: 0.001683
Epoch: 25 | AP @0.3: 0.8273 | AP @0.5: 0.8111 | AP @0.7: 0.7190 | comm_rate: 0.001133

python opencood/tools/train_dair.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_pb2comm.yaml

Town12_t_1_seq24_000016