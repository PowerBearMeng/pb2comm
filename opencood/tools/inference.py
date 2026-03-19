# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
# python /home/yty/mfh/code/inter/Where2comm/opencood/tools/inference.py --model_dir /home/yty/mfh/code/inter/Where2comm/opencood/logs/dair_where2comm_max_multiscale_resnet_2025_12_17_18_03_32 --fusion_method intermediate
# python /home/yty/mfh/code/inter/Where2comm/opencood/tools/inference.py --model_dir /home/yty/mfh/code/inter/Where2comm/opencood/logs/dair_where2comm_max_multiscale_resnet_2025_12_23_14_07_13

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import simple_vis
from tqdm import tqdm

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate_with_comm',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_n', type=int, default=290,
                        help='save how many numbers of visualization result?')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--eval_epoch', type=str, default=None,
                        help='Set the checkpoint')
    parser.add_argument('--comm_thre', type=float, default=None,
                        help='Communication confidence threshold')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'intermediate_with_comm', 'no']
    
    hypes = yaml_utils.load_yaml(None, opt)
    print(f'Fusion method: {hypes["model"]["core_method"]}')
    if opt.comm_thre is not None:
        hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre
    hypes['validate_dir'] = hypes['test_dir']
    # assert "test" in hypes['validate_dir']
    left_hand = True if "OPV2V" in hypes['test_dir'] else False
    print(f"Left hand visualizing: {left_hand}")

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=4,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)
    print(f'加载数据 : {hypes["fusion"]["core_method"]}')
    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    if opt.eval_epoch is not None:
        epoch_id = opt.eval_epoch
        epoch_id, model = train_utils.load_saved_model(saved_path, model, epoch_id)
    else:
        epoch_id, model = train_utils.load_saved_model(saved_path, model)
        
    model.eval()

    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0},
                   0.5: {'tp': [], 'fp': [], 'gt': 0},
                   0.7: {'tp': [], 'fp': [], 'gt': 0}}

    total_comm_rates = []
    # ================== 【新增】初始化统计列表 ==================
    time_stats = [] 
    req_bytes_stats = []
    trans_bytes_stats = []
    time_flow_stats = []
    time_blind_stats = []
    time_pb_attn_stats = []
    time_fusion_stats = []
    # ==========================================================
    # total_box = []
    for i, batch_data in tqdm(enumerate(data_loader)):
        # ================= 加上这段拦截代码 =================
        if batch_data is None:
            print(f"\n" + "!"*50)
            print(f"🚨 抓到内鬼了！第 {i} 个 Batch 读取为空（被完全裁减掉了）。")
            print("!"*50 + "\n")
            continue  # 直接跳过这个坏数据，让程序继续往下跑！
        # ===================================================
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            elif opt.fusion_method == 'no':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_no_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            
            elif opt.fusion_method == 'intermediate_with_comm':
                # pred_box_tensor, pred_score, gt_box_tensor, comm_rates = \
                #     inference_utils.inference_intermediate_fusion_withcomm(batch_data,
                #                                                   model,
                #                                                   opencood_dataset)
                # total_comm_rates.append(comm_rates)

                # # ================== 【修改】增加接收的参数 ==================
                # pred_box_tensor, pred_score, gt_box_tensor, comm_rates, time_to_req_map, req_map_bytes, transmitted_bytes = \
                #     inference_utils.inference_intermediate_fusion_withcomm(batch_data,
                #                                                   model,
                #                                                   opencood_dataset)
                # # ==========================================================
                # total_comm_rates.append(comm_rates)
                # # ================== 【新增】存入每一帧的结果 ==================
                # time_stats.append(time_to_req_map) 
                # req_bytes_stats.append(req_map_bytes)
                # trans_bytes_stats.append(transmitted_bytes)
                # # ==========================================================
                ##               PB
                # # ================== 【修改】增加接收的参数，抛弃 req 和 trans ==================
                # pred_box_tensor, pred_score, gt_box_tensor, comm_rates, time_flow, time_blind, time_pb_attn = \
                #     inference_utils.inference_intermediate_fusion_withcomm(batch_data,
                #                                                            model,
                #                                                            opencood_dataset)
                # ==========================================================
                # total_comm_rates.append(comm_rates)

                # # ================== 【新增】存入每一帧的时间结果 ==================
                # time_flow_stats.append(time_flow) 
                # time_blind_stats.append(time_blind)
                # time_pb_attn_stats.append(time_pb_attn)
                # # ==========================================================
                # 接收端

                pred_box_tensor, pred_score, gt_box_tensor, comm_rates, time_fusion = \
                    inference_utils.inference_intermediate_fusion_withcomm(batch_data,
                                                                           model,
                                                                           opencood_dataset)
    
                time_fusion_stats.append(time_fusion)
            else:
                raise NotImplementedError('Only early, late and intermediate, no, intermediate_with_comm'
                                          'fusion modes are supported.')
            if pred_box_tensor is None:
                continue

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

            if opt.save_vis_n and opt.save_vis_n >i:

                vis_save_path = os.path.join(opt.model_dir, 'vis_3d')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(opt.model_dir, 'vis_3d/3d_%05d.png' % i)
                simple_vis.visualize(pred_box_tensor,
                                    gt_box_tensor,
                                    batch_data['ego']['origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='3d',
                                    left_hand=left_hand,
                                    vis_pred_box=True)
                
                vis_save_path = os.path.join(opt.model_dir, 'vis_bev')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(opt.model_dir, 'vis_bev/bev_%05d.png' % i)
                simple_vis.visualize(pred_box_tensor,
                                    gt_box_tensor,
                                    batch_data['ego']['origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand,
                                    vis_pred_box=True)
    # print('total_box: ', sum(total_box)/len(total_box))

    if len(total_comm_rates) > 0:
        comm_rates = (sum(total_comm_rates)/len(total_comm_rates)).item()
    else:
        comm_rates = 0
    ap_30, ap_50, ap_70 = eval_utils.eval_final_results(result_stat, opt.model_dir)
    
    # =================== 【新增】计算各个模块的时间开销并输出 ===================
    # 1. 计算时间（为了避免 GPU 启动时的巨大开销影响均值，去掉前 10 帧预热数据）
    warmup = 10
    if len(time_flow_stats) > warmup:
        avg_time_flow = sum(time_flow_stats[warmup:]) / len(time_flow_stats[warmup:])
        avg_time_blind = sum(time_blind_stats[warmup:]) / len(time_blind_stats[warmup:])
        avg_time_pb_attn = sum(time_pb_attn_stats[warmup:]) / len(time_pb_attn_stats[warmup:])
    else:
        # 如果总帧数不足 10 帧，则直接计算所有帧的平均值
        avg_time_flow = sum(time_flow_stats) / len(time_flow_stats) if len(time_flow_stats) > 0 else 0
        avg_time_blind = sum(time_blind_stats) / len(time_blind_stats) if len(time_blind_stats) > 0 else 0
        avg_time_pb_attn = sum(time_pb_attn_stats) / len(time_pb_attn_stats) if len(time_pb_attn_stats) > 0 else 0
    # 算均值 (剔除前10帧)
    warmup = 10
    avg_fusion_time = sum(time_fusion_stats[warmup:]) / len(time_fusion_stats[warmup:])

    print(f"⏱️ Where2comm Fusion Time: {avg_fusion_time * 1000:.2f} ms")

    print("=" * 50)
    print(f"【各个模块时间开销统计 (Average ms per frame)】")
    print(f"⏱️ Flow Net Time:          {avg_time_flow * 1000:.2f} ms")
    print(f"⏱️ Blind Spot Calc Time:   {avg_time_blind * 1000:.2f} ms")
    print(f"⏱️ PB Attention Time:      {avg_time_pb_attn * 1000:.2f} ms")
    print("-" * 50)
    print(f"Total Measured Overhead:   {(avg_time_flow + avg_time_blind + avg_time_pb_attn) * 1000:.2f} ms")
    print("=" * 50)
    # ===============================================================

    with open(os.path.join(saved_path, 'result.txt'), 'a+') as f:
        msg = 'Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f}\n'.format(epoch_id, ap_30, ap_50, ap_70, comm_rates)
        f.write(msg)
        
        # =================== 【新增】把时间结果写入 txt 日志 ===================
        f.write("--- Latency Evaluation ---\n")
        f.write(f"Flow Net Time:        {avg_time_flow * 1000:.2f} ms\n")
        f.write(f"Blind Spot Calc Time: {avg_time_blind * 1000:.2f} ms\n")
        f.write(f"PB Attention Time:    {avg_time_pb_attn * 1000:.2f} ms\n")
        f.write(f"Total Overhead:       {(avg_time_flow + avg_time_blind + avg_time_pb_attn) * 1000:.2f} ms\n\n")
        # =================================================================

if __name__ == '__main__':
    main()

