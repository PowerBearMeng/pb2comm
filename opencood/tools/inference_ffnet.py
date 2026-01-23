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
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches

def visualize_roadside_check(gt_boxes, infra_conf_map, flow_map, lidar_range, save_path=None):
    """
    终极自适应版：彻底解决全蓝问题
    """
    # ================= 1. 数据转换 =================
    if isinstance(gt_boxes, torch.Tensor): gt_boxes = gt_boxes.cpu().numpy()
    
    # --- 处理置信度 ---
    if isinstance(infra_conf_map, torch.Tensor):
        c_map = infra_conf_map.detach().cpu().numpy()
        if len(c_map.shape) == 3: c_map = c_map[0]
    else:
        c_map = infra_conf_map
    conf_max = c_map.max()

    # --- 处理 Flow ---
    if isinstance(flow_map, torch.Tensor):
        f_map = flow_map.detach().cpu().numpy()
        if len(f_map.shape) == 3: f_map = f_map
        elif len(f_map.shape) == 4: f_map = f_map[0]
        dx, dy = f_map[0], f_map[1]
        speed_map = np.sqrt(dx**2 + dy**2)
    else:
        speed_map = flow_map
    flow_max = speed_map.max()

    print(f"[VIS DEBUG] Conf Max: {conf_max:.4f} | Flow Max: {flow_max:.6f}")

    # ================= 2. 画图 =================
    x_min, y_min, z_min, x_max, y_max, z_max = lidar_range
    extent = [y_min, y_max, x_min, x_max] 

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # === 左图：置信度 ===
    ax1 = axes[0]
    # 动态 vmax: 哪怕只有 0.1 也要显示红色
    conf_vmax = max(conf_max, 0.1) 
    im1 = ax1.imshow(c_map, cmap='jet', origin='lower', extent=extent, vmin=0.05, vmax=conf_vmax)
    ax1.set_title(f"1. Roadside Confidence (Max={conf_max:.2f})", color='black', fontsize=15)
    draw_boxes_clean(ax1, gt_boxes, color='white')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # === 右图：Flow (关键修改) ===
    ax2 = axes[1]
    
    # 【核心修正】
    # 无论 flow_max 是多少，都把它作为 vmax (上限设为红色)
    # 但为了防止纯噪声(全是0)导致的除零错误，设置一个极小的底限 0.05
    # 这样：
    # 如果 flow_max = 0.12，那么 0.12 就是红色，0.06 就是绿色。
    # 如果 flow_max = 5.0，那么 5.0 就是红色。
    dynamic_flow_vmax = max(flow_max, 0.05)
    
    im2 = ax2.imshow(speed_map, cmap='jet', origin='lower', extent=extent, vmin=0, vmax=dynamic_flow_vmax)
    ax2.set_title(f"2. Flow Prediction (Max={flow_max:.4f})\nAuto-Scaled to Red", color='black', fontsize=15)
    draw_boxes_clean(ax2, gt_boxes, color='lime')
    
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved to {save_path}")
    plt.close()
    
def draw_boxes_clean(ax, gt_boxes, color='lime'):
    """ 画矩形框 (忽略圆点) """
    if gt_boxes is None or len(gt_boxes) == 0: return

    for box in gt_boxes:
        if box.shape[0] < 7: continue # 跳过中心点数据
            
        phy_x, phy_y = box[0], box[1]
        l, w, yaw = box[3], box[4], box[6]
        
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        corners = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
        corners_rot = np.dot(corners, R.T) + np.array([phy_x, phy_y])
        
        plot_corners = corners_rot[:, [1, 0]] 
        rect = patches.Polygon(plot_corners, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # 箭头
        head_center = (plot_corners[0] + plot_corners[1]) / 2
        ax.plot([phy_y, head_center[0]], [phy_x, head_center[1]], color='red', linewidth=1)

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_n', type=int, default=10,
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
    # total_box = []
    time_stats = []
    for i, batch_data in tqdm(enumerate(data_loader)):
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
                pred_box_tensor, pred_score, gt_box_tensor, output_dict = \
                    inference_utils.inference_intermediate_fusion_ffnet(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            elif opt.fusion_method == 'no':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_no_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            
            elif opt.fusion_method == 'intermediate_with_comm':
                pred_box_tensor, pred_score, gt_box_tensor, comm_rates = \
                    inference_utils.inference_intermediate_fusion_withcomm(batch_data,
                                                                  model,
                                                                  opencood_dataset)
                total_comm_rates.append(comm_rates)
            else:
                raise NotImplementedError('Only early, late and intermediate, no, intermediate_with_comm'
                                          'fusion modes are supported.')
            if pred_box_tensor is None:
                continue
            # ==================== 【插入可视化代码】 ====================
            if opt.save_vis_n and i < opt.save_vis_n:
            
            # 1. 尝试获取路侧置信度 (Infra PSM)
            # 注意：Key 名字取决于你的模型返回字典，通常是 'psm_single_i' 或 'psm_infra'
                target_psm = None
                if 'psm_single_i' in output_dict and output_dict['psm_single_i'] is not None:
                    print("Using Roadside PSM for visualization.")
                    target_psm = output_dict['psm_single_i']
                elif 'psm' in output_dict:
                    print("Warning: Roadside PSM not found, using Fused PSM instead.")
                    target_psm = output_dict['psm']

                if target_psm is not None:
                    # Sigmoid 转概率
                    prob_map = torch.sigmoid(target_psm)[0, 0] 

                    if 'ffnet_loss_data' in output_dict and 'flow_vis' in output_dict['ffnet_loss_data']:
                        flow_map = output_dict['ffnet_loss_data']['flow_vis']
                        gt_boxes = gt_box_tensor[0]
                        os.makedirs(os.path.join(opt.model_dir, 'vis_check'), exist_ok=True)
                        # 调用新的可视化
                        visualize_roadside_check(
                            gt_boxes=gt_boxes,
                            infra_conf_map=prob_map,
                            flow_map=flow_map,
                            lidar_range=hypes['model']['args']['lidar_range'],
                            save_path=os.path.join(opt.model_dir, f'vis_check/frame_{i:05d}.png')
                        )
        # ==========================================================
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
    # =================== 打印时间 ===================
    avg_time = None
    if len(time_stats) > 0:
        avg_time = sum(time_stats) / len(time_stats)
        print(f"Average Inference Time (Latency): {avg_time * 1000:.2f} ms")
    # ===============================================

    with open(os.path.join(saved_path, 'result.txt'), 'a+') as f:
        msg = 'Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f}\n'.format(epoch_id, ap_30, ap_50, ap_70, comm_rates)
        if opt.comm_thre is not None:
            msg = 'Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f} | comm_thre: {:.04f}\n'.format(epoch_id, ap_30, ap_50, ap_70, comm_rates, opt.comm_thre)
        f.write(msg)
        print(msg)
        if avg_time is not None:
            f.write(f"Average Inference Time (Latency): {avg_time * 1000:.2f} ms\n")


if __name__ == '__main__':
    main()
