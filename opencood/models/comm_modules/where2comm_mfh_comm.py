# -*- coding: utf-8 -*-
# Author: Yue Hu <phyllis1sjtu@outlook.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        
        self.smooth = False
        self.thre = args['thre']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix, blind_spot_mask=None):
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape
        
        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        for b in range(B):
            N = record_len[b]
            # t_matrix[i, j] 表示从 i 变到 j
            # 我们需要从 Ego(0) 变到 Neighbor(k)
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            # 原始置信度图 (H, W)
            ori_communication_maps = batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(1) 
            
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            # 1. 基础逻辑：置信度 > 阈值 (有物体)
            ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
            # 这里的 mask 还是 Sender 视角的
            base_mask = torch.where(communication_maps > self.thre, ones_mask, zeros_mask)
            
            # =======================================================
            # 【核心修改 START】: 在发送端应用盲区 Mask
            # =======================================================
            if blind_spot_mask is not None:
                # blind_spot_mask shape: [B, 1, H, W] (Ego Frame)
                # 取出当前 batch 的 ego mask
                ego_bs_mask = blind_spot_mask[b] # [1, H, W]
                
                # 调整尺寸以防万一
                if ego_bs_mask.shape[-1] != W:
                    ego_bs_mask = F.interpolate(ego_bs_mask.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)

                # 我们需要为每一个 Agent 生成它对应的 Mask
                # Agent 0 是 Ego，Agent 1..N 是邻居
                final_masks_list = []
                
                for k in range(N): # 遍历当前场景下的所有车
                    if k == 0:
                        # Ego 自己：不需要过滤盲区
                        final_masks_list.append(base_mask[k:k+1])
                    else:
                        # Neighbor (Sender)：
                        # 变换矩阵: T_ego_to_neighbor
                        T_ego2sender = t_matrix[0, k] 
                        
                        # 执行 Warp (逆变换): 把 Ego 的盲区 Mask 变到 Sender 坐标系
                        warped_bs_mask = warp_affine_simple(
                            ego_bs_mask.unsqueeze(0), 
                            T_ego2sender.unsqueeze(0), 
                            (H, W)
                        )
                        

                        conf_only_mask = base_mask[k:k+1]
                        
                        # 【策略 3】你的策略: 置信度 AND 盲区
                        combined_mask = conf_only_mask * warped_bs_mask
                        final_masks_list.append(combined_mask)

                        # # =======================================================
                        # # 【新增打印逻辑】: 输出三者对比
                        # # =======================================================
                        # # 为了防止刷屏，我们只打印第一个邻居 (k==1) 的情况
                        # if k == 1:
                        #     # 1. 全量 (Total)
                        #     total_pixels = H * W
                            
                        #     # 2. 原版 Where2comm (Confidence Only)
                        #     num_conf = conf_only_mask.sum().item()
                            
                        #     # 3. 你的策略 (Confidence + BlindSpot)
                        #     num_final = combined_mask.sum().item()
                            
                        #     # print(f"\n>>> [Batch {b} | Neighbor {k} (Sender)] Transmission Analysis <<<")
                        #     # print(f"1. Raw Map Size (Full):           {total_pixels} pixels")
                        #     # print(f"2. Where2comm (Object Only):      {int(num_conf)} pixels")
                        #     # print(f"3. Ours (Object + BlindSpot):     {int(num_final)} pixels")
                            
                        #     # if num_conf > 0:
                        #     #     save_ratio = 100 * (1 - num_final / num_conf)
                        #     #     print(f"   => Further Reduction Rate:     {save_ratio:.2f}% (Saved vs Where2comm)")
                        #     # else:
                        #     #     print(f"   => No objects detected to send.")
                        #     # print("---------------------------------------------------------------")
                        # # =======================================================
                
                # 重新堆叠回 [N, 1, H, W]
                communication_mask = torch.cat(final_masks_list, dim=0)
                
            else:
                # 如果没有盲区 mask，就只用置信度 mask
                communication_mask = base_mask
            # =======================================================

            communication_rate = communication_mask[0].sum()/(H*W)
            
            # 这里的后续处理保持不变 (去对角线等)
            communication_mask_nodiag = communication_mask.clone()
            ones_mask = torch.ones_like(communication_mask).to(communication_mask.device)
            communication_mask_nodiag[::2] = ones_mask[::2] # 保持 Ego 自身数据

            communication_masks.append(communication_mask_nodiag)
            communication_rates.append(communication_rate)
            batch_communication_maps.append(ori_communication_maps * communication_mask_nodiag)

        communication_rates = sum(communication_rates)/B
        communication_masks = torch.concat(communication_masks, dim=0)
        return batch_communication_maps, communication_masks, communication_rates