# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
import cv2
import os
class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        
        # --- 核心算法参数 ---
        self.smooth = False
        self.thre = args['thre']
        if 'gaussian_smooth' in args:
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        self.vis_debug = args.get('visualize', False)
        self.print_debug = args.get('print_debug', False)
        if self.vis_debug:
            self.vis_interval = 1                        # 频率：每多少次保存一张图
            self.vis_save_dir = '/home/yty/mfh/code/inter/Where2comm/mfh_tool/pic'      # 路径：保存位置
            self.vis_count = 0                            # 计数器内部维护
        
        if self.vis_debug and not os.path.exists(self.vis_save_dir):
            os.makedirs(self.vis_save_dir)

    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def _print_debug_info(self, batch_idx, neighbor_idx, H, W, conf_only_mask, combined_mask):
        """只负责打印，不负责计算"""
        total_pixels = H * W
        num_conf = conf_only_mask.sum().item()
        num_final = combined_mask.sum().item()
        
        print(f"\n>>> [Batch {batch_idx} | Neighbor {neighbor_idx} (Sender)] Transmission Analysis <<<")
        print(f"1. Raw Map Size (Full):           {total_pixels} pixels")
        print(f"2. Where2comm (Object Only):      {int(num_conf)} pixels")
        print(f"3. Ours (Object + BlindSpot):     {int(num_final)} pixels")
        save_ratio = 0.0 
        if num_conf > 0:
            save_ratio = 100 * (1 - num_final / num_conf)
            print(f"   => Further Reduction Rate:     {save_ratio:.2f}% (Saved vs Where2comm)")
        else:
            print(f"   => No objects detected to send.")
        print("---------------------------------------------------------------")
        return save_ratio

    def _save_debug_image(self, ori_map, conf_mask, final_mask, T_ego2sender, b, k, save_ratio):
        """只负责画图和保存，包含核心的坐标变换逻辑"""
        self.vis_count += 1
        if self.vis_count % self.vis_interval != 0:
            return

        # 1. 准备基础数据 (转 numpy, squeeze 维度)
        # ori_map: [1, H, W] -> [H, W]
        # masks: [1, 1, H, W] -> [H, W] (需结合 ori_map 显示热力)
        H, W = ori_map.shape[-2], ori_map.shape[-1]
        
        raw_np = ori_map[0].detach().cpu().numpy()
        w2c_np = (ori_map * conf_mask)[0, 0].detach().cpu().numpy()
        ours_np = (ori_map * final_mask)[0, 0].detach().cpu().numpy()

        # 2. 【核心修复】Warp Point 策略：找 Ego 位置
        # 无论矩阵怎么变，把 Ego 中心点 Warp 过去永远是对的
        ego_point_map = torch.zeros((1, 1, H, W), device=T_ego2sender.device)
        ego_point_map[0, 0, H//2, W//2] = 1.0 # 标记 Ego 位置
        
        warped_ego_point = warp_affine_simple(
            ego_point_map, 
            T_ego2sender.unsqueeze(0), 
            (H, W)
        )
        # 找最大值位置 (即变换后的点)
        warped_ego_np = warped_ego_point[0, 0].detach().cpu().numpy()
        ego_y, ego_x = -1, -1
        if warped_ego_np.max() > 0.1: # 确保点还在图内
             ego_y, ego_x = np.unravel_index(np.argmax(warped_ego_np), warped_ego_np.shape)

        # 3. 绘图辅助函数
        def _process_img(data_np, label):
            # 转 uint8 热力图
            img_u8 = (data_np * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
            
            # 画 Ego 星星
            if 0 <= ego_x < W and 0 <= ego_y < H:
                cv2.drawMarker(heatmap, (ego_x, ego_y), (0, 255, 255), cv2.MARKER_STAR, 20, 2)
                cv2.putText(heatmap, "Ego", (ego_x + 10, ego_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 画标题
            cv2.putText(heatmap, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            return heatmap

        # 4. 生成三张图并拼接
        img_raw = _process_img(raw_np, "Raw Confidence")
        img_w2c = _process_img(w2c_np, "Where2comm")
        img_ours = _process_img(ours_np, "Ours (BlindAware)")
        
        combined_img = np.hstack((img_raw, img_w2c, img_ours))
        
        # 5. 保存
        file_name = os.path.join(self.vis_save_dir, f'step_{self.vis_count}_batch_{b}_ratio_{save_ratio:.2f}.png')
        cv2.imwrite(file_name, combined_img)
        print(f"   [Visual Saved]: {file_name}")

    def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix, blind_spot_mask=None):
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape
        
        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            ori_communication_maps = batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(1) 
            
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
            base_mask = torch.where(communication_maps > self.thre, ones_mask, zeros_mask)
            
            if blind_spot_mask is not None:
                # 1. 准备 Ego 盲区图
                ego_bs_mask = blind_spot_mask[b]
                if ego_bs_mask.shape[-1] != W:
                    ego_bs_mask = F.interpolate(ego_bs_mask.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)

                final_masks_list = []
                
                for k in range(N): 
                    # --- Ego 自身 ---
                    if k == 0:
                        final_masks_list.append(base_mask[k:k+1])
                        continue
                    
                    # --- Neighbor (Sender) ---
                    T_ego2sender = t_matrix[0, k] 
                    warped_bs_mask = warp_affine_simple(
                        ego_bs_mask.unsqueeze(0), 
                        T_ego2sender.unsqueeze(0), 
                        (H, W)
                    )
                    
                    conf_only_mask = base_mask[k:k+1] 
                    combined_mask = conf_only_mask * warped_bs_mask
                    final_masks_list.append(combined_mask)

                    # -----------------------------------------------
                    # [调用] 调试逻辑：只在开启且处理第一个邻居时调用
                    # -----------------------------------------------
                    if self.print_debug and k == 1:
                        # 打印数据
                        save_ratio = self._print_debug_info(b, k, H, W, conf_only_mask, combined_mask)
                    if self.vis_debug and k == 1:
                        self._save_debug_image(
                            ori_communication_maps[k], # 原始置信度图
                            conf_only_mask,            # W2C Mask
                            combined_mask,             # Ours Mask
                            T_ego2sender,              # 变换矩阵
                            b, k, save_ratio
                        )
                    # -----------------------------------------------
                
                communication_mask = torch.cat(final_masks_list, dim=0)
            else:
                communication_mask = base_mask

            # --- 后续标准处理保持不变 ---
            communication_rate = communication_mask[0].sum()/(H*W)
            communication_mask_nodiag = communication_mask.clone()
            ones_mask = torch.ones_like(communication_mask).to(communication_mask.device)
            communication_mask_nodiag[::2] = ones_mask[::2]

            communication_masks.append(communication_mask_nodiag)
            communication_rates.append(communication_rate)
            batch_communication_maps.append(ori_communication_maps * communication_mask_nodiag)

        communication_rates = sum(communication_rates)/B
        communication_masks = torch.concat(communication_masks, dim=0)
        return batch_communication_maps, communication_masks, communication_rates