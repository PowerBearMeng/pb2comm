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
        # 2. 【新增】风险阈值 (如果没有设置，默认给一个很高的值 1.0，相当于不启用)
        self.risk_threshold = args['risk_threshold']
        self.open_risk = args['risk']
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

    # def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix, blind_spot_mask=None):
    #     B, L, _, _, _ = pairwise_t_matrix.shape
    #     _, _, H, W = batch_confidence_maps[0].shape
        
    #     communication_masks = []
    #     communication_rates = []
    #     batch_communication_maps = []
        
    #     for b in range(B):
    #         N = record_len[b]
    #         t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
    #         ori_communication_maps = batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(1) 
            
    #         if self.smooth:
    #             communication_maps = self.gaussian_filter(ori_communication_maps)
    #         else:
    #             communication_maps = ori_communication_maps

    #         ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
    #         zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
    #         base_mask = torch.where(communication_maps > self.thre, ones_mask, zeros_mask)
            
    #         if blind_spot_mask is not None:
    #             # 1. 准备 Ego 盲区图
    #             ego_bs_mask = blind_spot_mask[b]
    #             if ego_bs_mask.shape[-1] != W:
    #                 ego_bs_mask = F.interpolate(ego_bs_mask.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)

    #             final_masks_list = []
                
    #             for k in range(N): 
    #                 # --- Ego 自身 ---
    #                 if k == 0:
    #                     final_masks_list.append(base_mask[k:k+1])
    #                     continue
                    
    #                 # --- Neighbor (Sender) ---
    #                 T_ego2sender = t_matrix[0, k] 
    #                 warped_bs_mask = warp_affine_simple(
    #                     ego_bs_mask.unsqueeze(0), 
    #                     T_ego2sender.unsqueeze(0), 
    #                     (H, W)
    #                 )
                    
    #                 conf_only_mask = base_mask[k:k+1] 
    #                 combined_mask = conf_only_mask * warped_bs_mask
    #                 final_masks_list.append(combined_mask)

    #                 # -----------------------------------------------
    #                 # [调用] 调试逻辑：只在开启且处理第一个邻居时调用
    #                 # -----------------------------------------------
    #                 if self.print_debug and k == 1:
    #                     # 打印数据
    #                     save_ratio = self._print_debug_info(b, k, H, W, conf_only_mask, combined_mask)
    #                 if self.vis_debug and k == 1:
    #                     self._save_debug_image(
    #                         ori_communication_maps[k], # 原始置信度图
    #                         conf_only_mask,            # W2C Mask
    #                         combined_mask,             # Ours Mask
    #                         T_ego2sender,              # 变换矩阵
    #                         b, k, save_ratio
    #                     )
    #                 # -----------------------------------------------
                
    #             communication_mask = torch.cat(final_masks_list, dim=0)
    #         else:
    #             communication_mask = base_mask

    #         # --- 后续标准处理保持不变 ---
    #         # ================= 修改开始 =================
    #         if N > 1:
    #             neighbor_mask = communication_mask[1] 
    #             communication_rate = neighbor_mask.sum() / (H * W)
    #         else:
    #             raise KeyError("单车场景下不应启用盲区通信率计算逻辑，请检查配置。")
    #         # ================= 修改结束 =================
    #         communication_mask_nodiag = communication_mask.clone()
    #         ones_mask = torch.ones_like(communication_mask).to(communication_mask.device)
    #         communication_mask_nodiag[::2] = ones_mask[::2]

    #         communication_masks.append(communication_mask_nodiag)
    #         communication_rates.append(communication_rate)
    #         batch_communication_maps.append(ori_communication_maps * communication_mask_nodiag)

    #     communication_rates = sum(communication_rates)/B
    #     # print(f"communication_rates: {communication_rates}  , batch size: {B}")
    #     communication_masks = torch.concat(communication_masks, dim=0)
    #     return batch_communication_maps, communication_masks, communication_rates
    

    # def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix, 
    #             blind_spot_mask=None, risk_map=None):
        
    #     B, L, _, _, _ = pairwise_t_matrix.shape
    #     _, _, H, W = batch_confidence_maps[0].shape
    #     # ==========================================================
    #     if risk_map is not None and isinstance(risk_map, torch.Tensor):
    #         split_risk_maps = []
    #         ptr = 0
    #         for n in record_len:
    #             n = int(n)
    #             split_risk_maps.append(risk_map[ptr : ptr + n])
    #             ptr += n
    #         risk_map = split_risk_maps
    #     # ==========================================================
    #     communication_masks = []
    #     communication_rates = []
    #     batch_communication_maps = []
        
    #     # 统计变量（用于看节省了多少）
    #     total_pixels_baseline = 0.0
    #     total_pixels_ours = 0.0

    #     for b in range(B):
    #         N = record_len[b]
            
    #         # ==========================================
    #         # 1. 准备基础数据 (处理通道数问题)
    #         # ==========================================
    #         raw_conf = batch_confidence_maps[b]
    #         # 强制压缩成 1 通道 [N, 1, H, W]
    #         if raw_conf.shape[1] > 1:
    #             conf_prob = raw_conf.sigmoid().max(dim=1)[0].unsqueeze(1)
    #         else:
    #             conf_prob = raw_conf.sigmoid()

    #         # 高斯平滑（如果有）
    #         if self.smooth:
    #             score_map = self.gaussian_filter(conf_prob)
    #         else:
    #             score_map = conf_prob

    #         # ==========================================
    #         # 2. 生成两张基础 Mask
    #         # ==========================================
    #         ones = torch.ones_like(score_map)
    #         zeros = torch.zeros_like(score_map)
            
    #         # Mask A: 置信度掩码 (看清了吗？)
    #         mask_conf = torch.where(score_map > self.thre, ones, zeros)
                
    #         # Mask B: 风险掩码 (Top-K% 精准硬截断版本)
    #         if risk_map is not None and self.open_risk:
    #             current_risk = risk_map[b]
                
    #             # 强制压缩成 1 通道
    #             if current_risk.shape[1] > 1:
    #                 current_risk = current_risk.max(dim=1)[0].unsqueeze(1)
                
    #             # 1️⃣ 只在有物体的地方选 (去掉 k，直接用 mask_conf)
    #             valid_mask = mask_conf > 0  
                
    #             risk_values = current_risk[valid_mask]  # 取出这些位置的 risk (变为 1D Tensor)
                
    #             # 2️⃣ P 取自你的实验参数 (0.1 ~ 0.9)
    #             P = float(self.risk_threshold)
    #             P = max(0.0, min(1.0, P)) # 安全限制在 0~1 之间
                
    #             # 计算出具体需要保留多少个像素
    #             K = int(P * risk_values.numel())
                
    #             mask_risk_high = torch.zeros_like(current_risk)
                
    #             if K > 0:
    #                 # 精准找出 Top K 个值和它们在 valid_mask 内部的相对索引
    #                 topk_vals, topk_idx = torch.topk(risk_values, K)
                    
    #                 # 展平以便赋值
    #                 flat_mask = mask_risk_high.view(-1)
    #                 # 找到 valid_mask 里所有为 True 的绝对索引
    #                 valid_indices = valid_mask.view(-1).nonzero().squeeze(1)
                    
    #                 # 映射回去：把这 Top K 个绝对位置设为 1
    #                 flat_mask[valid_indices[topk_idx]] = 1
                    
    #                 # 恢复原来的形状
    #                 mask_risk_high = flat_mask.view_as(current_risk)
    #             else:
    #                 # 如果 K=0 (P=0 或者 画面里没车)，全黑
    #                 mask_risk_high = torch.zeros_like(current_risk)
    #         else:
    #             mask_risk_high = ones # 没有风险图就默认全要
            
    #         # 准备盲区
    #         ego_bs_mask = None
    #         if blind_spot_mask is not None:
    #             ego_bs_mask = blind_spot_mask[b]
    #             if ego_bs_mask.shape[-1] != W:
    #                 ego_bs_mask = F.interpolate(ego_bs_mask.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)

    #         t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
    #         final_masks_list = []

    #         for k in range(N):
    #             # --- 情况 1: 自车 (Ego) ---
    #             if k == 0:
    #                 # 自车拥有豁免权！不受 Risk 限制！
    #                 # 只要置信度够高，就保留
    #                 my_mask = mask_conf[k:k+1]
    #                 final_masks_list.append(my_mask)
                    
    #                 # 统计
    #                 total_pixels_baseline += my_mask.sum().item()
    #                 total_pixels_ours += my_mask.sum().item()
    #                 continue
                
    #             # --- 情况 2: 路侧 (Infra) ---
    #             # 路侧必须同时满足：1.有车 2.危险
    #             mask_infra_baseline = mask_conf[k:k+1]              # 原本方案
    #             mask_infra_ours = mask_conf[k:k+1] * mask_risk_high[k:k+1] # 你的方案 (AND)

    #             # 叠加盲区/空间变换 (Warp)
    #             if ego_bs_mask is not None:
    #                 T_ego2sender = t_matrix[0, k] 
    #                 mask_spatial = warp_affine_simple(
    #                     ego_bs_mask.unsqueeze(0), 
    #                     T_ego2sender.unsqueeze(0), 
    #                     (H, W)
    #                 )
    #             else:
    #                 mask_spatial = torch.ones_like(mask_infra_ours)
                
    #             # 最终决定
    #             final_mask = mask_infra_ours * mask_spatial
    #             final_masks_list.append(final_mask)
                
    #             # 统计差距
    #             total_pixels_baseline += (mask_infra_baseline * mask_spatial).sum().item()
    #             total_pixels_ours += final_mask.sum().item()

    #         # 4. 堆叠回去
    #         communication_mask = torch.cat(final_masks_list, dim=0)

    #         # 5. 计算通信率
    #         if N > 1:
    #             neighbor_mask = communication_mask[1] 
    #             print("\n==== DEBUG ====")
    #             print("conf sum:", mask_conf[k].sum().item())
    #             print("risk sum:", mask_risk_high[k].sum().item())
    #             print("spatial sum:", mask_spatial.sum().item())
    #             print("final sum:", final_mask.sum().item())
    #             print("================\n")
    #             communication_rate = neighbor_mask.sum() / (H * W)
    #         else:
    #             communication_rate = torch.tensor(0.0).to(communication_mask.device)

    #         # 对角线置1 (自己跟自己通信永远是1)
    #         communication_mask_nodiag = communication_mask.clone()
    #         communication_mask_nodiag[::2] = ones[::2]

    #         communication_masks.append(communication_mask_nodiag)
    #         communication_rates.append(communication_rate)
    #         batch_communication_maps.append(conf_prob * communication_mask_nodiag)
        
    #     # 打印结果
    #     if total_pixels_baseline > 0:
    #         reduced = total_pixels_baseline - total_pixels_ours
    #         ratio = reduced / total_pixels_baseline
    #         print(f"[Stats] Baseline: {int(total_pixels_baseline)} -> Ours: {int(total_pixels_ours)} | Saved: {ratio:.2%}")

    #     communication_rates = sum(communication_rates) / B if B > 0 else 0
    #     communication_masks = torch.concat(communication_masks, dim=0)
        
    #     return batch_communication_maps, communication_masks, communication_rates

    def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix, 
                blind_spot_mask=None, risk_map=None, current_epoch=None):
        
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape
        # ==========================================================
        # 风险图切分 (防止降维报错)
        if risk_map is not None and isinstance(risk_map, torch.Tensor):
            split_risk_maps = []
            ptr = 0
            for n in record_len:
                n = int(n)
                split_risk_maps.append(risk_map[ptr : ptr + n])
                ptr += n
            risk_map = split_risk_maps
        # ==========================================================
        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        
        # 统计变量（用于看节省了多少）
        total_pixels_baseline = 0.0
        total_pixels_ours = 0.0

        for b in range(B):
            N = int(record_len[b])
            
            # ==========================================
            # 1. 准备基础数据 (处理通道数问题)
            # ==========================================
            raw_conf = batch_confidence_maps[b]
            if raw_conf.shape[1] > 1:
                conf_prob = raw_conf.sigmoid().max(dim=1)[0].unsqueeze(1)
            else:
                conf_prob = raw_conf.sigmoid()

            if self.smooth:
                score_map = self.gaussian_filter(conf_prob)
            else:
                score_map = conf_prob

            ones = torch.ones_like(score_map)
            zeros = torch.zeros_like(score_map)
            
            # Mask A: 置信度掩码
            mask_conf = torch.where(score_map > self.thre, ones, zeros)
            
            # 提前准备好当前场景的 Risk Map
            if risk_map is not None:
                current_risk_batch = risk_map[b]
                if current_risk_batch.shape[1] > 1:
                    current_risk_batch = current_risk_batch.max(dim=1)[0].unsqueeze(1)
            else:
                current_risk_batch = None
                
            # 准备盲区
            ego_bs_mask = None
            if blind_spot_mask is not None:
                ego_bs_mask = blind_spot_mask[b]
                if ego_bs_mask.shape[-1] != W:
                    ego_bs_mask = F.interpolate(ego_bs_mask.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)

            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            final_masks_list = []

            # ==========================================================
            # 3. 遍历每一个 Agent (将 Top-K 移入此循环！)
            # ==========================================================
            for k in range(N):
                # --- 情况 1: 自车 (Ego) ---
                if k == 0:
                    my_mask = mask_conf[k:k+1]
                    final_masks_list.append(my_mask)
                    total_pixels_baseline += my_mask.sum().item()
                    total_pixels_ours += my_mask.sum().item()
                    continue
                
                # --- 情况 2: 路侧 (Infra) ---
                mask_infra_baseline = mask_conf[k:k+1]

                # ==========================================
                # 【核心修复 1】：路侧专属动态排行榜 (Top-K%) + 连续松弛退火
                # ==========================================
                if current_risk_batch is not None and getattr(self, 'open_risk', True):
                    # 只取当前路侧的风险图和置信度图
                    current_risk_k = current_risk_batch[k:k+1]
                    mask_conf_k = mask_conf[k:k+1]
                    
                    valid_mask_k = mask_conf_k > 0  
                    risk_values_k = current_risk_k[valid_mask_k]  # 仅包含路侧数据！
                    
                    P = float(self.risk_threshold)
                    P = max(0.0, min(1.0, P))
                    
                    K_pixels = int(P * risk_values_k.numel())
                    
                    # 1. 初始化硬掩码和动态阈值
                    mask_risk_hard_k = torch.zeros_like(current_risk_k)
                    tau_k = 1e6 # 默认无限大阈值 (如果画面里没车，什么都不发)
                    
                    if K_pixels > 0:
                        topk_vals, topk_idx = torch.topk(risk_values_k, K_pixels)
                        
                        # 【绝妙的一步】：获取排行榜第 K 名的分数，这就是动态物理阈值 tau_k！
                        tau_k = topk_vals[-1].item() 
                        
                        # 生成测试期使用的绝对硬掩码 (Hard Mask)
                        flat_mask = mask_risk_hard_k.view(-1)
                        valid_indices = valid_mask_k.view(-1).nonzero().squeeze(1)
                        flat_mask[valid_indices[topk_idx]] = 1
                        mask_risk_hard_k = flat_mask.view_as(current_risk_k)

                    # 2. 连续松弛温度退火 (Temperature Annealing)
                    if self.training:
                        T_max = 1.0    # 初始高温
                        T_min = 0.01   # 结束低温
                        anneal_end_epoch = 25.0  # 第 25 轮结冰提前结束
                        print(f'current_epoch:{current_epoch}')
                        # 计算当前 Epoch 的温度 (指数衰减)
                        progress = min(current_epoch / anneal_end_epoch, 1.0)
                        T = T_max * ( (T_min / T_max) ** progress )
                        
                        # 使用动态阈值 tau_k 进行软化
                        soft_mask = torch.sigmoid((current_risk_k - tau_k) / T)
                        
                        # 软掩码只在置信度有效区域内生效，防止背景噪声大面积扩散
                        mask_risk_high_k = soft_mask * valid_mask_k.float()
                        
                    else:
                        # 【测试/推理阶段】：绝对零度，严格服从 Top-K% 硬掩码
                        mask_risk_high_k = mask_risk_hard_k
                        
                else:
                    mask_risk_high_k = torch.ones_like(mask_infra_baseline)
                # ==========================================

                mask_infra_ours = mask_infra_baseline * mask_risk_high_k

                # ==========================================
                # 【核心修复 2】：叠加盲区/空间变换 (1.0 - 可见图)
                # ==========================================
                if ego_bs_mask is not None:
                    T_ego2sender = t_matrix[0, k] 
                    mask_spatial = warp_affine_simple(
                        ego_bs_mask.unsqueeze(0), 
                        T_ego2sender.unsqueeze(0), 
                        (H, W)
                    )
                else:
                    mask_spatial = torch.ones_like(mask_infra_ours)
                
                # 最终决定
                final_mask = mask_infra_ours * mask_spatial
                final_masks_list.append(final_mask)
                
                total_pixels_baseline += (mask_infra_baseline * mask_spatial).sum().item()
                total_pixels_ours += final_mask.sum().item()

                # DEBUG 打印 (k == 1 时打印一次)
                if k == 1:
                    pass
                    # print("\n==== DEBUG ====")
                    # print(f"P Ratio Set: {self.risk_threshold}")
                    # print("conf sum (Infra):", mask_infra_baseline.sum().item())
                    # print("risk sum (Infra):", mask_risk_high_k.sum().item())
                    # print("spatial sum:", mask_spatial.sum().item())
                    # print("final sum:", final_mask.sum().item())
                    # print("================\n")

            # 4. 堆叠回去
            communication_mask = torch.cat(final_masks_list, dim=0)

            # 5. 计算通信率
            if N > 1:
                neighbor_mask = communication_mask[1] 
                communication_rate = neighbor_mask.sum() / (H * W)
            else:
                communication_rate = torch.tensor(0.0).to(communication_mask.device)

            communication_mask_nodiag = communication_mask.clone()
            communication_mask_nodiag[::2] = ones[::2]

            communication_masks.append(communication_mask_nodiag)
            communication_rates.append(communication_rate)
            batch_communication_maps.append(conf_prob * communication_mask_nodiag)
        
        if total_pixels_baseline > 0:
            reduced = total_pixels_baseline - total_pixels_ours
            ratio = reduced / total_pixels_baseline
            print(f"[Stats] Baseline: {int(total_pixels_baseline)} -> Ours: {int(total_pixels_ours)} | Saved: {ratio:.2%}")

        communication_rates = sum(communication_rates) / B if B > 0 else 0
        communication_masks = torch.concat(communication_masks, dim=0)
        
        return batch_communication_maps, communication_masks, communication_rates