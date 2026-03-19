# opencood/models/point_pillar_ffnet.py
import torch
import torch.nn as nn
from opencood.models.point_pillar_where2comm import PointPillarWhere2comm
from opencood.models.sub_modules.flow import FlowGenerator
import torch.nn.functional as F
from opencood.models.sub_modules.motion_head import MotionHead
from opencood.models.point_pillar_motion import sample_features_from_coords
from opencood.utils.blind_spot_utils import get_blind_spot_mask
from opencood.models.fuse_modules.where2comm_flow_blind import Where2comm
import torch
import time

class PointPillarPb2comm(PointPillarWhere2comm):
    def __init__(self, args):
        super(PointPillarPb2comm, self).__init__(args)
        # 初始化 FlowGenerator
        self.flow_generator = FlowGenerator(args['flow_generator_args'])
        self.fusion_net = Where2comm(args['fusion_args'])
        # 记得把 pc_range 存下来，sample 特征时要用
        self.pc_range = args['lidar_range']
        self.detach_motion = args['detach_motion']
        self.voxel_size = args['voxel_size']
        self.flow_embedding_dim = 32  
        self.use_motion = args['use_motion']
        self.flow_encoder = nn.Sequential(
            # 2 -> 32, 使用 3x3 卷积感知一点局部信息
            nn.Conv2d(2, self.flow_embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.flow_embedding_dim),
            nn.ReLU(inplace=True),
            # 可选：再加一层 1x1 整合一下，单层其实也够用
            nn.Conv2d(self.flow_embedding_dim, self.flow_embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.flow_embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.pred_len = args.get('pred_len', 5) # 预测未来多少个点
        motion_dim = 256
        self.motion_head = MotionHead(in_channels=motion_dim+self.flow_embedding_dim , pred_len=self.pred_len)
        
        # ===============================================================
        if args['backbone_fix']:
            self.backbone_fix()
            print("冻结 Backbone 参数，用于 Finetune FlowNet")
    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
        # 把 fusion也给冻结了
        if self.fusion_net:
            for p in self.fusion_net.parameters():
                p.requires_grad = False

    def warp_feature(self, x, flow):
        """
        Args:
            x: [B, C, H, W] (要被 warp 的特征图，即 t1 时刻特征)
            flow: [B, 2, H, W] (预测出的位移量，单位：像素)
        """
        B, C, H, W = x.size()
        
        # 1. 生成基础网格 (0, 1, 2, ... W-1)
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        
        grid = torch.cat((xx, yy), 1).float().to(x.device) # [B, 2, H, W]

        # 2. 加上预测的 flow (新的坐标 = 旧坐标 + 位移)
        vgrid = grid + flow

        # 3. 归一化到 [-1, 1] 区间供 grid_sample 使用
        # 公式: 2 * x / (W-1) - 1
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        # [B, 2, H, W] -> [B, H, W, 2]
        vgrid = vgrid.permute(0, 2, 3, 1)

        # 4. 采样
        # padding_mode='zeros' 意味着移出边界的部分用 0 填充
        # align_corners=True 对应上面的归一化公式
        output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return output

    def extract_bev_features_batch(self, data_dict_list, return_both=True):
        """
        批量提取多个数据的 BEV 特征（并行加速）
        Args: 
            data_dict_list:  List of voxel data dicts
            return_both: 是否同时返回 64 维和 384 维特征
        
        Returns: 
            如果 return_both=True:  (feat_64, feat_384)
            否则:  feat_64
        """
        all_voxel_features = []
        all_voxel_coords = []
        all_voxel_num_points = []

        batch_offset = 0
        for data_dict in data_dict_list:
            all_voxel_features.append(data_dict['voxel_features'])
            all_voxel_num_points.append(data_dict['voxel_num_points'])
            
            # 修正 batch index
            coords = data_dict['voxel_coords'].clone()
            coords[: , 0] += batch_offset
            all_voxel_coords.append(coords)
            
            batch_offset += data_dict['voxel_coords'][:, 0].max().item() + 1

        # 拼接
        combined_dict = {
            'voxel_features': torch.cat(all_voxel_features, dim=0),
            'voxel_coords': torch.cat(all_voxel_coords, dim=0),
            'voxel_num_points':  torch.cat(all_voxel_num_points, dim=0)
        }

        # 前向传播
        batch_dict = self.pillar_vfe(combined_dict)
        batch_dict = self.scatter(batch_dict)
        feat_64 = batch_dict['spatial_features']
        
        if return_both:
            batch_dict = self.backbone(batch_dict)
            feat_384 = batch_dict['spatial_features_2d']
            return feat_64, feat_384
        else:
            return feat_64    
    
    def warp_feature_for_loss(self, x, flow):
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(x.device)
        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return output

    def calculate_relative_risk_map_batch(self, flow_map, ego_xy, H, W):
        """
        批量计算以 ego_xy 为中心的风险图 (返回 Raw Value，不归一化)
        """
        B, _, _, _ = flow_map.shape
        device = flow_map.device
        
        # 1. 生成网格
        x_min, y_min, x_max, y_max = self.pc_range[0], self.pc_range[1], self.pc_range[3], self.pc_range[4]
        ys, xs = torch.meshgrid(
            torch.linspace(y_min, y_max, H, device=device),
            torch.linspace(x_min, x_max, W, device=device)
        )
        grid_pos = torch.stack([xs, ys], dim=0).unsqueeze(0).repeat(B, 1, 1, 1) 
        
        # 2. 计算相对位置
        ego_xy_map = ego_xy.view(B, 2, 1, 1).expand(-1, -1, H, W)
        rel_pos = grid_pos - ego_xy_map
        
        # 3. 距离 (分母加小量)
        dist = torch.norm(rel_pos, dim=1, keepdim=True) + 1e-6
        
        # 4. 径向速度
        dot = torch.sum(rel_pos * flow_map, dim=1, keepdim=True)
        radial_v = - dot / dist
        
        # 5. 原始风险值 (Raw Risk)
        # 这里不要归一化！让它保留 1/dist 的物理特性
        raw_risk = (1.0 / dist) * (1.0 + 5.0 * F.relu(radial_v))
        
        return raw_risk

    def forward(self, data_dict):
        ffnet_loss_data = {}
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        
        # 保存这个未经 Backbone 的特征，传给 Where2comm 用
        spatial_features_vfe = batch_dict['spatial_features'] 
        
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']
        # -----------------------------------------------------------
        # 2. 计算 Flow (用于多尺度融合 + Loss)
        # -----------------------------------------------------------
        flow_map_final = None # 这个是要传给 Fusion 的
        # ===========================================================
        # ⏱️ 测速 1: Flow Net 时间
        # ===========================================================

        if 'ffnet_t0' in data_dict.keys():
            ffnet_t0_dict = data_dict['ffnet_t0']
            ffnet_t1_dict = data_dict['ffnet_t1']
            ffnet_time = data_dict['ffnet_time']
            
            with torch.no_grad():
                combined_feat_64, combined_feat_384 = self.extract_bev_features_batch(
                    [ffnet_t0_dict, ffnet_t1_dict],
                    return_both=True
                )
                feat_t0_384, feat_t1_384 = torch.chunk(combined_feat_384, 2, dim=0)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_flow_start = time.time()
            # A. 预测 Flow (t0 -> t1)
            flow_pred = self.flow_generator(feat_t0_384, feat_t1_384)
            
            # B. 时间外推 (t1 -> t2)
            dt_01 = ffnet_time['t_0_1'].view(-1, 1, 1, 1).to(flow_pred.device)
            dt_12 = ffnet_time['t_1_2'].view(-1, 1, 1, 1).to(flow_pred.device)
            flow_t1_to_t2 = flow_pred / (dt_01 + 1e-6) * dt_12
            if torch.cuda.is_available(): torch.cuda.synchronize()
            time_flow = time.time() - t_flow_start
            # C. 准备传给 Fusion 的 Flow Map
            flow_map_final = flow_t1_to_t2

            # D. 计算 Loss 相关数据
            feat_pred_t2 = self.warp_feature_for_loss(feat_t1_384, flow_t1_to_t2)
            
            # 真实特征 (GT)
            feat_gt_t2 = spatial_features_2d[1::2].clone().detach()

            # 存入 Loss 字典
            ffnet_loss_data['flow_pred'] = feat_pred_t2
            ffnet_loss_data['flow_gt'] = feat_gt_t2
            ffnet_loss_data['flow_vis'] = flow_t1_to_t2

        # 3. 后处理 (压缩/DCN)
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)

        # 单车检测头 (用于生成通信 Mask)
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)
        # ================== 【修改】 生成 Risk Map 并传递 ==================
        risk_map_full = None 
        if flow_map_final is not None and 'pairwise_t_matrix' in data_dict:
            # 1. 投影计算
            t_infra_from_ego = pairwise_t_matrix[:, 0, 1].float() 
            B_scenes = t_infra_from_ego.shape[0]
            ego_origin = torch.tensor([0., 0., 0., 1.], device=t_infra_from_ego.device).view(1, 4).repeat(B_scenes, 1).float()
            ego_in_infra = torch.matmul(t_infra_from_ego, ego_origin.unsqueeze(-1)).squeeze(-1)
            ego_xy = ego_in_infra[:, :2]

            H_flow, W_flow = flow_map_final.shape[2], flow_map_final.shape[3]
            
            # ★ 1. 获取 原始风险值 (Raw Risk)
            raw_risk_map = self.calculate_relative_risk_map_batch(flow_map_final, ego_xy, H_flow, W_flow)
            
            # ★ 2. 上采样对齐 (Raw Risk)
            target_H, target_W = psm_single.shape[2], psm_single.shape[3]
            if H_flow != target_H:
                raw_risk_map = F.interpolate(raw_risk_map, size=(target_H, target_W), mode='bilinear')
            
            # ★ 3. 准备置信度掩码 (Object Mask)
            if psm_single.shape[1] > 1:
                conf_map = psm_single.sigmoid().max(dim=1, keepdim=True)[0]
            else:
                conf_map = psm_single.sigmoid()
                
            # ==========================================================
            # 【必须要加的修复代码】：提取路侧专属置信度！
            # 否则你是在用 Ego 的视野过滤 Infra 的风险图！
            # ==========================================================
            infra_conf_list = []
            ptr = 0
            for b_idx, n_agents in enumerate(record_len):
                n = int(n_agents)
                if n >= 2:
                    # 取出属于当前场景中，路侧 (Index 1) 的置信度
                    infra_conf_list.append(conf_map[ptr + 1]) 
                else:
                    infra_conf_list.append(torch.zeros_like(conf_map[ptr]))
                ptr += n
            
            infra_conf_map = torch.stack(infra_conf_list, dim=0)
            
            # 用路侧自己的置信度去生成 object_mask
            object_mask = (infra_conf_map > 0.01).float()
            
            # ★ 4. 掩码过滤 (Masking)
            # 现在维度完美对齐，且逻辑正确
            masked_risk_map = raw_risk_map * object_mask
            # ==========================================================
            # ★ 4. 掩码过滤 (Masking)
            # 背景区域 (无车区域) 的风险值直接变为 0
            masked_risk_map = raw_risk_map * object_mask
            
            # ★ 5. 局部归一化 (Local Normalization)
            # 现在，最大值就是“最危险的车的风险值”，而不是“路面”
            # 注意：每个 batch 单独归一化
            B_psm = masked_risk_map.shape[0]
            risk_min = masked_risk_map.flatten(2).min(dim=2)[0].view(B_psm, 1, 1, 1)
            risk_max = masked_risk_map.flatten(2).max(dim=2)[0].view(B_psm, 1, 1, 1)
            
            # 归一化公式
            risk_map_norm = (masked_risk_map - risk_min) / (risk_max - risk_min + 1e-6)
            
            # ★ 6. 填入 Full Map (初始化全0)
            N_all, _, H_psm, W_psm = psm_single.shape
            
            # 必须和 psm_single 同设备、同类型
            risk_map_full = torch.zeros((N_all, 1, H_psm, W_psm), 
                                      dtype=psm_single.dtype, 
                                      device=psm_single.device)
            
            # ==========================================================
            # 【核心修复】精确填空 (只负责对齐数据，掩码过滤交给 Fusion Net)
            # ==========================================================
            current_feature_ptr = 0
            
            for b, n_agents in enumerate(record_len):
                n = int(n_agents)
                
                if n >= 2:
                    infra_global_idx = current_feature_ptr + 1
                    # 直接把当前场景的 risk_map_norm 填进去，不做任何阈值截断！
                    risk_map_full[infra_global_idx] = risk_map_norm[b]
                
                current_feature_ptr += n
            # ==========================================================

        # =======================================================
        # 【修改后逻辑】: 使用路侧点云 origin_lidar_i 进行猜测
        # =======================================================
        blind_spot_mask = None
        
        # 修改判断条件，优先查找 origin_lidar_i
        target_lidar_key = 'origin_lidar_i' 
        if target_lidar_key not in data_dict and 'origin_lidar' in data_dict:
             # 如果没有独立的路侧点云，回退到融合点云 (仅用于调试或可视化开启时)
            raise TypeError("Using 'origin_lidar' for blind spot mask calculation. This may be incorrect if 'origin_lidar_i' is available.")
        # ===========================================================
        # ⏱️ 测速 2: 盲区检测时间
        # ===========================================================
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_blind_start = time.time()
        if target_lidar_key in data_dict:
            _, _, H, W = spatial_features_2d.shape
            real_batch_size = len(record_len)
            gt_range = self.pc_range   
            # 【关键修改】：这里获取的是路侧点云
            batch_origin_lidar = data_dict[target_lidar_key] 
            
            mask_list = []
            
            for b in range(real_batch_size):
                lidar_tensor = batch_origin_lidar[b]
                if isinstance(lidar_tensor, torch.Tensor):
                    lidar_np = lidar_tensor.cpu().numpy()
                    # print(lidar_np.shape)
                else:
                    lidar_np = lidar_tensor

                # 调用盲区计算函数
                # 注意：因为 Dataset 里已经把路侧点云投影到了 Ego 坐标系 (T_ego_infra * P_infra)
                # 所以这里 ego_pose 依然设为 (0,0,0)，代表“以车为原点”
                mask_np = get_blind_spot_mask(
                    lidar_np, 
                    ego_pose=(0,0,0), 
                    lidar_range=gt_range, 
                    target_feat_shape=(H, W),
                    voxel_size=self.voxel_size
                )
                
                mask_tensor = torch.from_numpy(mask_np).to(spatial_features_2d.device).float()
                mask_list.append(mask_tensor)

            blind_spot_mask = torch.stack(mask_list, dim=0).unsqueeze(1)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        time_blind = time.time() - t_blind_start
        print("time_blind:"f'{time_blind}')
        # 4. 调用 Fusion (带 Flow!)
        # -----------------------------------------------------------
        # ===========================================================
        # ⏱️ 测速 3: PB Attn (Fusion) 时间
        # ===========================================================
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_attn_start = time.time()
        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                spatial_features_vfe, 
                psm_single,
                record_len,
                pairwise_t_matrix, 
                self.backbone, # 传入 backbone 实例
                [self.shrink_conv, self.cls_head, self.reg_head],
                flow_map=flow_map_final, # <--- 你的 Flow 在这里进入多尺度循环
                blind_spot_mask=blind_spot_mask,
                risk_map=risk_map_full,
                current_epoch=data_dict.get('epoch', 0)
            )
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            # 单尺度备用逻辑
            fused_feature, communication_rates, result_dict = self.fusion_net(
                spatial_features_2d,
                psm_single,
                record_len,
                pairwise_t_matrix
            )
        # -----------------------------------------------------------
        if torch.cuda.is_available(): torch.cuda.synchronize()
        time_pb_attn = time.time() - t_attn_start
        # 5. 最终检测头
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm, 'rm': rm}
        output_dict.update(result_dict)
        
        # 保存 Loss 数据
        output_dict['ffnet_loss_data'] = ffnet_loss_data

        # ================== 【修改 3】 轨迹预测分支 ==================
        if 'object_bbx_center' in data_dict and self.use_motion:
            gt_centers = data_dict['object_bbx_center']
            
            # 【核心修改】根据开关决定是否截断梯度
            if self.detach_motion:
                base_feature = fused_feature.detach() # 截断！保护 Detection
                # print("截断 Motion 分支的梯度")
            else:
                base_feature = fused_feature # 不截断！Motion 会改变 Backbone
                # print("不截断 Motion 分支的梯度")
            # 获取目标尺寸
            H, W = base_feature.shape[2], base_feature.shape[3]
            
            # B. 处理 Flow 并编码
            if flow_map_final is not None:
                # 这里验证了尺寸其实是一样的！
                # print("使用 Flow 进行轨迹预测")
                # print(f"flow_map_final shape: {flow_map_final.shape}, target size: {(H, W)}")
                # 1. 插值对齐尺寸 (因为 flow_map_final 可能是基于 384 特征算的低分辨率图)
                flow_aligned = F.interpolate(
                    flow_map_final, 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=True
                )
                
                # 2. 编码: [B, 2, H, W] -> [B, 32, H, W]
                # 这一步把“物理数值”变成了“语义特征”
                flow_embedding = self.flow_encoder(flow_aligned)
                
                # 3. 拼接: [B, 256, H, W] + [B, 32, H, W] -> [B, 288, H, W]
                features_for_motion = torch.cat([base_feature, flow_embedding], dim=1)
                
            else:
                # 容错：如果没有 Flow (比如单帧测试)，拼一个全 0 的 Tensor
                pass
                dummy_flow = torch.zeros(
                    base_feature.shape[0], 
                    self.flow_embedding_dim, 
                    H, W
                ).to(base_feature.device)
                features_for_motion = torch.cat([base_feature, dummy_flow], dim=1)

            # C. 采样
            # 现在的 features_for_motion 包含了“位置”和“速度”两方面信息
            obj_feats = sample_features_from_coords(
                features_for_motion, 
                gt_centers[..., :2], 
                self.pc_range
            )
            
            # D. 预测
            traj_preds = self.motion_head(obj_feats)
            output_dict['traj_preds'] = traj_preds
        # ==========================================================
        # 保存单车结果 (保持不变)
        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        output_dict.update({
            'psm_single_v': torch.cat([batch[0:1] for batch in split_psm_single], dim=0),
            'psm_single_i': torch.cat([batch[1:2] for batch in split_psm_single], dim=0),
            'rm_single_v': torch.cat([batch[0:1] for batch in split_rm_single], dim=0),
            'rm_single_i': torch.cat([batch[1:2] for batch in split_rm_single], dim=0),
            'comm_rate': communication_rates,
            'time_flow': time_flow,
            'time_blind': time_blind,
            'time_pb_attn': time_pb_attn
        })
        
        return output_dict