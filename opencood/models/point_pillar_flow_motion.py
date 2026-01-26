# opencood/models/point_pillar_ffnet.py
import torch
import torch.nn as nn
from opencood.models.point_pillar_where2comm import PointPillarWhere2comm
from opencood.models.sub_modules.flow import FlowGenerator
import torch.nn.functional as F
from opencood.models.fuse_modules.where2comm_flow import Where2comm
from opencood.models.sub_modules.motion_head import MotionHead
from opencood.models.point_pillar_motion import sample_features_from_coords
class PointPillarFlowMotion(PointPillarWhere2comm):
    def __init__(self, args):
        super(PointPillarFlowMotion, self).__init__(args)
        # 初始化 FlowGenerator
        self.flow_generator = FlowGenerator(args['flow_generator_args'])
        self.fusion_net = Where2comm(args['fusion_args'])
        self.pred_len = args.get('pred_len', 5) # 预测未来多少个点
        self.motion_head = MotionHead(in_channels=256, pred_len=self.pred_len)
        
        # 记得把 pc_range 存下来，sample 特征时要用
        self.pc_range = args['lidar_range']
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
        # ... (复制你原来的 warp_feature 代码放到这里) ...
        # 仅仅为了代码整洁，你可以直接复用 where2comm 里的那个函数，或者保留你自己的
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

        if 'ffnet_t0' in data_dict.keys():
            ffnet_t0_dict = data_dict['ffnet_t0']
            ffnet_t1_dict = data_dict['ffnet_t1']
            ffnet_time = data_dict['ffnet_time']
            
            with torch.no_grad():
                combined_feat_64, combined_feat_384 = self.extract_bev_features_batch(
                    [ffnet_t0_dict, ffnet_t1_dict],
                    return_both=True
                )
                # feat_t0_64 = combined_feat_64[0::2]
                # feat_t1_64 = combined_feat_64[1::2]
                feat_t0_384, feat_t1_384 = torch.chunk(combined_feat_384, 2, dim=0)
                # feat_t0_384 = combined_feat_384[0::2]
                # feat_t1_384 = combined_feat_384[1::2]

            # A. 预测 Flow (t0 -> t1)
            flow_pred = self.flow_generator(feat_t0_384, feat_t1_384)
            
            # B. 时间外推 (t1 -> t2)
            dt_01 = ffnet_time['t_0_1'].view(-1, 1, 1, 1).to(flow_pred.device)
            dt_12 = ffnet_time['t_1_2'].view(-1, 1, 1, 1).to(flow_pred.device)
            flow_t1_to_t2 = flow_pred / (dt_01 + 1e-6) * dt_12
            
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

        # 4. 调用 Fusion (带 Flow!)
        # -----------------------------------------------------------
        if self.multi_scale:
            # 【关键】把 flow_map_final 传进去
            # 【注意】第一个参数传 spatial_features_vfe (未经过 Backbone 的)，
            # 因为 Where2comm 多尺度模式会自己跑 Backbone
            fused_feature, communication_rates, result_dict = self.fusion_net(
                spatial_features_vfe, 
                psm_single,
                record_len,
                pairwise_t_matrix, 
                self.backbone, # 传入 backbone 实例
                [self.shrink_conv, self.cls_head, self.reg_head],
                flow_map=flow_map_final # <--- 你的 Flow 在这里进入多尺度循环
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

        # 5. 最终检测头
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm, 'rm': rm}
        output_dict.update(result_dict)
        
        # 保存 Loss 数据
        output_dict['ffnet_loss_data'] = ffnet_loss_data

        # ================== 【新增 3】 轨迹预测分支 ==================
        # 逻辑完全参考 PointPillarMotion
        if 'object_bbx_center' in data_dict:
            # (B, N, 7) -> GT 中心的位置
            gt_centers = data_dict['object_bbx_center']
            # 从融合后的特征图 (fused_feature) 上采样出物体特征
            # fused_feature: [B, 256, H, W]
            obj_feats = sample_features_from_coords(
                fused_feature, 
                gt_centers[..., :2], 
                self.pc_range
            )
            traj_preds = self.motion_head(obj_feats) # (B, N, pred_len*2)
            
            # 存入 output_dict
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
            'comm_rate': communication_rates
        })
        
        return output_dict