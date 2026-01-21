# opencood/models/point_pillar_ffnet.py
import torch
import torch.nn as nn
from opencood.models.point_pillar_where2comm import PointPillarWhere2comm
from opencood.models.sub_modules.flow import FlowGenerator
import torch.nn.functional as F
class PointPillarFlow(PointPillarWhere2comm):
    def __init__(self, args):
        super(PointPillarFlow, self).__init__(args)
        # 初始化 FlowGenerator
        self.flow_generator = FlowGenerator(args['flow_generator_args'])
        if args['backbone_fix']:
            self.backbone_fix()
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
        batch_dict = self.backbone(batch_dict)
        
        # 获取当前帧特征 [2B, 384, H, W]
        spatial_features_2d = batch_dict['spatial_features_2d']

        if 'ffnet_t0' in data_dict.keys():
            # 确保数据是成对的 (Veh, Infra)
            if not torch.all(record_len == 2):
                 raise ValueError("FFNet requires fixed [Vehicle, Infrastructure] pairs.")

            ffnet_t0_dict = data_dict['ffnet_t0']
            ffnet_t1_dict = data_dict['ffnet_t1']
            ffnet_time = data_dict['ffnet_time']
            # 并行提取特征
            with torch.no_grad():
                combined_feat_64, combined_feat_384 = self.extract_bev_features_batch(
                    [ffnet_t0_dict, ffnet_t1_dict],
                    return_both=True
                )
                
                feat_t0_64 = combined_feat_64[0::2]
                feat_t1_64 = combined_feat_64[1::2]
                feat_t1_384 = combined_feat_384[1::2]

            # FlowNet 预测
            flow_pred = self.flow_generator(feat_t0_64, feat_t1_64)
            # 时间缩放
            dt_01 = ffnet_time['t_0_1'].view(-1, 1, 1, 1).to(flow_pred.device)
            dt_12 = ffnet_time['t_1_2'].view(-1, 1, 1, 1).to(flow_pred.device)
            
            # 预测 t2
            flow_t1_to_t2 = flow_pred / (dt_01 + 1e-6) * dt_12
            feat_pred_t2 = self.warp_feature(feat_t1_384, flow_t1_to_t2) # <-- 新代码
            feat_gt_t2 = spatial_features_2d[1:: 2].clone().detach()  # [B, 384, H, W]
            # 【替换】将预测的路侧特征填入 spatial_features_2d
            spatial_features_2d[1::2] = feat_pred_t2
            # 将预测值和真实值存入字典，传给 train.py
            ffnet_loss_data['flow_pred'] = feat_pred_t2
            ffnet_loss_data['flow_gt'] = feat_gt_t2
        else:
            print("Warning: FFNet temporal data not found in input. Skipping FFNet prediction.")
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
            
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
            
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)

        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                spatial_features_2d,
                psm_single,
                record_len,
                pairwise_t_matrix, 
                self.backbone,
                [self.shrink_conv, self.cls_head, self.reg_head]
            )
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                spatial_features_2d,
                psm_single,
                record_len,
                pairwise_t_matrix
            )

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm, 'rm': rm}
        output_dict.update(result_dict)
        
        # 整理单车结果
        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        
        psm_single_v = torch.cat([batch[0:1] for batch in split_psm_single], dim=0)
        psm_single_i = torch.cat([batch[1:2] for batch in split_psm_single], dim=0)
        rm_single_v = torch.cat([batch[0:1] for batch in split_rm_single], dim=0)
        rm_single_i = torch.cat([batch[1:2] for batch in split_rm_single], dim=0)

        output_dict.update({
            'psm_single_v': psm_single_v,
            'psm_single_i': psm_single_i,
            'rm_single_v': rm_single_v,
            'rm_single_i': rm_single_i,
            'comm_rate': communication_rates,
            'ffnet_loss_data': ffnet_loss_data
        })
        
        return output_dict