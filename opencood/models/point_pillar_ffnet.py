# opencood/models/point_pillar_ffnet.py
import torch
import torch.nn as nn
from opencood.models.point_pillar_where2comm import PointPillarWhere2comm
from opencood.models.sub_modules.flow_net import FlowGenerator

class PointPillarFFNet(PointPillarWhere2comm):
    def __init__(self, args):
        super(PointPillarFFNet, self).__init__(args)
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

            
    def extract_bev_feature(self, data_dict):
        """
        提取历史帧特征 (t0, t1)。
        注意：这里只提取到 Backbone 输出 (384维)，【不进行压缩】。
        """
        batch_dict = self.pillar_vfe(data_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        
        # 直接返回 384 维特征
        return batch_dict['spatial_features_2d']

    def forward(self, data_dict):
        # 1. 主路特征提取 (t2)
        # ----------------------------------------------------
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

        # 2. FFNet 预测与替换 (在 384 维进行)
        # ----------------------------------------------------
        if 'ffnet_t0' in data_dict.keys():
            # 确保数据是成对的 (Veh, Infra)
            if not torch.all(record_len == 2):
                 raise ValueError("FFNet requires fixed [Vehicle, Infrastructure] pairs.")

            ffnet_t0_dict = data_dict['ffnet_t0']
            ffnet_t1_dict = data_dict['ffnet_t1']
            ffnet_time = data_dict['ffnet_time']
            # 提取历史特征 (384维)
            with torch.no_grad():
                 feat_t0 = self.extract_bev_feature(ffnet_t0_dict) 
                 feat_t1 = self.extract_bev_feature(ffnet_t1_dict) 

            # 生成 Flow
            flow_pred = self.flow_generator(feat_t0, feat_t1)
            
            # 时间缩放
            dt_01 = ffnet_time['t_0_1'].view(-1, 1, 1, 1).to(flow_pred.device)
            dt_12 = ffnet_time['t_1_2'].view(-1, 1, 1, 1).to(flow_pred.device)
            
            # 预测 t2
            feat_pred_t2 = feat_t1 + flow_pred / (dt_01 + 1e-6) * dt_12
            
            # 【替换】将预测的路侧特征填入 spatial_features_2d
            spatial_features_2d[1::2] = feat_pred_t2
            # ================== 【新增】计算 FFNet Loss 所需数据 ==================
            # 我们需要提取 t2 时刻的【真实特征】(Ground Truth) 来监督 feat_pred_t2
            # 只有在训练模式下，且提供了 ffnet_t2 数据时才计算
            if 'ffnet_t2' in data_dict:
                ffnet_t2_dict = data_dict['ffnet_t2']
                with torch.no_grad():
                    # 提取真实的 t2 特征 (384维度)
                    feat_gt_t2 = self.extract_bev_feature(ffnet_t2_dict)
                
                # 将预测值和真实值存入字典，传给 train.py
                ffnet_loss_data['flow_pred'] = feat_pred_t2
                ffnet_loss_data['flow_gt'] = feat_gt_t2
        else:
            print("Warning: FFNet temporal data not found in input. Skipping FFNet prediction.")
        # 3. 压缩/下采样 (384 -> 256)
        # ----------------------------------------------------
        # 这一步移动到了 替换 之后
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
            
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
            
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)

        # 4. 融合 (Where2comm)
        # ----------------------------------------------------
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