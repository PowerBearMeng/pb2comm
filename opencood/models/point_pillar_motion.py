# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from numpy import record
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.dcn_net import DCNNet
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.fuse_modules.where2comm_attn import Where2comm
from opencood.models.sub_modules.motion_head import MotionHead
import torch.nn.functional as F  # 【新增】用于 grid_sample
import torch

def sample_features_from_coords(feature_map, coords, pc_range):
    """
    Args:
        feature_map: (B, C, H, W)
        coords: (B, N, 2) [x, y]
        pc_range: list
    """
    # ================= [修改点] =================
    # 强制将坐标转为与 feature_map 相同的类型 (float32) 和 设备 (cuda)
    coords = coords.to(dtype=feature_map.dtype, device=feature_map.device)
    # ===========================================

    # 1. 获取尺寸
    B, C, H, W = feature_map.shape
    x = coords[..., 0]
    y = coords[..., 1]
    
    # 2. 归一化到 [-1, 1]
    x_min, y_min = pc_range[0], pc_range[1]
    x_max, y_max = pc_range[3], pc_range[4]
    
    norm_x = 2 * (x - x_min) / (x_max - x_min) - 1
    norm_y = 2 * (y - y_min) / (y_max - y_min) - 1
    
    # (B, N, 1, 2)
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)
    
    # 3. 采样
    object_features = F.grid_sample(feature_map, grid, align_corners=True, padding_mode='zeros')
    
    object_features = object_features.squeeze(-1).permute(0, 2, 1)
    return object_features

class PointPillarMotion(nn.Module):
    def __init__(self, args):
        super(PointPillarMotion, self).__init__()
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.dcn = False
        if 'dcn' in args:
            self.dcn = True
            self.dcn_net = DCNNet(args['dcn'])

        # self.fusion_net = TransformerFusion(args['fusion_args'])
        self.fusion_net = Where2comm(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        
        c_dim = args['shrink_header']['dim'][0] 
        self.pred_len = args.get('pred_len', 5) # 最好在 yaml 里加一个
        self.pc_range = args['lidar_range']
        self.motion_head = MotionHead(in_channels=c_dim, pred_len=self.pred_len)

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
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        # N, C, H', W'. [N, 384, 100, 352]
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        # dcn
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)
        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)

        # print('spatial_features_2d: ', spatial_features_2d.shape)
        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(batch_dict['spatial_features'],
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix, 
                                            self.backbone,
                                            [self.shrink_conv, self.cls_head, self.reg_head])
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features_2d,
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix)
            
            
        # print('fused_feature: ', fused_feature.shape)
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm
                       }
        output_dict.update(result_dict)
        
        # ==================================================
        # 【新增】轨迹预测分支
        # ==================================================
        # 1. 检查是否有 'ego' 键 (通常在 train.py 里 collate 好的 batch 都有)
        if 'object_bbx_center' in data_dict:
            # (B, N, 7) -> 这里的 N 是 max_num (e.g. 100)
            gt_centers = data_dict['object_bbx_center']
            
            # 2. 采样特征
            # fused_feature: [B, 256, H, W]
            # gt_centers[..., :2]: [B, N, 2] (x, y)
            obj_feats = sample_features_from_coords(
                fused_feature, 
                gt_centers[..., :2], 
                self.pc_range
            )
            
            # 3. 预测轨迹
            # 输入: (B, N, 256) -> 输出: (B, N, 5, 2)
            traj_preds = self.motion_head(obj_feats)
            
            # 4. 存入 output_dict
            output_dict['traj_preds'] = traj_preds

        # ==================================================

        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        psm_single_v = []
        psm_single_i = []
        rm_single_v = []
        rm_single_i = []
        for b in range(len(split_psm_single)):
            psm_single_v.append(split_psm_single[b][0:1])
            psm_single_i.append(split_psm_single[b][1:2])
            rm_single_v.append(split_rm_single[b][0:1])
            rm_single_i.append(split_rm_single[b][1:2])
        psm_single_v = torch.cat(psm_single_v, dim=0)
        psm_single_i = torch.cat(psm_single_i, dim=0)
        rm_single_v = torch.cat(rm_single_v, dim=0)
        rm_single_i = torch.cat(rm_single_i, dim=0)
        output_dict.update({'psm_single_v': psm_single_v,
                       'psm_single_i': psm_single_i,
                       'rm_single_v': rm_single_v,
                       'rm_single_i': rm_single_i,
                       'comm_rate': communication_rates
                       })
        return output_dict
