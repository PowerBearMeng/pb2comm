# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from numpy import record
import torch.nn as nn
import numpy as np
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.dcn_net import DCNNet
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.utils.blind_spot_utils import get_blind_spot_mask
from opencood.models.fuse_modules.where2comm_mfh import Where2comm
import torch

class PointPillarWhere2commmfh(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2commmfh, self).__init__()
        self.lidar_range = args['lidar_range']
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
        
        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)

        # =======================================================
        # 【最终修正版逻辑】: 使用 record_len 来确定循环次数
        # =======================================================
        blind_spot_mask = None
        # 确保有原始点云
        if 'origin_lidar' in data_dict:
            # 获取特征图尺寸 (H, W)
            # 注意：这里我们只取 H 和 W，不要取 B！
            _, _, H, W = spatial_features_2d.shape
            
            # 获取真实的 Batch Size (场景数量)
            # record_len 是一个列表，长度等于 batch_size，记录了每个场景有多少辆车
            real_batch_size = len(record_len)
            
            gt_range = self.lidar_range
            
            # 获取 Batch 点云数据 [Batch_Size, N, C]
            batch_origin_lidar = data_dict['origin_lidar']
            
            mask_list = []
            
            # === 使用真实的 Batch Size 进行循环 ===
            for b in range(real_batch_size):
                # 1. 获取当前样本的点云
                lidar_tensor = batch_origin_lidar[b]
                
                # 转为 Numpy
                if isinstance(lidar_tensor, torch.Tensor):
                    lidar_np = lidar_tensor.cpu().numpy()
                else:
                    lidar_np = lidar_tensor

                # 2. 计算当前样本的 Mask
                mask_np = get_blind_spot_mask(
                    lidar_np, 
                    ego_pose=(0,0,0), 
                    lidar_range=gt_range, 
                    target_feat_shape=(H, W),
                    voxel_size=0.4
                )
                
                # 转为 Tensor
                mask_tensor = torch.from_numpy(mask_np).to(spatial_features_2d.device).float()
                mask_list.append(mask_tensor)

            # === 堆叠 Mask ===
            # 结果形状应该是 [Batch_Size, 1, H, W]
            blind_spot_mask = torch.stack(mask_list, dim=0).unsqueeze(1)
            
        else:
            if not self.training:
                 print("DEBUG: 'origin_lidar' NOT found in data_dict!")
        # =======================================================

        if self.multi_scale:
            # 注意：你需要确保 self.fusion_net 的 forward 接受 blind_spot_mask 参数
            fused_feature, communication_rates, result_dict = self.fusion_net(
                                            batch_dict['spatial_features'],
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix, 
                                            self.backbone,
                                            [self.shrink_conv, self.cls_head, self.reg_head],
                                            blind_spot_mask=blind_spot_mask) # <--- 传入 Mask
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            # 单尺度模式
            fused_feature, communication_rates, result_dict = self.fusion_net(
                                            spatial_features_2d,
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix,
                                            blind_spot_mask=blind_spot_mask) # <--- 传入 Mask
            
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm
                       }
        
        # =======================================================
        # 【恢复被我误删的逻辑】: 这里的代码必须保留！
        # =======================================================
        output_dict.update(result_dict)
        
        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        psm_single_v = []
        psm_single_i = []
        rm_single_v = []
        rm_single_i = []
        
        # 这里的 try-except 或者 len 判断是为了防止有些数据只有单车没有邻居
        # 但原来的代码逻辑是这样的：
        for b in range(len(split_psm_single)):
            # 假设每个场景至少有1个Ego和1个Infra (record_len >= 2)
            # 如果实际数据中有些只有Ego，这里可能会越界，保持你原有逻辑即可
            if split_psm_single[b].shape[0] > 1:
                psm_single_v.append(split_psm_single[b][0:1])
                psm_single_i.append(split_psm_single[b][1:2])
                rm_single_v.append(split_rm_single[b][0:1])
                rm_single_i.append(split_rm_single[b][1:2])
            else:
                # 处理只有单车的情况，或者根据需要填充
                psm_single_v.append(split_psm_single[b][0:1])
                # 如果没有 infra，可能需要 append 一个全零或者处理方式
                # 暂时保持你原代码的假设 (默认有 infra)
        
        if len(psm_single_v) > 0:
            psm_single_v = torch.cat(psm_single_v, dim=0)
            rm_single_v = torch.cat(rm_single_v, dim=0)
            output_dict.update({
                'psm_single_v': psm_single_v,
                'rm_single_v': rm_single_v,
                'comm_rate': communication_rates
            })

        if len(psm_single_i) > 0:
            psm_single_i = torch.cat(psm_single_i, dim=0)
            rm_single_i = torch.cat(rm_single_i, dim=0)
            output_dict.update({
                'psm_single_i': psm_single_i,
                'rm_single_i': rm_single_i
            })

        return output_dict