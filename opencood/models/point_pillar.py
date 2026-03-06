# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
# ==================== 【新增 1】导入所需模块 ====================
from opencood.models.sub_modules.motion_head import MotionHead
from opencood.models.point_pillar_motion import sample_features_from_coords
# ================================================================

class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()
        self.pc_range = args['lidar_range']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_num'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
                                  kernel_size=1)
        
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(128 * 3, args['dir_args']['num_bins'] * args['anchor_num'],
                                  kernel_size=1) # BIN_NUM = 2
        else:
            self.use_dir = False
        # self.open_motion = True
        # self.pred_len = 5
        # motion_dim = 384
        # self.motion_head = MotionHead(in_channels=motion_dim , pred_len=self.pred_len)

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {'psm': psm,
                       'rm': rm}
                       
        if self.use_dir:
            dm = self.dir_head(spatial_features_2d)
            output_dict.update({'dm': dm})
        
        # # ==================== 【新增 3】轨迹预测分支 ====================
        # if 'object_bbx_center' in data_dict and self.open_motion:
        #     gt_centers = data_dict['object_bbx_center']
        #     # ========== 【新增打印】 ==========
        #     if gt_centers.shape[1] == 0: # 假设维度是 [B, N, 7]，N为0说明没车
        #         print("\n[DEBUG-Model] ⚠️ 这一帧 gt_centers 里根本没有车！")
        #     # ==================================
        #     # 直接从 384 维的 BEV 特征图中提取对应坐标的特征
        #     obj_feats = sample_features_from_coords(
        #         spatial_features_2d, 
        #         gt_centers[..., :2], 
        #         self.pc_range
        #     )
            
        #     # 经过 MotionHead 预测轨迹
        #     traj_preds = self.motion_head(obj_feats)
        #     output_dict['traj_preds'] = traj_preds
        # # ================================================================
        # else:
        #    print("\n[DEBUG-Model] ❌ 没进轨迹预测分支！'object_bbx_center' 是否存在:", 'object_bbx_center' in data_dict)
        return output_dict