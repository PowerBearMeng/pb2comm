# opencood/models/sub_modules/flow_generator.py
import torch
import torch.nn as nn
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone

class ReduceInfTC(nn.Module):
    """
    压缩层：用于调整 Flow Backbone 的输出，使其与特征图维度一致。
    对应原 FFNet 代码中的 ReduceInfTC。
    """
    def __init__(self, channel = 768):
        super(ReduceInfTC, self).__init__()
        self.conv1_2 = nn.Conv2d(channel//2, channel//4, kernel_size=3, stride=2, padding=0)
        self.bn1_2 = nn.BatchNorm2d(channel//4, track_running_stats=True)
        self.conv1_3 = nn.Conv2d(channel//4, channel//8, kernel_size=3, stride=2, padding=0)
        self.bn1_3 = nn.BatchNorm2d(channel//8, track_running_stats=True)
        self.conv1_4 = nn.Conv2d(channel//8, channel//64, kernel_size=3, stride=2, padding=1)
        self.bn1_4 = nn.BatchNorm2d(channel//64, track_running_stats=True)

        # ----------------- Flow Prediction (H/8) -----------------
        # 在最底层先预测一个粗糙的光流 (Batch, 2, H/8, W/8)
        self.flow_head_small = nn.Conv2d(channel//64, 2, kernel_size=3, stride=1, padding=1)

        # ----------------- Flow Upsampling (2->2) -----------------
        # 这里就是你想要的 "2->2" 层，只负责把光流变大，不负责处理特征，非常轻量
        self.up_flow_1 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=2, padding=1, bias=True)
        self.up_flow_2 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=2, padding=0, bias=True)
        self.up_flow_3 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=2, padding=0, output_padding=1, bias=True)
    
    def forward(self, x):
        out = torch.relu(self.bn1_2(self.conv1_2(x)))
        out = torch.relu(self.bn1_3(self.conv1_3(out)))
        out = torch.relu(self.bn1_4(self.conv1_4(out)))

        # --- Predict Low-Res Flow ---
        flow = self.flow_head_small(out) # [B, 2, H/8, W/8]

        # --- Upsample Flow (2->2) ---
        # 注意：不加 ReLU，因为光流是线性叠加的，且有正有负
        flow = self.up_flow_1(flow) # -> [B, 2, H/4, W/4]
        flow = self.up_flow_2(flow) # -> [B, 2, H/2, W/2]
        flow = self.up_flow_3(flow) # -> [B, 2, H, W]
        return flow

class FlowGenerator(nn.Module):
    def __init__(self, args):
        super(FlowGenerator, self).__init__()
        feature_dim = args.get('feature_dim', 64) 
        
        import copy
        backbone_cfg = copy.deepcopy(args['backbone'])
        
        input_channels = feature_dim * 2 
        backbone_cfg['in_channels'] = input_channels
        
        self.flow_backbone = BaseBEVBackbone(backbone_cfg, input_channels)
        self.pre_encoder = ReduceInfTC(768)  # 出来的是 [B, 2, H, W]

    def forward(self, feat_t0, feat_t1):
        """
        Args:
            feat_t0: [B, 64, H, W]
            feat_t1: [B, 64, H, W]
        """
        # 1. 拼接 -> [B, 128, H, W]
        input_feat = torch.cat([feat_t0, feat_t1], dim=1)
        
        # 2. Backbone -> [B, 128, H, W]
        backbone_out = self.flow_backbone({'spatial_features': input_feat})
        flow_feat = backbone_out['spatial_features_2d']
        
        # 3. Refine -> [B, 2, H, W]
        # 输出的 flow_pred 维度应与 feat_t1 一致，以便相加
        flow_pred = self.pre_encoder(flow_feat)
        
        return flow_pred