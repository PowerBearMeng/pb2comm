# opencood/models/sub_modules/flow_generator.py
import torch
import torch.nn as nn
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone

class ReduceInfTC(nn.Module):
    """
    压缩层：用于调整 Flow Backbone 的输出，使其与特征图维度一致。
    对应原 FFNet 代码中的 ReduceInfTC。
    """
    def __init__(self, channel):
        super(ReduceInfTC, self).__init__()
        self.conv1_2 = nn.Conv2d(channel//2, channel//4, kernel_size=3, stride=2, padding=0)
        self.bn1_2 = nn.BatchNorm2d(channel//4, track_running_stats=True)
        self.conv1_3 = nn.Conv2d(channel//4, channel//8, kernel_size=3, stride=2, padding=0)
        self.bn1_3 = nn.BatchNorm2d(channel//8, track_running_stats=True)
        self.conv1_4 = nn.Conv2d(channel//8, channel//64, kernel_size=3, stride=2, padding=1)
        self.bn1_4 = nn.BatchNorm2d(channel//64, track_running_stats=True)

        self.deconv2_1 = nn.ConvTranspose2d(channel//64, channel//8, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = nn.BatchNorm2d(channel//8, track_running_stats=True)
        self.deconv2_2 = nn.ConvTranspose2d(channel//8, channel//4, kernel_size=3, stride=2, padding=0)
        self.bn2_2 = nn.BatchNorm2d(channel//4, track_running_stats=True)
        self.deconv2_3 = nn.ConvTranspose2d(channel//4, channel//2, kernel_size=3, stride=2, padding=0, output_padding=1)
        self.bn2_3 = nn.BatchNorm2d(channel//2, track_running_stats=True)

    def forward(self, x):
        out = torch.relu(self.bn1_2(self.conv1_2(x)))
        out = torch.relu(self.bn1_3(self.conv1_3(out)))
        out = torch.relu(self.bn1_4(self.conv1_4(out)))

        out = torch.relu(self.bn2_1(self.deconv2_1(out)))
        out = torch.relu(self.bn2_2(self.deconv2_2(out)))
        x_1 = torch.relu(self.bn2_3(self.deconv2_3(out)))
        return x_1

class FlowGenerator(nn.Module):
    def __init__(self, args):
        super(FlowGenerator, self).__init__()
        feature_dim = args.get('feature_dim', 384) 
        
        import copy
        backbone_cfg = copy.deepcopy(args['backbone'])
        
        input_channels = feature_dim * 2 
        backbone_cfg['in_channels'] = input_channels
        
        # 实例化 Flow Backbone (接受 768 维输入)
        self.flow_backbone = BaseBEVBackbone(backbone_cfg, input_channels)
        
        # 计算 Backbone 输出维度 (通常也是 384)
        if 'num_upsample_filter' in backbone_cfg:
            backbone_output_dim = sum(backbone_cfg['num_upsample_filter'])
        else:
            backbone_output_dim = backbone_cfg['num_filters'][-1]

        self.pre_encoder = ReduceInfTC(backbone_output_dim * 2) 

    def forward(self, feat_t0, feat_t1):
        """
        Args:
            feat_t0: [B, 384, H, W]
            feat_t1: [B, 384, H, W]
        """
        # 1. 拼接 -> [B, 768, H, W]
        input_feat = torch.cat([feat_t0, feat_t1], dim=1)
        
        # 2. Backbone -> [B, 384, H, W]
        backbone_out = self.flow_backbone({'spatial_features': input_feat})
        flow_feat = backbone_out['spatial_features_2d']
        
        # 3. Refine -> [B, 384, H, W]
        # 输出的 flow_pred 维度应与 feat_t1 一致，以便相加
        flow_pred = self.pre_encoder(flow_feat)
        
        return flow_pred