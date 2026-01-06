import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import copy
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
class ReduceInfTC(nn.Module):
    """
    FFNet中的核心压缩网络 (Encoder-Decoder结构)
    用于从拼接的特征中提取特征流信息。
    """
    def __init__(self, input_channel = 384):
        
        super(ReduceInfTC, self).__init__()
        
        c = input_channel
        print(f"ReduceInfTC 初始化: 输入通道 {input_channel}")
        # Encoder (下采样)
        self.conv1_2 = nn.Conv2d(c, c//2, kernel_size=3, stride=2, padding=0)
        self.bn1_2 = nn.BatchNorm2d(c//2, track_running_stats=True)
        
        self.conv1_3 = nn.Conv2d(c//2, c//4, kernel_size=3, stride=2, padding=0)
        self.bn1_3 = nn.BatchNorm2d(c//4, track_running_stats=True)
        
        self.conv1_4 = nn.Conv2d(c//4, c//32, kernel_size=3, stride=2, padding=1)
        self.bn1_4 = nn.BatchNorm2d(c//32, track_running_stats=True)

        # Decoder (上采样还原)
        # 注意：padding 和 output_padding 需要根据你的特征图尺寸微调，
        # OpenCOOD 常见的 BEV 尺寸下，这里使用通用的 stride=2 上采样
        self.deconv2_1 = nn.ConvTranspose2d(c//32, c//4, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = nn.BatchNorm2d(c//4, track_running_stats=True)
        
        self.deconv2_2 = nn.ConvTranspose2d(c//4, c//2, kernel_size=3, stride=2, padding=0)
        self.bn2_2 = nn.BatchNorm2d(c//2, track_running_stats=True)
        
        # 最后一层输出我们需要的目标通道数 (output_channel)
        self.deconv2_3 = nn.ConvTranspose2d(c//2, c, kernel_size=3, stride=2, padding=0, output_padding=1)
        self.bn2_3 = nn.BatchNorm2d(c, track_running_stats=True)
    def forward(self, x):
        # x: [B, input_channel, H, W]
        
        # Encoder
        out = F.relu(self.bn1_2(self.conv1_2(x)))
        out = F.relu(self.bn1_3(self.conv1_3(out)))
        out = F.relu(self.bn1_4(self.conv1_4(out)))
        
        # Decoder
        out = F.relu(self.bn2_1(self.deconv2_1(out)))
        out = F.relu(self.bn2_2(self.deconv2_2(out)))
        
        # Output
        out = F.relu(self.bn2_3(self.deconv2_3(out)))
        
        return out

class FlowGenerator(nn.Module):
    def __init__(self, args):
        super(FlowGenerator, self).__init__()
        
        # 1. 独立的 Voxel Feature Encoder (VFE)
        # FFNet 逻辑：FlowNet 拥有自己独立的一套 VFE 参数
        self.vfe = PillarVFE(args['pillar_vfe'],
                             num_point_features=4,
                             voxel_size=args['voxel_size'],
                             point_cloud_range=args['lidar_range'])

        # 2. 独立的 Scatter (把 Voxel 变成 BEV 图片)
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        # 3. 独立的、加宽的 Backbone
        # 核心逻辑：输入通道翻倍 (因为是 T0 + T1 拼接)
        
        # 提取原始骨干网配置
        backbone_cfg = copy.deepcopy(args['base_bev_backbone'])
        
        # --- 关键修改：翻倍输入通道 ---
        # 检查是 ResNet 还是普通 VoxelNet
        if 'resnet' in backbone_cfg and backbone_cfg['resnet']:
            # 对于 ResNetBackbone，第一层通常是 layer0 或者 conv1
            # 我们需要让 ResNet 接受双倍通道的输入 (例如 64 -> 128)
            # 注意：OpenCOOD 的 ResNetBEVBackbone 可能没有直接暴露 first_conv_in_channels 参数
            # 你可能需要去 ResNetBEVBackbone 的 __init__ 里确认一下它是否支持修改输入通道
            # 如果不支持，需要微调一下 ResNet 代码。
            # 这里假设它默认是 64，我们后面会处理
            pass 
        else:
            # 普通 BaseBEVBackbone (VoxelNet)
            # layer_nums: [3, 5, 5]
            # layer_strides: [2, 2, 2]
            # num_filters: [64, 128, 256] -> 第一层输入通常对应 Scatter 的输出 (64)
            # 我们需要把 Backbone 的第一层 filter 适应 128 的输入吗？
            # 不，Backbone 的输入通道是由 Scatter 的输出决定的。
            # Scatter 输出 = 64 (num_features)。
            # T0 Scatter + T1 Scatter 拼接 = 64 + 64 = 128。
            # 所以 Backbone 的输入必须能接受 128。
            pass

        # 为了保险，我们在这里实例化 Backbone，并传入一个特殊的参数 (如果你的 Backbone 支持)
        # 或者我们手动修改 backbone_cfg
        
        # *假设使用 ResNet*
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(backbone_cfg, 64 * 2) # 假设第二个参数是 in_channels
        else:
            self.backbone = BaseBEVBackbone(backbone_cfg, 64 * 2) 

        # 4. 预测头 (Compressor)
        # 你的 ReduceInfTC，输入是 Backbone 输出的通道数 (比如 384)
        # 注意：FFNet 的 backbone_flow 输出和普通 backbone 一样
        # 因为它只是输入变厚了，中间层 filter 数没变
        self.pre_encoder = ReduceInfTC(input_channel=384) 

    def forward(self, data_dict_t0, data_dict_t1):
        """
        Args:
            data_dict_t0: 包含 T0 时刻 voxel_features 等信息的字典
            data_dict_t1: 包含 T1 时刻 voxel_features 等信息的字典
        """
        
        # --- 处理 T0 ---
        batch_dict_0 = self.vfe(data_dict_t0)
        batch_dict_0 = self.scatter(batch_dict_0)
        # [B, 64, H, W]
        feat_t0 = batch_dict_0['spatial_features'] 

        # --- 处理 T1 ---
        batch_dict_1 = self.vfe(data_dict_t1)
        batch_dict_1 = self.scatter(batch_dict_1)

        # [B, 64, H, W]
        feat_t1 = batch_dict_1['spatial_features']

        # --- 核心逻辑：拼接 ---
        # [B, 128, H, W]
        cat_feat = torch.cat([feat_t0, feat_t1], dim=1)
        
        # --- 进 Backbone ---
        # 为了适配 BaseBEVBackbone 的输入格式，我们需要包装一下
        batch_dict_flow = {'spatial_features_2d': cat_feat}
        batch_dict_flow = self.backbone(batch_dict_flow)
        
        # [B, 384, H, W]
        flow_feat = batch_dict_flow['spatial_features_2d']
        
        # --- 进 Head ---
        # [B, 256, H, W]
        flow_pred = self.pre_encoder(flow_feat)
        
        return flow_pred