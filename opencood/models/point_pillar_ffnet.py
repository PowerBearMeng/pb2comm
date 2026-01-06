import torch
import torch.nn as nn
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
# 引入我们之前定义的 FlowGenerator (即 FeatureFlowNet 里的 workers)
from opencood.models.sub_modules.flow_net import FeatureFlowNet as FlowGenerator

class PointPillarFFNet(nn.Module):
    def __init__(self, args):
        super(PointPillarFFNet, self).__init__()
        
        self.pillar_vfe = PillarVFE(args['pillar_vfe_args'], train=True, preserve_keypoint=False)
        self.scatter = PointPillarScatter(args['point_pillar_scatter_args'])

        self.backbone = BaseBEVBackbone(args['base_bev_backbone_args'], 256)
        
        # 关键一步：冻结老师网络！
        for p in self.backbone.parameters():
            p.requires_grad = False
            
        # -------------------------------------------------------
        # 3. 学生网络 (FlowNet) - 负责预测
        # -------------------------------------------------------
        # 注意：这里需要传入特定的参数，因为它的输入通道是 Backbone 的 2 倍
        # 如果你之前写的 feature_flow_net.py 里包含了 backbone 逻辑，就直接用
        # 如果是轻量级的，就只在这里定义 FlowHead
        
        # **FFNet原版逻辑**：FlowGenerator 也有一个 Backbone，而且是独立的！
        # 所以我们需要再定义一个 backbone 给 flow 用，或者 FlowGenerator 内部自己有
        self.flow_net = FlowGenerator(args) 

    def forward(self, data_dict):
        # -------------------------------------------------------
        # 1. 准备数据
        # -------------------------------------------------------
        # 假设 data_dict 里已经通过 dataloader 拿到了 t0, t1, t2 的点云
        # 并且已经做好了 VFE 和 Scatter 变成了 BEV 特征图 (pseudo-image)
        # 形状: [B, C, H, W]
        
        # 这里的 key 需要你根据 dataloader 的实现来定
        bev_t0 = data_dict['bev_feat_t0'] 
        bev_t1 = data_dict['bev_feat_t1']
        bev_t2 = data_dict['bev_feat_t2'] # 这是 GT

        # -------------------------------------------------------
        # 2. 老师跑一遍 T2 (生成目标)
        # -------------------------------------------------------
        with torch.no_grad(): # 确保不传梯度
            feat_t2_gt = self.backbone(bev_t2) # 得到 T2 时刻的高级特征

        # -------------------------------------------------------
        # 3. 学生跑一遍 T0+T1 (进行预测)
        # -------------------------------------------------------
        # FlowNet 内部会做 concat(t0, t1) -> backbone -> compress
        pred_feat_t2, flow = self.flow_net(bev_t0, bev_t1)

        # -------------------------------------------------------
        # 4. 计算 Loss (如果在模型内算的话)
        # -------------------------------------------------------
        # 也可以只返回 output_dict，在外部 train.py 里算
        output_dict = {
            'pred_feat': pred_feat_t2,
            'gt_feat': feat_t2_gt,
            'flow': flow
        }
        
        return output_dict