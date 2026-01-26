# opencood/models/sub_modules/flow.py
import torch
import torch.nn as nn

class FlowGenerator(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 假设输入特征维度是 384 (来自 Backbone)
        input_dim = args.get('feature_dim', 384)
        
        # 1. 降维 + 融合 (1x1 Conv)
        # 先把两个 384 的特征压扁并融合，减少计算量
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=1),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True)
        )
        
        # 2. 轻量级流计算网络 (取代 ResNet)
        # 使用几个 3x3 卷积，甚至可以加一点 dilation (空洞) 来微调感受野
        self.flow_net = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_dim, input_dim //2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim //2),
            nn.ReLU(inplace=True),
            
            # Layer 2 (可以重复几次)
            nn.Conv2d(input_dim //2, input_dim //4, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim //4),
            nn.ReLU(inplace=True),
            
            # Layer 3
            nn.Conv2d(input_dim //4, input_dim //8, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim //8),
            nn.ReLU(inplace=True)
        )
        
        # 3. 输出头 (回归 dx, dy)
        self.flow_head = nn.Conv2d(input_dim //8, 2, kernel_size=1)
        
        # 初始化
        nn.init.normal_(self.flow_head.weight, mean=0, std=0.001)
        nn.init.constant_(self.flow_head.bias, 0)

    def forward(self, feat_t0, feat_t1):
        # feat_t0, feat_t1: [B, 384, H, W]
        
        # 1. Concat
        x = torch.cat([feat_t0, feat_t1], dim=1) # [B, 768, H, W]
        
        # 2. Reduce & Extract
        x = self.reduce_conv(x) # [B, 384, H, W]
        x = self.flow_net(x)    # [B, 48, H, W]
        
        # 3. Regress Flow
        flow = self.flow_head(x) # [B, 2, H, W]
        
        return flow