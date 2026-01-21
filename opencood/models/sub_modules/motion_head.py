import torch
import torch.nn as nn

class MotionHead(nn.Module):
    def __init__(self, in_channels, pred_len=5):
        super(MotionHead, self).__init__()
        self.pred_len = pred_len
        
        # 一个简单的 MLP：输入特征 -> 隐藏层 -> 输出 (未来帧数 * 2)
        # 这里的 output 维度是 pred_len * 2 (因为是 x, y 偏移量)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len * 2) 
        )

    def forward(self, x):
        """
        x: (B, N, C) - Batch里每辆车的特征向量
        """
        B, N, C = x.shape
        # 经过 MLP
        out = self.mlp(x) # (B, N, pred_len * 2)
        
        # Reshape 成 (B, N, pred_len, 2)
        out = out.view(B, N, self.pred_len, 2)
        return out