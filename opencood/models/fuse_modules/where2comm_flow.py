# opencood/models/comm_modules/where2comm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.comm_modules.where2comm import Communication
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context

class AttenFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttenFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x

class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0)[0]


class EncodeLayer(nn.Module):
    def __init__(self, channels, n_head=8, dropout=0):
        super(EncodeLayer, self).__init__()
        self.attn = nn.MultiheadAttention(channels, n_head, dropout)
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, q, k, v, confidence_map=None):
        """
        order (seq, batch, feature)
        Args:
            q: (1, H*W, C)
            k: (N, H*W, C)
            v: (N, H*W, C)
        Returns:
            outputs: ()
        """
        residual = q
        if confidence_map is not None:
            context, weight = self.attn(q,k,v, quality_map=confidence_map) # (1, H*W, C)
        else:
            context, weight = self.attn(q,k,v) # (1, H*W, C)
        context = self.dropout1(context)
        output1 = self.norm1(residual + context)

        # feed forward net
        residual = output1 # (1, H*W, C)
        context = self.linear2(self.relu(self.linear1(output1)))
        context = self.dropout2(context)
        output2 = self.norm2(residual + context)

        return output2

class TransformerFusion(nn.Module):
    def __init__(self, channels=256, n_head=8, with_spe=True, with_scm=True, dropout=0):
        super(TransformerFusion, self).__init__()

        self.encode_layer = EncodeLayer(channels, n_head, dropout)
        self.with_spe = with_spe
        self.with_scm = with_scm
        
    def forward(self, batch_neighbor_feature, batch_neighbor_feature_pe, batch_confidence_map, record_len):
        x_fuse = []
        B = len(record_len)
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            neighbor_feature = batch_neighbor_feature[b]
            _, C, H, W = neighbor_feature.shape
            neighbor_feature_flat = neighbor_feature.view(N,C,H*W)  # (N, C, H*W)

            if self.with_spe:
                neighbor_feature_pe = batch_neighbor_feature_pe[b]
                neighbor_feature_flat_pe = neighbor_feature_pe.view(N,C,H*W)  # (N, C, H*W)
                query = neighbor_feature_flat_pe[0:1,...].permute(0,2,1)  # (1, H*W, C)
                key = neighbor_feature_flat_pe.permute(0,2,1)  # (N, H*W, C)
            else:
                query = neighbor_feature_flat[0:1,...].permute(0,2,1)  # (1, H*W, C)
                key = neighbor_feature_flat.permute(0,2,1)  # (N, H*W, C)
            
            value = neighbor_feature_flat.permute(0,2,1)

            if self.with_scm:
                confidence_map = batch_confidence_map[b]
                fused_feature = self.encode_layer(query, key, value, confidence_map)  # (1, H*W, C)
            else:
                fused_feature = self.encode_layer(query, key, value)  # (1, H*W, C)
            
            fused_feature = fused_feature.permute(0,2,1).reshape(1, C, H, W)

            x_fuse.append(fused_feature)
        x_fuse = torch.concat(x_fuse, dim=0)
        return x_fuse

def add_pe_map(x):
    # scale = 2 * math.pi
    temperature = 10000
    num_pos_feats = x.shape[-3] // 2  # positional encoding dimension. C = 2d

    mask = torch.zeros([x.shape[-2], x.shape[-1]], dtype=torch.bool, device=x.device)  #[H, W]
    not_mask = ~mask
    y_embed = not_mask.cumsum(0, dtype=torch.float32)  # [H, W]
    x_embed = not_mask.cumsum(1, dtype=torch.float32)  # [H, W]

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)  # [0,1,2,...,d]
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)  # 10000^(2k/d), k is [0,0,1,1,...,d/2,d/2]

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)  # [C, H, W]

    if len(x.shape) == 4:
        x_pe = x + pos[None,:,:,:]
    elif len(x.shape) == 5:
        x_pe = x + pos[None,None,:,:,:]
    return x_pe

# ================== 【新增辅助函数】 Warp 实现 ==================
def warp_feature(x, flow):
    """
    Args:
        x: [B, C, H, W] 特征图
        flow: [B, 2, H, W] 预测的位移场 (单位: 像素)
    """
    B, C, H, W = x.size()
    # 1. 生成网格
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(x.device)

    # 2. 加上 Flow
    vgrid = grid + flow

    # 3. 归一化到 [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1) # [B, H, W, 2]

    # 4. 采样
    output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return output
# ==============================================================



class Where2comm(nn.Module):
    def __init__(self, args):
        super(Where2comm, self).__init__()

        self.communication = False
        self.round = 1
        if 'communication' in args:
            self.communication = True
            self.naive_communication = Communication(args['communication'])
            if 'round' in args['communication']:
                self.round = args['communication']['round']
        self.discrete_ratio = args['voxel_size'][0]      
        self.downsample_rate = args['downsample_rate']
        
        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        
        # 初始化多尺度融合模块
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.agg_mode == 'ATTEN':
                    fuse_network = AttenFusion(num_filters[idx])
                elif self.agg_mode == 'MAX':
                    fuse_network = MaxFusion()
                # ... (Transformer logic omitted for brevity) ...
                self.fuse_modules.append(fuse_network)
        else:
            if self.agg_mode == 'ATTEN':
                self.fuse_modules = AttenFusion(args['agg_operator']['feature_dim'])
            elif self.agg_mode == 'MAX':
                self.fuse_modules = MaxFusion()   

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    # ================== 【核心修改】 Forward ==================
    def forward(self, x, rm, record_len, pairwise_t_matrix, backbone=None, heads=None, flow_map=None):
        """
        新增参数: flow_map (Tensor): [B, 2, H_orig, W_orig], 预测出的从 t1 到 t2 的位移
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # 空间变换矩阵预处理 (保持原逻辑)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        if self.multi_scale:
            ups = []
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(x) # 获取多尺度特征列表 [feat1, feat2, feat3]
            
            for i in range(self.num_levels):
                # 获取当前层的特征
                x_curr = feats[i] if with_resnet else backbone.blocks[i](x)

                # ================== 【插入点：多尺度时间对齐】 ==================
                if flow_map is not None:
                    # 1. 获取当前特征图尺寸 (e.g., 50x176)
                    curr_h, curr_w = x_curr.shape[2], x_curr.shape[3]
                    
                    # 2. 获取 Flow 原始尺寸 (e.g., 200x704)
                    flow_h, flow_w = flow_map.shape[2], flow_map.shape[3]
                    
                    # 3. 将 Flow Map 缩放到当前尺寸
                    flow_curr = F.interpolate(flow_map, size=(curr_h, curr_w), mode='bilinear', align_corners=True)
                    
                    # 4. 【关键】缩放数值：分辨率变小了，像素位移也要变小
                    scale_x = curr_w / flow_w
                    scale_y = curr_h / flow_h
                    
                    flow_curr_scaled = flow_curr.clone()
                    flow_curr_scaled[:, 0] *= scale_x
                    flow_curr_scaled[:, 1] *= scale_y
                    
                    # 5. 执行 Warp (将 t1 特征变换为 t2 特征)
                    x_curr = warp_feature(x_curr, flow_curr_scaled)
                # ==========================================================

                ############ 1. Communication #########
                if i==0:
                    if self.communication:
                        batch_confidence_maps = self.regroup(rm, record_len)
                        comm_maps, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                        x_curr = x_curr * communication_masks
                    else:
                        communication_rates = torch.tensor(0).to(x.device)
                        comm_maps = None 
                
                ############ 2. Split #######################
                batch_node_features = self.regroup(x_curr, record_len)
                
                ############ 3. Fusion (空间融合) ###########
                x_fuse = []
                for b in range(B):
                    N = record_len[b]
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    curr_h_b, curr_w_b = node_features.shape[2], node_features.shape[3]
                    
                    # 空间 Warp
                    neighbor_feature = warp_affine_simple(node_features,
                                                    t_matrix[0, :, :, :],
                                                    (curr_h_b, curr_w_b))
                    # 融合
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
                x_fuse = torch.stack(x_fuse)

                ############ 4. Deconv (上采样) #############
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)
                
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)

        else:
            # (非多尺度的逻辑保持不变...)
            pass
        
        return x_fuse, communication_rates, {'comm_maps': comm_maps}