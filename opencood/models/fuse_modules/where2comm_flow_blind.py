# opencood/models/comm_modules/where2comm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.comm_modules.where2comm_mfh_comm import Communication
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
import numpy as np
class ScaledDotProductAttention(nn.Module):

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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PB2CommFusion(nn.Module):
    def __init__(self, feature_dim):
        super(PB2CommFusion, self).__init__()
        self.q_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.k_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.v_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.sqrt_dim = math.sqrt(feature_dim)

    def forward(self, neighbor_feature, neighbor_risk, neighbor_conf):
        """
        V2I 场景：
        neighbor_feature: [N, C, H, W] N=2时 [Ego, Infra]
        neighbor_risk: [N, 1 (或 2), H, W]   Ego的风险图
        neighbor_conf: [N, 1 (或 2), H, W]   各方的置信度
        """
        # ========================================================
        # 🌟 新增：维度保护机制（把多通道的置信度/风险图压缩成 1 通道）
        # ========================================================
        if neighbor_conf.shape[1] > 1:
            neighbor_conf = torch.max(neighbor_conf, dim=1, keepdim=True)[0]
            
        if neighbor_risk.shape[1] > 1:
            neighbor_risk = torch.max(neighbor_risk, dim=1, keepdim=True)[0]
        # ========================================================

        N, C, H, W = neighbor_feature.shape
        
        # 提取 Ego 接收方
        ego_feat = neighbor_feature[0:1]  # [1, C, H, W]
        ego_risk = neighbor_risk[0:1]     # [1, 1, H, W] (现在绝对是1通道了)
        
        # 兜底：只有自己时直接返回
        if N == 1:
            return ego_feat[0]
        
        # 生成本车的 Query
        Q = self.q_conv(ego_feat)  # [1, C, H, W]
        
        # 初始化输出
        out_feat = ego_feat.clone()  # [1, C, H, W]
        
        # ═══════════════════════════════════════════════════════
        # STEP 1: 供给端 (Supply) - 所有邻车信息的可信度加权
        # ═══════════════════════════════════════════════════════
        
        # 累积所有协作车的加权贡献
        aggregated_contrib = torch.zeros_like(ego_feat)  # [1, C, H, W]
        
        for i in range(1, N):
            collab_feat = neighbor_feature[i:i+1]  # [1, C, H, W]
            collab_conf = neighbor_conf[i:i+1]     # [1, 1, H, W]
            
            # 生成路端的 K, V
            K = self.k_conv(collab_feat)  # [1, C, H, W]
            V = self.v_conv(collab_feat)  # [1, C, H, W]
            
            # 计算特征匹配度 (空间上的相似性)
            # [1, C, H, W] 在每个空间位置计算通道维度的相似度
            energy = torch.sum(Q * K, dim=1, keepdim=True) / self.sqrt_dim
            # energy: [1, 1, H, W]
            
            base_attn = torch.sigmoid(energy)  # [1, 1, H, W] in (0, 1)
            
            # 【供给端权重】= 特征匹配度 * 邻车置信度
            # 只在这里做加权，不做归一化（因为只有一个邻车贡献）
            supply_weight = base_attn * collab_conf  # [1, 1, H, W]
            
            # 累积供给
            aggregated_contrib = aggregated_contrib + supply_weight * V
            # [1, C, H, W] += [1, 1, H, W] * [1, C, H, W]
        
        # ══════════════════════════════════════════════════���════
        # STEP 2: 需求端 (Demand) - 本车风险门控
        # ═══════════════════════════════════════════════════════
        
        # 【关键】需求门控：
        # - ego_risk 接近 0（安全区域）→ 不相信外来信息
        # - ego_risk 接近 1（危险区域）→ 充分吸收外来信息
        
        # demand_gate = ego_risk  # [1, 1, H, W]
        demand_gate = torch.clamp(ego_risk, min=0.05)
        # 应用需求门控到聚合特征
        gated_contrib = demand_gate * aggregated_contrib
        # [1, 1, H, W] * [1, C, H, W] → [1, C, H, W]
        
        # ═══════════════════════════════════════════════════════
        # STEP 3: 融合
        # ═══════════════════════════════════════════════════════
        
        # 残差连接
        out_feat = ego_feat + gated_contrib  # [1, C, H, W]
        
        return out_feat[0]  # [C, H, W]

class Where2comm(nn.Module):
    def __init__(self, args):
        super(Where2comm, self).__init__()

        self.communication = False
        self.round = 1
        if 'communication' in args and args['blind'] is True:
            self.communication = True
            self.naive_communication = Communication(args['communication'])
            if 'round' in args['communication']:
                self.round = args['communication']['round']
        self.discrete_ratio = args['voxel_size'][0]      
        self.downsample_rate = args['downsample_rate']
        
        self.agg_mode = args['agg_operator']['mode']
        print(self.agg_mode)
        print("-------------------------")
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
                elif self.agg_mode == 'PB':
                    fuse_network = PB2CommFusion(num_filters[idx])
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
    def forward(self, x, rm, record_len, pairwise_t_matrix, backbone=None, heads=None, 
                flow_map=None, blind_spot_mask=None, risk_map=None, current_epoch=None):
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
            
            # for i in range(self.num_levels):
            #     # 获取当前层的特征
            #     x_curr = feats[i] if with_resnet else backbone.blocks[i](x)

            #     # ================== 【插入点：多尺度时间对齐】 ==================
            #     if flow_map is not None:
            #         # 1. 获取当前特征图尺寸 (e.g., 50x176)
            #         curr_h, curr_w = x_curr.shape[2], x_curr.shape[3]
                    
            #         # 2. 获取 Flow 原始尺寸 (e.g., 200x704)
            #         flow_h, flow_w = flow_map.shape[2], flow_map.shape[3]
                    
            #         # 3. 将 Flow Map 缩放到当前尺寸
            #         flow_curr = F.interpolate(flow_map, size=(curr_h, curr_w), mode='bilinear', align_corners=True)
                    
            #         # 4. 【关键】缩放数值：分辨率变小了，像素位移也要变小
            #         scale_x = curr_w / flow_w
            #         scale_y = curr_h / flow_h
                    
            #         flow_curr_scaled = flow_curr.clone()
            #         flow_curr_scaled[:, 0] *= scale_x
            #         flow_curr_scaled[:, 1] *= scale_y
                    
            #         # 5. 执行 Warp (将 t1 特征变换为 t2 特征)
            #         x_curr = warp_feature(x_curr, flow_curr_scaled)
            #     # ==========================================================

            #     ############ 1. Communication #########
            #     if i==0:
            #         if self.communication:
            #             batch_confidence_maps = self.regroup(rm, record_len)
            #             comm_maps, communication_masks, communication_rates \
            #                 = self.naive_communication(batch_confidence_maps, 
            #                     record_len, 
            #                     pairwise_t_matrix, 
            #                     blind_spot_mask=blind_spot_mask,
            #                     risk_map=risk_map,
            #                     current_epoch=current_epoch)
            #             x_curr = x_curr * communication_masks
            #         else:
            #             communication_rates = torch.tensor(0).to(x.device)
            #             comm_maps = None 
                
            #     ############ 2. Split #######`l################
            #     batch_node_features = self.regroup(x_curr, record_len)
                
            #     ############ 3. Fusion (空间融合) ###########
            #     # 原来切分特征和置信度的代码：
            #     # batch_node_features = self.regroup(x, record_len)
            #     batch_conf_maps = self.regroup(rm, record_len)
                
            #     # 【新增 1】：按 record_len 切分你的 risk_map
            #     # (注意：如果你的 risk_map 和 rm 形状一样，也是 [N, 1, H, W]，就直接 regroup)
            #     if risk_map is not None:
            #         batch_risk_maps = self.regroup(risk_map, record_len)
            #     x_fuse = []
            #     for b in range(B):
            #         N = record_len[b]
            #         t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            #         node_features = batch_node_features[b]
            #         node_confs = batch_conf_maps[b]
            #         node_risks = batch_risk_maps[b] # 取出当前 batch 的 risk
                    
            #         C_feat, H_feat, W_feat = node_features.shape[1:]
            #         curr_h_b, curr_w_b = node_features.shape[2], node_features.shape[3]
                    
            #         # 空间 Warp
            #         neighbor_feature = warp_affine_simple(node_features,
            #                                         t_matrix[0, :, :, :],
            #                                         (curr_h_b, curr_w_b))
            #         # 融合
            #         if self.agg_mode == 'PB':
            #             # 【新增 2】：对齐置信度和 risk
            #             neighbor_conf = warp_affine_simple(node_confs, t_matrix[0, :, :, :], (H_feat, W_feat))
            #             neighbor_risk = warp_affine_simple(node_risks, t_matrix[0, :, :, :], (H_feat, W_feat))
                        
            #             # 【新增 3】：调用你的完美融合模块！
            #             fused_out = self.fuse_modules[i](neighbor_feature, neighbor_risk, neighbor_conf)
            #             x_fuse.append(fused_out)
            #         else:
            #             x_fuse.append(self.fuse_modules[i](neighbor_feature))
            #         # x_fuse.append(self.fuse_modules[i](neighbor_feature))
            #     x_fuse = torch.stack(x_fuse)

            #     ############ 4. Deconv (上采样) #############
            #     if len(backbone.deblocks) > 0:
            #         ups.append(backbone.deblocks[i](x_fuse))
            #     else:
            #         ups.append(x_fuse)
                
            for i in range(self.num_levels):
                # 获取当前层的特征
                x_curr = feats[i] if with_resnet else backbone.blocks[i](x)

                # ==========================================================
                # 【新增防御】：创建局部变量（分身），防止循环污染！
                # ==========================================================
                rm_i = rm
                risk_map_i = risk_map

                # ================== 【时空对齐】 ==================
                if flow_map is not None:
                    # 1. 获取尺寸
                    curr_h, curr_w = x_curr.shape[2], x_curr.shape[3]
                    flow_h, flow_w = flow_map.shape[2], flow_map.shape[3]
                    
                    # 2. 缩放 Flow Map
                    flow_curr = F.interpolate(flow_map, size=(curr_h, curr_w), mode='bilinear', align_corners=True)
                    scale_x = curr_w / flow_w
                    scale_y = curr_h / flow_h
                    
                    flow_curr_scaled = flow_curr.clone()
                    flow_curr_scaled[:, 0] *= scale_x
                    flow_curr_scaled[:, 1] *= scale_y
                    
                    # 3. 【推肉体】：执行特征 Warp
                    x_curr = warp_feature(x_curr, flow_curr_scaled)
                    
                    # 4. 【推灵魂】：执行辅助图 Warp
                    if risk_map is not None:
                        # 先缩放尺寸，再推移，并赋值给【分身】risk_map_i
                        risk_curr = F.interpolate(risk_map, size=(curr_h, curr_w), mode='nearest')
                        risk_map_i = warp_feature(risk_curr, flow_curr_scaled)
                        
                    if rm is not None:
                        # 先缩放尺寸，再推移，并赋值给【分身】rm_i
                        rm_curr = F.interpolate(rm, size=(curr_h, curr_w), mode='nearest')
                        rm_i = warp_feature(rm_curr, flow_curr_scaled)
                # ==========================================================

                ############ 1. Communication #########
                if i==0:
                    if self.communication:
                        # 【注意】：这里改用分身 rm_i 和 risk_map_i
                        batch_confidence_maps = self.regroup(rm_i, record_len)
                        comm_maps, communication_masks, communication_rates \
                            = self.naive_communication(batch_confidence_maps, 
                                record_len, 
                                pairwise_t_matrix, 
                                blind_spot_mask=blind_spot_mask,
                                risk_map=risk_map_i,  # <--- 使用推移后的风险图
                                current_epoch=current_epoch)
                        x_curr = x_curr * communication_masks
                    else:
                        communication_rates = torch.tensor(0).to(x.device)
                        comm_maps = None 
                
                ############ 2. Split ################
                batch_node_features = self.regroup(x_curr, record_len)
                
                ############ 3. Fusion (空间融合) ###########
                # 【注意】：这里全都要切分分身 rm_i 和 risk_map_i ！！！
                batch_conf_maps = self.regroup(rm_i, record_len)
                if risk_map_i is not None:
                    batch_risk_maps = self.regroup(risk_map_i, record_len)
                    
                x_fuse = []
                for b in range(B):
                    N = record_len[b]
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    
                    node_features = batch_node_features[b]
                    node_confs = batch_conf_maps[b]
                    node_risks = batch_risk_maps[b] if risk_map_i is not None else None
                    
                    # 严格使用 2 和 3 获取 H 和 W
                    H_feat, W_feat = node_features.shape[2], node_features.shape[3]
                    
                    # 空间 Warp (对齐到 Ego 视角)
                    neighbor_feature = warp_affine_simple(node_features, t_matrix[0, :, :, :], (H_feat, W_feat))
                    
                    if self.agg_mode == 'PB':
                        neighbor_conf = warp_affine_simple(node_confs, t_matrix[0, :, :, :], (H_feat, W_feat))
                        neighbor_risk = warp_affine_simple(node_risks, t_matrix[0, :, :, :], (H_feat, W_feat))
                        
                        # 调用完美融合模块
                        fused_out = self.fuse_modules[i](neighbor_feature, neighbor_risk, neighbor_conf)
                        x_fuse.append(fused_out)
                    else:
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
            pass
        
        return x_fuse, communication_rates, {'comm_maps': comm_maps}