import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeightedSmoothL1Loss(nn.Module):
    """
    基础的 Smooth L1 Loss 实现 (保持不变)
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets
        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)
        return loss

class PointPillarFlowLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarFlowLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        # === 1. 检测任务权重 ===
        self.cls_weight = args['cls_weight']
        self.reg_coe = args['reg']
        
        # === 2. Flow 任务权重 ===
        # 如果 yaml 没写 flow_weight，默认给 1.0
        self.flow_coe = args.get('flow_weight', 1.0)
        # 过滤背景/空气的阈值
        self.mask_threshold = args.get('mask_threshold', 1e-2)
        
        self.loss_dict = {}
        self.use_dir = False

    def forward(self, output_dict, target_dict, prefix=''):
        """
        同时计算 检测 Loss + Flow Loss
        """
        # ==================================================
        # Part 1: 检测 Loss (Classification + Regression)
        # ==================================================
        rm = output_dict['rm{}'.format(prefix)]
        psm = output_dict['psm{}'.format(prefix)]
        targets = target_dict['targets']

        # --- Cls Loss ---
        cls_preds = psm.permute(0, 2, 3, 1).contiguous()
        box_cls_labels = target_dict['pos_equal_one']
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        
        cls_targets = box_cls_labels.unsqueeze(dim=-1).squeeze(dim=-1)
        one_hot_targets = torch.zeros(*list(cls_targets.shape), 2, dtype=cls_preds.dtype, device=cls_targets.device)
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:]
        
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # --- Reg Loss ---
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm, targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)
        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        # ==================================================
        # Part 2: Flow Loss (速度预测)
        # ==================================================
        flow_loss = self.compute_flow_loss(output_dict, target_dict)
        
        # ==================================================
        # Part 3: 总 Loss
        # ==================================================
        total_loss = reg_loss + conf_loss + flow_loss

        # 记录到字典方便打印
        self.loss_dict.update({
            'total_loss{}'.format(prefix): total_loss,
            'reg_loss{}'.format(prefix): reg_loss,
            'conf_loss{}'.format(prefix): conf_loss,
            'flow_loss{}'.format(prefix): flow_loss
        })

        return total_loss

    def compute_flow_loss(self, output_dict, target_dict):
        """
        计算 Flow 分支的 Loss
        使用 Detection GT (pos_equal_one) 作为 Mask
        """
        # 1. 检查是否有数据
        if 'ffnet_loss_data' not in output_dict:
            return torch.tensor(0.0).to(output_dict['psm'].device)

        ffnet_data = output_dict['ffnet_loss_data']
        if 'flow_pred' not in ffnet_data or 'flow_gt' not in ffnet_data:
            return torch.tensor(0.0).to(output_dict['psm'].device)

        pred_feat = ffnet_data['flow_pred']   # [B, C, H, W]
        target_feat = ffnet_data['flow_gt']   # [B, C, H, W]
        
        # 【修改点 1】在这里提前获取 C，无论走哪个 if 分支都能用
        B, C, H, W = pred_feat.shape 

        # 2. 获取正样本 Mask (Where Cars Are)
        pos_mask = target_dict['pos_equal_one']
        
        # 维度检查与调整
        if len(pos_mask.shape) == 3: # [B, H, W]
            mask_object = pos_mask.unsqueeze(1) # [B, 1, H, W]
        elif len(pos_mask.shape) == 2: # [B, H*W]
            # 这里不需要再获取 B,C,H,W 了，前面已经获取了
            mask_object = pos_mask.view(B, 1, H, W)
        else:
            # 防御性代码
            mask_object = (torch.abs(target_feat).sum(dim=1, keepdim=True) > 1e-2).float()

        # 1. 找出所有非空区域 (Lidar Occupancy)
        mask_lidar = (torch.abs(target_feat).sum(dim=1, keepdim=True) > 1e-4).float()
        # 在计算 loss 之前
        flow_mag = torch.sqrt(pred_feat[:, 0]**2 + pred_feat[:, 1]**2) # 计算流的模长
        print(f"Debug -> Flow Pred Mean: {flow_mag.mean().item():.6f} | Max: {flow_mag.max().item():.6f}")
        # 2. 组合 Mask (策略 B)
        final_mask = mask_lidar * 0.1 + mask_object * 0.9 
        
        # 3. 计算 Smooth L1 Loss
        loss_pixel = F.smooth_l1_loss(pred_feat, target_feat, beta=1.0, reduction='none')
        
        # 4. 应用 Mask
        loss_masked = loss_pixel * final_mask
        
        # 5. 归一化
        num_valid = mask_lidar.sum() * C + 1e-6 # 这里最好也乘上 C，保持量级一致
        
        # 打印诊断信息
        # 注意：你刚才打印的 Mean 是 0.1975，说明特征值确实比较小
        # print(f"GT Feat Mean: {target_feat.abs().mean().item():.4f}, Max: {target_feat.max().item():.4f}")
        
        flow_loss = loss_masked.sum() / num_valid

        # === 诊断打印 ===
        diff = torch.abs(pred_feat - target_feat)
        
        # 【修改点 2】现在 C 已经被定义了，这里就不会报错了
        car_diff = (diff * mask_object).sum() / (mask_object.sum() * C + 1e-6)
        
        mask_bg = mask_lidar * (1 - mask_object)
        bg_diff = (diff * mask_bg).sum() / (mask_bg.sum() * C + 1e-6)
        
        print(f"Debug -> Car Abs Diff: {car_diff.item():.4f} | BG Abs Diff: {bg_diff.item():.4f}")
        
        return flow_loss * self.flow_coe
    def cls_loss_func(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)
        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)
        loss = focal_weight * bce_loss
        if weights.shape.__len__() == 2 or (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)
        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        loss = torch.clamp(input, min=0) - input * target + torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    def logging(self, epoch, batch_id, batch_len, writer=None):
        """
        打印所有 Loss 组件
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        conf_loss = self.loss_dict.get('conf_loss', 0)
        flow_loss = self.loss_dict.get('flow_loss', 0)

        print_msg = "[epoch {}][{}/{}], || Loss: {:.2f} || Conf: {:.2f} | Reg: {:.2f} | Flow: {:.2f}".format(
            epoch, batch_id + 1, batch_len, total_loss, conf_loss, reg_loss, flow_loss
        )
        print(print_msg)

        if writer is not None:
            writer.add_scalar('Total_Loss', total_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Conf_Loss', conf_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Reg_Loss', reg_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Flow_Loss', flow_loss, epoch * batch_len + batch_id)