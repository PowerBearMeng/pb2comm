import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeightedSmoothL1Loss(nn.Module):
    """
    基础的 Smooth L1 Loss (保持不变)
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
        target = torch.where(torch.isnan(target), input, target)
        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        return loss

class PointPillarFlowMotionLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarFlowMotionLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        # === 1. 检测权重 ===
        self.cls_weight = args['cls_weight']
        self.reg_coe = args['reg']
        
        # === 2. 轨迹权重 (来自 Motion) ===
        # 如果 yaml 没配置 traj，建议给个默认值 0.2
        self.traj_coe = args.get('traj', 0.2) 

        # === 3. Flow 权重 (来自 Flow) ===
        # 如果 yaml 没配置 flow_weight，默认给 1.0
        self.flow_coe = args.get('flow_weight', 1.0)
        
        self.loss_dict = {}
        self.use_dir = False

    def forward(self, output_dict, target_dict, prefix=''):
        """
        同时计算: Conf + Reg + Traj + Flow
        """
        rm = output_dict['rm{}'.format(prefix)]
        psm = output_dict['psm{}'.format(prefix)]
        targets = target_dict['targets']

        # ==================================================
        # Part 1: Classification Loss (保持不变)
        # ==================================================
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

        # ==================================================
        # Part 2: Regression Loss (保持不变)
        # ==================================================
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm, targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)
        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        # ==================================================
        # Part 3: Flow Loss (新增，来自 PointPillarFlowLoss)
        # ==================================================
        flow_loss = self.compute_flow_loss(output_dict, target_dict, prefix=prefix)

        # ==================================================
        # Part 4: Trajectory Loss (新增，来自 PointPillarMotionLoss)
        # ==================================================
        traj_loss = self.compute_trajectory_loss(output_dict, target_dict)

        # ==================================================
        # Total Loss
        # ==================================================
        total_loss = reg_loss + conf_loss + flow_loss + traj_loss

        self.loss_dict.update({
            'total_loss{}'.format(prefix): total_loss,
            'reg_loss{}'.format(prefix): reg_loss,
            'conf_loss{}'.format(prefix): conf_loss,
            'flow_loss{}'.format(prefix): flow_loss,
            'traj_loss{}'.format(prefix): traj_loss
        })

        return total_loss

    def compute_flow_loss(self, output_dict, target_dict, prefix=''):
        """
        计算 Flow 分支的 Loss (参考 PointPillarFlowLoss)
        """
        # 1. 检查是否有 flow 数据
        if prefix != '': 
            return torch.tensor(0.0).to(output_dict['psm'].device)
        if 'ffnet_loss_data' not in output_dict:
            return torch.tensor(0.0).to(output_dict['psm'].device)

        ffnet_data = output_dict['ffnet_loss_data']
        # 确保字典里有需要的数据
        if 'flow_pred' not in ffnet_data or 'flow_gt' not in ffnet_data:
            return torch.tensor(0.0).to(output_dict['psm'].device)

        pred_feat = ffnet_data['flow_pred']
        target_feat = ffnet_data['flow_gt']
        
        B, C, H, W = pred_feat.shape 

        # 2. 获取 Mask
        pos_mask = target_dict['pos_equal_one']
        if len(pos_mask.shape) == 3:
            mask_object = pos_mask.unsqueeze(1)
        elif len(pos_mask.shape) == 2:
            mask_object = pos_mask.view(B, 1, H, W)
        else:
            mask_object = (torch.abs(target_feat).sum(dim=1, keepdim=True) > 1e-2).float()

        mask_lidar = (torch.abs(target_feat).sum(dim=1, keepdim=True) > 1e-4).float()
        
        # 组合 Mask
        final_mask = mask_lidar * 0.1 + mask_object * 0.9 
        
        loss_pixel = F.smooth_l1_loss(pred_feat, target_feat, beta=1.0, reduction='none')
        loss_masked = loss_pixel * final_mask
        
        num_valid = mask_lidar.sum() * C + 1e-6
        flow_loss = loss_masked.sum() / num_valid
        
        return flow_loss * self.flow_coe

    def compute_trajectory_loss(self, output_dict, target_dict):
        """
        计算轨迹预测 Loss (参考 PointPillarMotionLoss)
        """
        if 'traj_preds' not in output_dict:
            return torch.tensor(0.0).to(output_dict['psm'].device)
        
        preds = output_dict['traj_preds']
        
        targets = None
        mask = None
        
        # 查找真值
        if 'ego' in target_dict and 'object_traj' in target_dict['ego']:
            targets = target_dict['ego']['object_traj']
            mask = target_dict['ego']['object_traj_mask']
        elif 'object_traj' in target_dict:
            targets = target_dict['object_traj']
            mask = target_dict['object_traj_mask']
            
        if targets is None or mask is None:
            return torch.tensor(0.0).to(preds.device)

        targets = targets.to(preds.device)
        mask = mask.to(preds.device)

        loss = F.smooth_l1_loss(preds, targets, reduction='none')
        loss = loss * mask.unsqueeze(-1)

        valid_elements = mask.sum() * 2 + 1e-6 
        traj_loss = loss.sum() / valid_elements
        return traj_loss * self.traj_coe

    # Cls Loss 函数 (保持不变)
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
        打印所有 Loss
        """
        total_loss = [v.item() for k, v in self.loss_dict.items() if 'total_loss' in k]
        if len(total_loss) > 1:
            total_loss = sum(total_loss)
        else:
            total_loss = total_loss[0] if isinstance(total_loss, list) and len(total_loss) > 0 else total_loss
            
        print_msg = "[epoch {}][{}/{}], || Loss: {:.2f} ||".format(epoch, batch_id + 1, batch_len, total_loss)
        # 自动遍历字典打印所有子 Loss
        for k, v in self.loss_dict.items():
            if 'total_loss' not in k: # total loss 已经打在最前面了
                print_msg += '{}: {:.2f} | '.format(k.replace('_loss', '').replace('_single', ''), v.item())

        print(print_msg)

        if writer is not None:
            for k, v in self.loss_dict.items():
                writer.add_scalar(k, v.item(), epoch * batch_len + batch_id)