import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedSmoothL1Loss(nn.Module):
    """
    (保持不变) Code-wise Weighted Smooth L1 Loss...
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


class PointPillarMotionLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarMotionLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        self.cls_weight = args['cls_weight']
        self.reg_coe = args['reg']
        
        # 【新增 1】轨迹 Loss 的权重系数
        # 建议在 yaml 里加一个 'traj_coe: 0.2'，如果没加默认用 0.2
        self.traj_coe = args['traj']
        self.loss_dict = {}
        self.use_dir = False

    def forward(self, output_dict, target_dict, prefix=''):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        rm = output_dict['rm{}'.format(prefix)]  # [B, 14, 50, 176]
        psm = output_dict['psm{}'.format(prefix)] # [B, 2, 50, 176]
        targets = target_dict['targets']

        # ... (原有的 Classification Loss 计算逻辑，保持不变) ...
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
        cls_targets = box_cls_labels
        cls_targets = cls_targets.unsqueeze(dim=-1)
        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), 2,
            dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds,
                                          one_hot_targets,
                                          weights=cls_weights) 
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # ... (原有的 Regression Loss 计算逻辑，保持不变) ...
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm, targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin,
                                          reg_targets_sin,
                                          weights=reg_weights)
        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        # ==================================================
        # 【新增 2】 计算轨迹预测 Loss
        # ==================================================
        traj_loss = self.compute_trajectory_loss(output_dict, target_dict)
        
        # 将轨迹 Loss 加到总 Loss 中
        total_loss = reg_loss + conf_loss + traj_loss

        # 更新 loss_dict 以便 Logging 显示
        self.loss_dict.update({'total_loss{}'.format(prefix): total_loss,
                               'reg_loss{}'.format(prefix): reg_loss,
                               'conf_loss{}'.format(prefix): conf_loss,
                               'traj_loss{}'.format(prefix): traj_loss}) # 记录它

        return total_loss

    def compute_trajectory_loss(self, output_dict, target_dict):
        """
        计算轨迹预测的 Smooth L1 Loss，并应用 Mask
        """
        # 1. 检查是否有预测结果
        if 'traj_preds' not in output_dict:
            print("\n[DEBUG-Loss] ❌ output_dict 里根本没有 'traj_preds'！")
            return torch.tensor(0.0).to(output_dict['psm'].device)
        
        preds = output_dict['traj_preds'] 
        
        # 2. 安全地获取真值 (Targets)
        # ----------------- [核心修改] -----------------
        # 不要抛错，如果找不到真值，说明这是在算单车 Loss，直接跳过即可
        targets = None
        mask = None
        
        # 优先查找 ego 下的数据 (适配 train.py 的结构)
        if 'ego' in target_dict and 'object_traj' in target_dict['ego']:
            targets = target_dict['ego']['object_traj']
            mask = target_dict['ego']['object_traj_mask']
        # 兼容直接在 dict 里的情况 (适配刚才我们手动塞进去的情况)
        elif 'object_traj' in target_dict:
            targets = target_dict['object_traj']
            mask = target_dict['object_traj_mask']
            
        # 如果还是找不到，说明当前是在计算 single_v 或 single_i 的辅助 loss
        # 此时不需要计算轨迹 loss，直接返回 0
        if targets is None or mask is None:
            return torch.tensor(0.0).to(preds.device)
        # ----------------- [修改结束] -----------------
        # ========== 【新增：检查 Mask 到底是不是全 0】 ==========
        valid_count = mask.sum().item()
        if valid_count == 0:
            print("\n[DEBUG-Loss] ⚠️ Mask 全是 0！Loss 被乘成了 0。说明这一帧没有任何轨迹真值。")
        # ========================================================
        # 3. 确保设备一致
        targets = targets.to(preds.device)
        mask = mask.to(preds.device)

        # 4. 计算 Smooth L1 Loss
        loss = F.smooth_l1_loss(preds, targets, reduction='none')

        # 5. 应用 Mask
        loss = loss * mask.unsqueeze(-1)

        # 6. 归一化
        valid_elements = mask.sum() * 2 + 1e-6 
        traj_loss = loss.sum() / valid_elements

        return traj_loss * self.traj_coe

    def cls_loss_func(self, input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        # ... (保持不变) ...
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)
        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)
        loss = focal_weight * bce_loss
        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)
        assert weights.shape.__len__() == loss.shape.__len__()
        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2


    def logging(self, epoch, batch_id, batch_len, writer = None):
        """
        Print out  the loss function for current iteration.
        """
        total_loss = [v.item() for k, v in self.loss_dict.items() if 'total_loss' in k]
        if len(total_loss) > 1:
            total_loss = sum(total_loss)
        else:
            total_loss = total_loss[0]
            
        # 这里的 loop 会自动把 loss_dict 里的 traj_loss 也打印出来
        # 所以不需要改 print_msg 的构建逻辑
        
        print_msg = "[epoch {}][{}/{}], || Loss: {:.2f} ||".format(epoch, batch_id + 1, batch_len, total_loss)
        for k, v in self.loss_dict.items():
            print_msg += '{}: {:.2f} | '.format(k.replace('_loss', '').replace('_single', ''), v.item())

        if self.use_dir:
            dir_loss = self.loss_dict['dir_loss']
            print_msg += " || Dir Loss: %.4f" % dir_loss.item()

        print(print_msg)

        if not writer is None:
            for k, v in self.loss_dict.items():
                writer.add_scalar(k, v.item(), epoch*batch_len + batch_id)
            if self.use_dir:
                writer.add_scalar('dir_loss', dir_loss.item(),
                            epoch*batch_len + batch_id)