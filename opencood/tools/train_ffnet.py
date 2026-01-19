# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yue Hu <18671129361@sjtu.edu.cn>
# Modified for FFNet Integration

import argparse
import os
import statistics
import torch
import torch.nn.functional as F  # <--- [新增] 用于计算 MSE Loss
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--pretrained_path', default='opencood/ffnet_weight/latest.pth', 
                        help='Path to the pretrained model (Stage 1 checkpoint)')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=8,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=8,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    # ================= 【新增】加载预训练权重逻辑 =================
    if opt.pretrained_path:
        print(f"Loading pretrained weights from: {opt.pretrained_path}")
        # 加载 checkpoint
        checkpoint = torch.load(opt.pretrained_path, map_location='cpu')
        
        # 处理 state_dict (有些保存时带 'module.' 前缀，需要去掉)
        model_dict = model.state_dict()
        pretrained_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
        
        # 如果直接是 state_dict
        if not isinstance(pretrained_dict, dict):
             # 有些版本保存结构不同，做个容错，通常直接 torch.load 出来的就是 dict
             pretrained_dict = checkpoint

        # 【核心逻辑】筛选 key
        # 1. 过滤掉不匹配的 key (比如 FlowGenerator 就在预训练里没有)
        # 2. 过滤掉形状不匹配的 key
        load_dict = {k: v for k, v in pretrained_dict.items() 
                     if k in model_dict and v.shape == model_dict[k].shape}
        
        # 打印一下加载了多少层，心里有底
        print(f"Matched {len(load_dict)} / {len(model_dict)} layers.")
        # 【新增验证】打印出哪些层没有被加载
        missing_keys = [k for k in model_dict.keys() if k not in load_dict]
        print("\n=== Missing Keys (Should be flow_generator) ===")
        # 只打印前10个看看样子
        for k in missing_keys[:10]:
            print(k)
        print(f"Total missing: {len(missing_keys)}\n")
        # 更新当前模型权重
        model_dict.update(load_dict)
        model.load_state_dict(model_dict)
        
        print("Pretrained weights loaded successfully (Strict=False)!")
    # =============================================================
    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        # lr scheduler setup
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate
    with_round_loss = False
    
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
            
        for i, batch_data in enumerate(train_loader):
            if batch_data is None:
                continue
            
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            
            batch_data['ego']['epoch'] = epoch
            
            # 模型前向传播
            output_dict = model(batch_data['ego'])
            
            # 1. 计算主要的检测 Loss (Detection Loss)
            # 即使 Backbone 冻结了，我们也需要这个 Loss 来通过 Head 监督 FlowNet
            final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
            
            # 2. [新增] 计算 FFNet 的相似度 Loss (MSE Loss)
            # -----------------------------------------------------------
            if 'ffnet_loss_data' in output_dict:
                ffnet_data = output_dict['ffnet_loss_data']
                if 'flow_pred' in ffnet_data and 'flow_gt' in ffnet_data:
                    # 使用 MSE Loss 强迫预测特征接近真实特征
                    loss_ffnet = F.mse_loss(ffnet_data['flow_pred'], ffnet_data['flow_gt'])
                    
                    # 权重系数，建议设为 1.0，可视 Tensorboard 情况调整
                    ffnet_loss_weight = 1.0 
                    final_loss += ffnet_loss_weight * loss_ffnet
                    
                    # 记录到 Tensorboard
                    writer.add_scalar('Train_FFNet_Loss', loss_ffnet.item(), epoch * len(train_loader) + i)
            # -----------------------------------------------------------

            # 3. 计算辅助 Loss (Single Loss)
            if len(output_dict) > 2:
                # 注意：如果你的 output_dict 里有很多杂项，这里最好用 key 判断，或者 trust 原来的逻辑
                if 'label_dict_single_v' in batch_data['ego']:
                    single_loss_v = criterion(output_dict, batch_data['ego']['label_dict_single_v'], prefix='_single_v')
                    final_loss += single_loss_v
                    
                if 'label_dict_single_i' in batch_data['ego']:
                    single_loss_i = criterion(output_dict, batch_data['ego']['label_dict_single_i'], prefix='_single_i')
                    final_loss += single_loss_i

                # Round Loss (多轮通信)
                if 'fusion_args' in hypes['model']['args']:
                    if 'communication' in hypes['model']['args']['fusion_args']:
                        comm = hypes['model']['args']['fusion_args']['communication']
                        if ('round' in comm) and comm['round'] > 1:
                            round_loss_v = 0
                            with_round_loss = True
                            for round_id in range(1, comm['round']):
                                round_loss_v += criterion(output_dict, batch_data['ego']['label_dict'], prefix='_v{}'.format(round_id))
                            final_loss += round_loss_v

            criterion.logging(epoch, i, len(train_loader), writer)
            # =================== 【新增】手动打印 FFNet Loss ===================
            # 为了防止刷屏太快，我们和 logging 保持一致，通常每 10 或 50 步打印一次
            # 如果你不知道 criterion 内部的频率，可以自己设一个，比如 % 50
            if i % 50 == 0 and 'ffnet_loss_data' in output_dict:
                if 'flow_pred' in output_dict['ffnet_loss_data']:
                    # 注意：要在终端显示，必须把 tensor 转为 float (.item())
                    print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]  ||  FFNet Loss: {loss_ffnet.item():.5f}")
            # =================================================================
            # back-propagation
            final_loss.backward()
            optimizer.step()

        # ================= Validation Loop =================
        if epoch % hypes['train_params']['eval_freq'] == 0:
            torch.cuda.empty_cache()
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    output_dict = model(batch_data['ego'])

                    final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
                    
                    # Validation 时也加上 MSE Loss 方便观察
                    if 'ffnet_loss_data' in output_dict:
                        ffnet_data = output_dict['ffnet_loss_data']
                        if 'flow_pred' in ffnet_data and 'flow_gt' in ffnet_data:
                            loss_ffnet = F.mse_loss(ffnet_data['flow_pred'], ffnet_data['flow_gt'])
                            final_loss += 1.0 * loss_ffnet # 保持权重一致

                    if len(output_dict) > 2:
                        if 'label_dict_single_v' in batch_data['ego']:
                            single_loss_v = criterion(output_dict, batch_data['ego']['label_dict_single_v'], prefix='_single_v')
                            final_loss += single_loss_v
                        if 'label_dict_single_i' in batch_data['ego']:
                            single_loss_i = criterion(output_dict, batch_data['ego']['label_dict_single_i'], prefix='_single_i')
                            final_loss += single_loss_i

                        if 'fusion_args' in hypes['model']['args']:
                            if 'communication' in hypes['model']['args']['fusion_args']:
                                comm = hypes['model']['args']['fusion_args']['communication']
                                if ('round' in comm) and comm['round'] > 1:
                                    for round_id in range(1, comm['round']):
                                        round_loss_v = criterion(output_dict, batch_data['ego']['label_dict'], prefix='_v{}'.format(round_id))
                                        final_loss += round_loss_v
                    
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch, valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step(epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)
    torch.cuda.empty_cache()
    # 如果需要在训练结束后自动跑测试，可以打开下面的开关
    run_test = True
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()