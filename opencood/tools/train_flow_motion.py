# opencood/tools/train_ffnet.py
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yue Hu <18671129361@sjtu.edu.cn>
# Modified for FFNet/Flow Integration

import argparse
import os
import statistics
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--pretrained_path', default='', # opencood/ffnet_weight/latest.pth
                        help='Path to the pretrained model (Stage 1 checkpoint)')
    opt = parser.parse_args()
    return opt

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

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

    if torch.cuda.is_available():
        model.to(device)

    # 打印参数检查清单 (保持你原来的逻辑)
    print("\n" + "="*50)
    print("🔍  TRAINABLE PARAMETERS CHECKLIST  🔍")
    print("="*50)
    trainable_count = 0
    frozen_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(f"✅ [Update] {name}  (Size: {param.shape})") # 嫌太长可以注释掉
            trainable_count += 1
        else:
            frozen_count += 1
    print("-" * 50)
    print(f"Total Trainable Layers: {trainable_count}")
    print(f"Total Frozen Layers:    {frozen_count}")
    print("="*50 + "\n")

    # 加载预训练权重逻辑
    if opt.pretrained_path:
        print(f"Loading pretrained weights from: {opt.pretrained_path}")
        checkpoint = torch.load(opt.pretrained_path, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
        if not isinstance(pretrained_dict, dict):
             pretrained_dict = checkpoint
        load_dict = {k: v for k, v in pretrained_dict.items() 
                     if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(load_dict)
        model.load_state_dict(model_dict)
        print("Pretrained weights loaded successfully (Strict=False)!")

    # ==================== 【改动 2】 使用新 Loss 初始化 ====================
    criterion = train_utils.create_loss(hypes)
    # =====================================================================

    optimizer = train_utils.setup_optimizer(hypes, model)
    
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
            
        for i, batch_data in enumerate(train_loader):
            if batch_data is None:
                continue
            
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            batch_data['ego']['max_epoch'] = hypes['train_params']['epoches'] - 5
            output_dict = model(batch_data['ego'])
            # ================= 【你缺失的部分】 =================
            # 必须手动把轨迹真值塞进 label_dict，否则 Loss 函数看不见！
            if 'object_traj' in batch_data['ego']:
                batch_data['ego']['label_dict']['object_traj'] = batch_data['ego']['object_traj']
                batch_data['ego']['label_dict']['object_traj_mask'] = batch_data['ego']['object_traj_mask']
            # ===================================================
            # ==================== 【改动 3】 极简的 Loss 计算 ====================
            # 这一行现在同时计算：Detection Loss + Flow Loss (带 Mask)
            final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
            # ===================================================================

            # 辅助 Loss (单车视角)
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
                                round_loss_v += criterion(output_dict, batch_data['ego']['label_dict'], prefix='_v{}'.format(round_id))
                            final_loss += round_loss_v
            criterion.logging(epoch, i, len(train_loader), writer)

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
                    model.eval() # 别忘了 eval 模式

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    output_dict = model(batch_data['ego'])

                    # 验证集也使用同样的 Loss 标准
                    final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
                    
                    # 辅助 Loss (保持一致)
                    if len(output_dict) > 2:
                        if 'label_dict_single_v' in batch_data['ego']:
                            single_loss_v = criterion(output_dict, batch_data['ego']['label_dict_single_v'], prefix='_single_v')
                            final_loss += single_loss_v
                        if 'label_dict_single_i' in batch_data['ego']:
                            single_loss_i = criterion(output_dict, batch_data['ego']['label_dict_single_i'], prefix='_single_i')
                            final_loss += single_loss_i

                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch, valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))
        
        scheduler.step(epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)
    torch.cuda.empty_cache()
    
    # 自动推理逻辑
    run_test = True
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method intermediate_with_comm"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()