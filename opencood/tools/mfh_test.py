import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from opencood.utils import box_utils
# 引入必要的库 (假设脚本放在 opencood/tools/ 下)
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import simple_vis
from opencood.utils import common_utils

def test_parser():
    parser = argparse.ArgumentParser(description="Check GT Data Loading")
    parser.add_argument('--model_dir', type=str, default='opencood/hypes_yaml/v2x-seq/dair_where2comm_max_multiscale_resnet.yaml',
                        help='Model directory containing config.yaml')
    return parser.parse_args()

def main():
    opt = test_parser()
    
    # 1. 加载配置文件
    # 这一步非常关键，确保读取的是你训练时的配置
    file= str(opt.model_dir)
    hypes = yaml_utils.load_yaml(file)
    
    # 强制把 validate_dir 指向 test_dir，确保加载的是测试集数据
    if 'test_dir' in hypes:
        hypes['validate_dir'] = hypes['test_dir']
    
    print(f"Dataset Dir: {hypes['validate_dir']}")

    # 2. 构建数据集 (train=False 表示加载测试/验证模式)
    print('Building Dataset...')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    
    # 3. 创建 DataLoader
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1, # 只能为1，方便可视化
                             num_workers=2,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    # 4. 准备保存路径
    save_path = os.path.join('/home/yty/mfh/code/inter/Where2comm/opencood/logs', 'vis_gt_check_bev')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"Saving visualization results to: {save_path}")

    # 获取配置中的参数
    left_hand = True if "OPV2V" in hypes['test_dir'] else False
    gt_range = hypes['postprocess']['gt_range'] # [x_min, y_min, z_min, x_max, y_max, z_max]
    box_order = hypes['postprocess']['order']
    # 5. 循环读取数据并画图
    for i, batch_data in tqdm(enumerate(data_loader)):
        if i >= 50000: # 只画前 500 帧检查，够用了
            break

        # ================== 核心：提取数据 ==================
        # batch_data['ego']['origin_lidar'] 是一个 list，取第一个元素
        pcd = batch_data['ego']['origin_lidar'][0]
        
        # 提取 GT Boxes (Ego 坐标系下)
        # shape: (1, Max_Num, 7) -> (Max_Num, 7)
        gt_boxes = batch_data['ego']['object_bbx_center'][0]
        gt_mask = batch_data['ego']['object_bbx_mask'][0]
        # 1. 检查 Mask 前的数量
        print(f"\n[Frame {i}] Raw boxes count: {len(gt_boxes)}")
        # 过滤掉填充的无效框 (Padding)
        # 这一步非常重要！否则你会看到原点有一堆乱七八糟的框
        if gt_mask is not None:
            gt_boxes = gt_boxes[gt_mask == 1]
        # ================== 核心修复 ==================
        print(f"\n[Frame {i}]  boxes count: {len(gt_boxes)}")
        # 3. 将 (N, 7) 转换为 (N, 8, 3) 的角点格式
        if len(gt_boxes) > 0:
            # box_utils.boxes_to_corners_3d 支持 tensor 输入，返回 tensor
            gt_corners = box_utils.boxes_to_corners_3d(gt_boxes, box_order)
        else:
            gt_corners = torch.zeros((0, 8, 3))
        # =============================================
        # ================== Debug：打印坐标中心 ==================
        # 如果点云均值和GT均值差很远，说明坐标系没对齐
        pcd_center = pcd[:, :3].mean(dim=0).cpu().numpy() if isinstance(pcd, torch.Tensor) else pcd[:, :3].mean(axis=0)
        gt_center = gt_boxes[:, :3].mean(dim=0).cpu().numpy() if len(gt_boxes) > 0 else "No GT"
        # print(f"Frame {i} | PCD Center: {pcd_center} | GT Center: {gt_center}")

        # ================== 可视化 ==================
        vis_file_path = os.path.join(save_path, 'bev_gt_%05d.png' % i)
        
        # 调用 simple_vis.visualize
        # 注意：这里我们把 vis_pred_box 设为 False，因为只看 GT
        simple_vis.visualize(pred_box_tensor=None, # 没有预测框
                             gt_tensor=gt_corners,   # 只有 GT 框
                             pcd=pcd,
                             pc_range=gt_range,
                             save_path=vis_file_path,
                             method='bev',         # 强制 BEV 视图
                             left_hand=left_hand,
                             vis_gt_box=True,      # 画 GT
                             vis_pred_box=False)   # 不画预测

    print("Done! Please check the folder:", save_path)

if __name__ == '__main__':
    main()