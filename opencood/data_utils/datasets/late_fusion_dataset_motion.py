# -*- coding: utf-8 -*-
# Author: Quanhao Li <quanhaoli2022@163.com> Yifan Lu <yifan_lu@sjtu.edu.cn>, 
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for late fusion
"""
import random
import math
from collections import OrderedDict
import os
import opencood.data_utils.post_processor as post_processor
import numpy as np
import torch
from torch.utils.data import Dataset
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils
import json
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.transformation_utils import tfm_to_pose
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.common_utils import read_json
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.data_utils.datasets.basedataset import BaseDataset
def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data
class LateFusionDatasetMotion(BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    detection outputs to ego.
    """
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)
        self.max_cav = 2
        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        assert 'proj_first' in params['fusion']['args']
        if params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False


        assert 'clip_pc' in params['fusion']['args']
        if params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False
        
        if 'select_kp' in params:
            self.select_keypoint = params['select_kp']
        else:
            self.select_keypoint = None

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        if self.train:
            split_dir = params['root_dir']
            traj_json_path = params['traj_train']
        else:
            split_dir = params['validate_dir']
            traj_json_path = params['traj_test']

        self.root_dir = params['data_dir']

        self.split_info = load_json(split_dir)
        co_datainfo = load_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        self.co_data = OrderedDict()
        # =========================================================
        # 【修改】无条件加载轨迹数据库 (Train & Test 通用)
        # =========================================================
        self.pred_len = 5
        self.traj_database = {}       
        if os.path.exists(traj_json_path):
            print(f"[Dataset] (Debug Mode) Loading trajectory labels from {traj_json_path} ...")
            self.traj_database = load_json(traj_json_path)
            print(f"[Dataset] Trajectory labels loaded. Count: {len(self.traj_database)}")
        else:
            print(f"[Dataset] Warning: {traj_json_path} not found. Trajectory prediction will be skipped.")
            raise FileNotFoundError(f"{traj_json_path} not found.")
        ################################################################################
        for frame_info in co_datainfo:
            veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            self.co_data[veh_frame_id] = frame_info

    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)
        if self.train:
            reformat_data_dict = self.get_item_train(base_data_dict)
        else:
            reformat_data_dict = self.get_item_test(base_data_dict, idx)

        return reformat_data_dict

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        veh_frame_id = self.split_info[idx]
        frame_info = self.co_data[veh_frame_id]
        # system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()
        data[0] = OrderedDict() 
        data[0]['ego'] = True
        data[1] = OrderedDict()
        data[1]['ego'] = False
 
        data[0]['params'] = OrderedDict()
        data[0]['params']['vehicles'] = load_json(os.path.join(self.root_dir, 'vehicle-side/label/lidar/{}.json'.format(veh_frame_id)))['objects']

        vehicle_pose = load_json(os.path.join(self.root_dir,'vehicle-side/label/lidar/'+str(veh_frame_id)+'.json'))
        vehicle_sensor_pose = vehicle_pose['sensor_pose']
        data[0]['params']['lidar_pose'] = [vehicle_sensor_pose['x'], vehicle_sensor_pose['y'], vehicle_sensor_pose['z'],
                                           vehicle_sensor_pose['roll'], vehicle_sensor_pose['yaw'], vehicle_sensor_pose['pitch']]

        vehicle_lidar_path = os.path.join(self.root_dir, 'vehicle-side/velodyne/{}.bin'.format(veh_frame_id))
        data[0]['lidar_np'], _ = pcd_utils.read_bin(vehicle_lidar_path)
                # motion
        if veh_frame_id in self.traj_database:
            data[0]['traj_gt_record'] = self.traj_database[veh_frame_id]
        else:
            data[0]['traj_gt_record'] = None
        
        if self.clip_pc:
            data[0]['lidar_np'] = data[0]['lidar_np'][data[0]['lidar_np'][:,0]>0]

        data[1]['params'] = OrderedDict()
        data[1]['params']['vehicles'] = load_json(os.path.join(self.root_dir, 'infrastructure-side/label/virtuallidar/{}.json'.format(veh_frame_id)))['objects']
        infra_pose = load_json(os.path.join(self.root_dir,'infrastructure-side/label/virtuallidar/'+str(veh_frame_id)+'.json'))
        infra_sensor_pose = infra_pose['sensor_pose']
        data[1]['params']['lidar_pose'] = [infra_sensor_pose['x'], infra_sensor_pose['y'], infra_sensor_pose['z'],
                                           infra_sensor_pose['roll'], infra_sensor_pose['yaw'], infra_sensor_pose['pitch']]
    
        infra_lidar_path = os.path.join(self.root_dir, 'infrastructure-side/velodyne/{}.bin'.format(veh_frame_id))
        data[1]['lidar_np'], _ = pcd_utils.read_bin(infra_lidar_path)
        
        # data[1]['params']['vehicles'] = load_json(os.path.join(self.root_dir, 'vehicle-side/label/lidar/{}.json'.format(veh_frame_id)))['objects']

        # vehicle_pose = load_json(os.path.join(self.root_dir,'vehicle-side/label/lidar/'+str(veh_frame_id)+'.json'))
        # vehicle_sensor_pose = vehicle_pose['sensor_pose']
        # data[1]['params']['lidar_pose'] = [vehicle_sensor_pose['x'], vehicle_sensor_pose['y'], vehicle_sensor_pose['z'],
        #                                    vehicle_sensor_pose['roll'], vehicle_sensor_pose['yaw'], vehicle_sensor_pose['pitch']]

        # vehicle_lidar_path = os.path.join(self.root_dir, 'vehicle-side/velodyne/{}.bin'.format(veh_frame_id))
        # data[1]['lidar_np'], _ = pcd_utils.read_bin(vehicle_lidar_path)

        return data

    def get_item_single_car(self, selected_cav_base):
        """
        Process a single CAV's information for the train/test pipeline.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        # remove points that hit ego vehicle
        lidar_np = mask_ego_points(lidar_np)
        # generate the bounding box(n, 7) under the cav's space
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                    selected_cav_base[
                                                           'params'][
                                                           'lidar_pose_clean'])

        # data augmentation
        # lidar_np, object_bbx_center, object_bbx_mask = \
        #     self.augment(lidar_np, object_bbx_center, object_bbx_mask)
        if self.visualize:
            selected_cav_processed.update({'origin_lidar': lidar_np})

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(lidar_np)
        selected_cav_processed.update({'processed_lidar': lidar_dict})

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()
        selected_cav_processed.update({'anchor_box': anchor_box})

        selected_cav_processed.update({'object_bbx_center': object_bbx_center,
                                       'object_bbx_mask': object_bbx_mask,
                                       'object_ids': object_ids})

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=object_bbx_mask)
        selected_cav_processed.update({'label_dict': label_dict})
        traj_record = selected_cav_base.get('traj_gt_record')
        max_num = self.params['postprocess']['max_num']

        gt_traj = np.zeros((max_num, self.pred_len, 2), dtype=np.float32)
        gt_traj_mask = np.zeros((max_num, self.pred_len), dtype=np.float32)

        if traj_record is not None:
            saved_ids = traj_record['gt_ids']
            saved_trajs = traj_record['gt_traj']
            saved_masks = traj_record['gt_traj_mask']

            id_to_idx = {str(uid): i for i, uid in enumerate(saved_ids)}
            # ========== 【新增：观察到底能不能匹配上】 ==========
            match_count = 0
            for k, query_id in enumerate(object_ids):
                if k >= max_num: break
                if str(query_id) in id_to_idx:
                    src_idx = id_to_idx[str(query_id)]
                    gt_traj[k] = saved_trajs[src_idx]
                    gt_traj_mask[k] = saved_masks[src_idx]
                    match_count += 1
            
            if match_count == 0 and len(object_ids) > 0:
                print(f"\n[DEBUG-Dataset] ❌ 完蛋，一个都没匹配上！")
                print(f"   当前帧检测到的 object_ids (从单帧标签读取): {object_ids}")
                print(f"   当前帧轨迹库里的 saved_ids (从轨迹库读取): {saved_ids}")
            # ====================================================
            for k, query_id in enumerate(object_ids):
                if k >= max_num:
                    break
                if str(query_id) in id_to_idx:
                    src_idx = id_to_idx[str(query_id)]
                    gt_traj[k] = saved_trajs[src_idx]
                    gt_traj_mask[k] = saved_masks[src_idx]

        selected_cav_processed.update({
            'object_traj': gt_traj,
            'object_traj_mask': gt_traj_mask
        })
        return selected_cav_processed

    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Notice: it is a wrap of postprocessor function

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list, no use.
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        return self.post_processor.generate_object_center_carla_late_fusion_with_traj_id(cav_contents) 
        
    def get_item_train(self, base_data_dict):
        processed_data_dict = OrderedDict()
        base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])
        # during training, we return a random cav's data
        # only one vehicle is in processed_data_dict
        if not self.visualize:
            # selected_cav_id, selected_cav_base = \
            #     random.choice(list(base_data_dict.items()))
            selected_cav_id, selected_cav_base = \
                list(base_data_dict.items())[0]
        else:
            selected_cav_id, selected_cav_base = \
                list(base_data_dict.items())[0]
        selected_cav_processed = self.get_item_single_car(selected_cav_base)
        processed_data_dict.update({'ego': selected_cav_processed})

        return processed_data_dict

    def get_item_test(self, base_data_dict, idx):
        """
            processed_data_dict.keys() = ['ego', "650", "659", ...]
        """
        base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

        processed_data_dict = OrderedDict()
        ego_id = -1
        ego_lidar_pose = []
        cav_id_list = []
        lidar_pose_list = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            # if distance > self.params['comm_range']:
            #     continue
            cav_id_list.append(cav_id)
            lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])

        for cav_id in cav_id_list:
            selected_cav_base = base_data_dict[cav_id]
            # find the transformation matrix from current cav to ego.
            cav_lidar_pose = selected_cav_base['params']['lidar_pose']
            transformation_matrix = x1_to_x2(cav_lidar_pose, ego_lidar_pose)
            cav_lidar_pose_clean = selected_cav_base['params']['lidar_pose_clean']
            transformation_matrix_clean = x1_to_x2(cav_lidar_pose_clean, ego_lidar_pose_clean)

            selected_cav_processed = \
                self.get_item_single_car(selected_cav_base)
            selected_cav_processed.update({'transformation_matrix': transformation_matrix,
                                           'transformation_matrix_clean': transformation_matrix_clean})
            update_cav = "ego" if cav_id == ego_id else cav_id
            processed_data_dict.update({update_cav: selected_cav_processed})

        return processed_data_dict
    def collate_batch_train(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for early and late fusion dataset. (Added Trajectory Support)
        """
        # during training, we only care about ego.
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        label_dict_list = []
        
        # ==================== 【新增 1：初始化轨迹与 ID 列表】 ====================
        object_ids = []
        object_traj_list = []
        object_traj_mask_list = []
        # ========================================================================

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            label_dict_list.append(ego_dict['label_dict'])

            # ==================== 【新增 2：加入批次列表】 ====================
            object_ids.append(ego_dict.get('object_ids', []))
            object_traj_list.append(ego_dict['object_traj'])
            object_traj_mask_list.append(ego_dict['object_traj_mask'])
            # ================================================================

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # ==================== 【新增 3：转换为 Tensor】 ====================
        object_traj = torch.from_numpy(np.array(object_traj_list))
        object_traj_mask = torch.from_numpy(np.array(object_traj_mask_list))
        # ==================================================================

        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(processed_lidar_list)
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)
            
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   # 严格保留你原本处理 anchor_box 的方式
                                   'anchor_box': torch.from_numpy(ego_dict['anchor_box']), 
                                   'label_dict': label_torch_dict,
                                   # ==================== 【新增 4：更新到最终输出字典】 ====================
                                   'object_ids': object_ids[0] if len(object_ids) > 0 else [], 
                                   'object_traj': object_traj,
                                   'object_traj_mask': object_traj_mask
                                   # ======================================================================
                                   })
                                   
        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict
    def collate_batch_test(self, batch):
        """
        Customized collate function for pytorch dataloader during testing
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # currently, we only support batch size of 1 during testing
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]

        output_dict = {}

        # for late fusion, we also need to stack the lidar for better
        # visualization
        if self.visualize:
            projected_lidar_list = []
            origin_lidar = []

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            # shape: (1, max_num, 7)
            object_bbx_center = \
                torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = \
                torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']
            # ========== 【修复 2：加入轨迹 Tensor 转换】 ==========
            object_traj = \
                torch.from_numpy(np.array([cav_content['object_traj']]))
            object_traj_mask = \
                torch.from_numpy(np.array([cav_content['object_traj_mask']]))
            # ====================================================

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box':
                    torch.from_numpy(np.array(
                        cav_content[
                            'anchor_box']))})
            if self.visualize:
                transformation_matrix = cav_content['transformation_matrix']
                origin_lidar = [cav_content['origin_lidar']]

                if (self.params['only_vis_ego'] is False) or (cav_id=='ego'):
                    import copy
                    projected_lidar = copy.deepcopy(cav_content['origin_lidar'])
                    projected_lidar[:, :3] = \
                        box_utils.project_points_by_matrix_torch(
                            projected_lidar[:, :3],
                            transformation_matrix)
                    projected_lidar_list.append(projected_lidar)

            # processed lidar dictionary
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(
                    [cav_content['processed_lidar']])
            # label dictionary
            label_torch_dict = \
                self.post_processor.collate_batch([cav_content['label_dict']])

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = \
                torch.from_numpy(
                    np.array(cav_content['transformation_matrix'])).float()
            
            # late fusion training, no noise
            transformation_matrix_clean_torch = transformation_matrix_torch

            output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,
                                        'object_traj': object_traj,           # <--- 加上
                                        'object_traj_mask': object_traj_mask, # <--- 加上
                                        'processed_lidar': processed_lidar_torch_dict,
                                        'label_dict': label_torch_dict,
                                        'object_ids': object_ids,
                                        'transformation_matrix': transformation_matrix_torch,
                                        'transformation_matrix_clean': transformation_matrix_clean_torch})

            if self.visualize:
                origin_lidar = \
                    np.array(
                        downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({'origin_lidar': origin_lidar})

        if self.visualize:
            projected_lidar_stack = [torch.from_numpy(
                np.vstack(projected_lidar_list))]
            output_dict['ego'].update({'origin_lidar': projected_lidar_stack})

            output_dict['ego'].update({'origin_lidar_v':
                    [torch.from_numpy(projected_lidar_list[0])]})
            output_dict['ego'].update({'origin_lidar_i':
                    [torch.from_numpy(projected_lidar_list[1])]})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        
        The object id can not used for identifying the same object.
        here we will to use the IoU to determine it.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.
        output_dict :dict
            The dictionary containing the output of the model.
        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx_by_iou(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def post_process_no_fusion(self, data_dict, output_dict_ego):
        """
        The object id can not used for identifying the same object.
        here we will to use the IoU to determine it.
        """
        data_dict_ego = OrderedDict()
        data_dict_ego['ego'] = data_dict['ego']
        gt_box_tensor = self.post_processor.generate_gt_bbx_by_iou(data_dict)

        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict_ego, output_dict_ego)
        return pred_box_tensor, pred_score, gt_box_tensor
    
    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)
        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask