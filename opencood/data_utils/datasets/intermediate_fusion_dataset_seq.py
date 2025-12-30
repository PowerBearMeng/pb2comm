import os
import json
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair import IntermediateFusionDatasetDAIR
from opencood.utils.transformation_utils import tfm_to_pose, pose_to_tfm
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.pre_processor import build_preprocessor
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils # 必须导入这个工具

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

class IntermediateFusionDatasetSeq(IntermediateFusionDatasetDAIR):
    """
    针对 V2X-Seq (FFNet/VIC-3D) 数据集的适配类。
    """
    def __init__(self, params, visualize, train=True):
        print("DEBUG: 成功加载了 IntermediateFusionDatasetSeq !!!")  # 👈 加这句
        # 手动初始化，跳过父类中导致报错的 data_info.json 读取部分
        self.params = params
        self.visualize = visualize
        self.train = train
        
        self.data_augmentor = DataAugmentor(params['data_augment'], train)
        
        # 参数初始化
        self.max_cav = 2
        if 'proj_first' in params['fusion']['args'] and params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False

        if "kd_flag" in params.keys():
            self.kd_flag = params['kd_flag']
        else:
            self.kd_flag = False

        if 'clip_pc' in params['fusion']['args'] and params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False
        
        if 'select_kp' in params:
            self.select_keypoint = params['select_kp']
        else:
            self.select_keypoint = None

        self.pre_processor = build_preprocessor(params['preprocess'], train)
        self.post_processor = post_processor.build_postprocessor(params['postprocess'], train)

        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = params['data_dir']
        self.split_info = load_json(split_dir)

    def retrieve_base_data(self, idx):
        frame_info = self.split_info[idx]
        veh_frame_id = frame_info['vehicle_frame']
        infra_frame_id = frame_info['infrastructure_frame']
        data = OrderedDict()
        
        # --- 1. Vehicle Side (Ego) ---
        data[0] = OrderedDict() 
        data[0]['ego'] = True
        data[0]['params'] = OrderedDict()

        # Pose 设为 Identity
        identity_pose = np.eye(4)
        data[0]['params']['lidar_pose'] = tfm_to_pose(identity_pose)
        data[0]['params']['lidar_pose_clean'] = tfm_to_pose(identity_pose)

        # 加载标签
        vehicles_label = []
        if 'cooperative_label_w2v_path' in frame_info:
            label_path = os.path.join(self.root_dir, frame_info['cooperative_label_w2v_path'])
            vehicles_label = self.load_and_format_label(label_path)
        
        data[0]['params']['vehicles'] = vehicles_label
        ######################## Single View GT ########################
        vehicle_side_path = os.path.join(self.root_dir, 'vehicle-side/label/lidar/{}.json'.format(veh_frame_id))
        data[0]['params']['vehicles_single']  = self.load_and_format_label(vehicle_side_path)
        ######################## Single View GT ########################

        # 加载点云
        veh_lidar_path = os.path.join(self.root_dir, frame_info['vehicle_pointcloud_bin_path'])
        data[0]['lidar_np'] = self.load_bin_file(veh_lidar_path)
        
        if self.clip_pc and data[0]['lidar_np'] is not None:
             data[0]['lidar_np'] = data[0]['lidar_np'][data[0]['lidar_np'][:, 0] > 0]

        # --- 2. Infrastructure Side ---
        data[1] = OrderedDict()
        data[1]['ego'] = False
        data[1]['params'] = OrderedDict()

        if 'calib_lidar_i2v_path' in frame_info:
            i2v_path = os.path.join(self.root_dir, frame_info['calib_lidar_i2v_path'])
            i2v_json_content = load_json(i2v_path)
            
            offset = frame_info.get('system_error_offset')
            if offset is None:
                offset = {"delta_x": 0.0, "delta_y": 0.0}
            # offset = {"delta_x": 0.0, "delta_y": 0.0}

            try:
                t_i2v = inf_side_rot_and_trans_to_trasnformation_matrix(i2v_json_content, offset)
            except (IndexError, TypeError):
                # 兼容不同格式
                if isinstance(i2v_json_content['translation'], list):
                    i2v_json_content['translation'] = [[x] for x in i2v_json_content['translation']]
                    t_i2v = inf_side_rot_and_trans_to_trasnformation_matrix(i2v_json_content, offset)
                else:
                    raise ValueError(f"Unknown translation format in {i2v_path}")
            data[1]['params']['lidar_pose'] = tfm_to_pose(t_i2v)
            data[1]['params']['lidar_pose_clean'] = tfm_to_pose(t_i2v)
        else:
            return None 
        
        data[1]['params']['vehicles'] = []
         ######################## Single View GT ########################
        infra_side_path = os.path.join(self.root_dir, 'infrastructure-side/label/virtuallidar/{}.json'.format(infra_frame_id))
        data[1]['params']['vehicles_single'] = self.load_and_format_label(infra_side_path)
        ######################## Single View GT ########################
    
        inf_lidar_path = os.path.join(self.root_dir, frame_info['infrastructure_pointcloud_bin_path'])
        data[1]['lidar_np'] = self.load_bin_file(inf_lidar_path)

        return data

    def load_bin_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        if np.isnan(points).any():
            points = points[~np.isnan(points).any(axis=1)]
        return points

    def load_and_format_label(self, path):
        raw_labels = load_json(path)
        formatted_labels = []
        for obj in raw_labels:
            if obj['type'] not in ['Car', 'Van', 'Truck', 'Bus']:
                continue
            new_obj = {}
            loc = obj['3d_location']
            new_obj['location'] = [loc['x'], loc['y'], loc['z']]
            dim = obj['3d_dimensions']
            new_obj['dimensions'] = [dim['h'], dim['w'], dim['l']]
            new_obj['angle'] = obj['rotation']
            new_obj['type'] = obj['type']
            new_obj['id'] = obj.get('vehicle_id', -1)
            formatted_labels.append(new_obj)
        return formatted_labels

    # ======================================================================
    # 【核心修复】: 必须在这里覆盖这两个函数，防止调用父类导致崩溃
    # ======================================================================
    def generate_object_center(self, cav_contents, reference_lidar_pose, return_visible_mask=False):
        """
        覆盖父类方法，直接处理 V2X-Seq 格式的标签，不再调用 post_processor.generate_object_center_dairv2x
        """
        order = self.params['postprocess']['order'] 
        lidar_range = self.params['preprocess']['cav_lidar_range']
        
        object_dict = {}
        for content in cav_contents:
            if 'params' not in content or 'vehicles' not in content['params']:
                continue
                
            vehicles = content['params']['vehicles']
            for obj in vehicles:
                obj_id = obj['id']
                loc = np.array(obj['location'])
                dim = obj['dimensions'] # [h, w, l]
                h, w, l = dim[0], dim[1], dim[2]
                yaw = obj['angle']
                # 构造 [x, y, z, dx, dy, dz, yaw]
                if order == 'hwl':
                    box = np.array([loc[0], loc[1], loc[2], h, w, l, yaw])
                else: 
                    box = np.array([loc[0], loc[1], loc[2], l, w, h, yaw])
                
                # 过滤范围 (使用原始 range，不加额外 padding)
                box_expanded = box.reshape(1, 7)
                box_filtered = box_utils.mask_boxes_outside_range_numpy(
                    box_expanded, lidar_range, order
                )
                
                if box_filtered.shape[0] > 0:
                    object_dict[obj_id] = box_filtered[0]
        # 转换为 Tensor 格式
        max_num = self.params['postprocess']['max_num']
        object_np = np.zeros((max_num, 7))
        mask = np.zeros(max_num)
        object_ids = []

        if len(object_dict) > 0:
            for i, (key, val) in enumerate(object_dict.items()):
                if i >= max_num:
                    break
                object_np[i] = val
                mask[i] = 1
                object_ids.append(key)

        return object_np, mask, object_ids
    
    def generate_object_center_single(self, cav_contents, reference_lidar_pose, return_visible_mask=False):
        """
        专门处理单车视角 (Single View) 的标签，读取 'vehicles_single' 键。
        """
        # 1. 容错处理：防止传入字典导致报错
        if isinstance(cav_contents, dict):
            cav_contents = [cav_contents]

        order = self.params['postprocess']['order'] 
        lidar_range = self.params['preprocess']['cav_lidar_range']
        
        
        object_dict = {}
        for content in cav_contents:
            # === 【关键修改点 1】: 检查 'vehicles_single' ===
            if 'params' not in content or 'vehicles_single' not in content['params']:
                continue
            
            # === 【关键修改点 2】: 读取 'vehicles_single' ===   
            vehicles = content['params']['vehicles_single']
            
            for obj in vehicles:
                obj_id = obj['id']
                loc = np.array(obj['location'])
                dim = obj['dimensions'] # [h, w, l]
                h, w, l = dim[0], dim[1], dim[2]
                yaw = obj['angle']
                
                # 构造 [x, y, z, dx, dy, dz, yaw]
                if order == 'hwl':
                    box = np.array([loc[0], loc[1], loc[2], h, w, l, yaw])
                else: 
                    box = np.array([loc[0], loc[1], loc[2], l, w, h, yaw])
                
                # 过滤范围
                box_expanded = box.reshape(1, 7)
                box_filtered = box_utils.mask_boxes_outside_range_numpy(
                    box_expanded, lidar_range, order
                )
                
                if box_filtered.shape[0] > 0:
                    object_dict[obj_id] = box_filtered[0]

        # 转换为 Tensor 格式 (这部分逻辑与 generate_object_center 完全一致)
        max_num = self.params['postprocess']['max_num']
        object_np = np.zeros((max_num, 7))
        mask = np.zeros(max_num)
        object_ids = []

        if len(object_dict) > 0:
            for i, (key, val) in enumerate(object_dict.items()):
                if i >= max_num:
                    break
                object_np[i] = val
                mask[i] = 1
                object_ids.append(key)

        return object_np, mask, object_ids