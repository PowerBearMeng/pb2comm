import os
import json
import math
import struct
import numpy as np
import open3d as o3d
import glob

class DataVisualizer:
    def __init__(self, root_dir, seq="seq01", view="vehicle"):
        """
        :param root_dir: 数据根目录，例如 "./record"
        :param seq: 序列名，例如 "seq01"
        :param view: 视角文件夹名，例如 "vehicle" 或 "roadside0"
        """
        self.base_path = os.path.join(root_dir, seq, view)
        self.points_dir = os.path.join(self.base_path, "points")
        # self.labels_dir = os.path.join(root_dir, seq, "world","labels")
        self.labels_dir = os.path.join(self.base_path, "labels") # 或者 "world" 看你想看过滤后的还是全量的
        
        # 获取所有帧的文件名
        self.frames = sorted([f.split('.')[0] for f in os.listdir(self.points_dir) if f.endswith('.bin')])
        print(f"Found {len(self.frames)} frames in {self.base_path}")
        
        self.current_idx = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

    def _get_matrix_from_pose(self, pose_dict):
        """
        从字典(x,y,z,pitch,yaw,roll)构建 4x4 变换矩阵 (Carla/Unreal Convention)
        顺序: Scale -> Rotation(Z-Y-X) -> Translation
        这里只考虑 Rotation 和 Translation
        """
        x = pose_dict['x']
        y = pose_dict['y']
        z = pose_dict['z']
        
        # Carla uses degrees, convert to radians
        pitch = math.radians(pose_dict['pitch'])
        yaw = math.radians(pose_dict['yaw'])
        roll = math.radians(pose_dict['roll'])

        # Rotation Matrix calculation (Z-Y-X order for Carla/Unreal)
        # cy = math.cos(yaw)
        # sy = math.sin(yaw)
        # cp = math.cos(pitch)
        # sp = math.sin(pitch)
        # cr = math.cos(roll)
        # sr = math.sin(roll)
        
        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        # 参考 CARLA 官方转换逻辑
        c_y = np.cos(yaw)
        s_y = np.sin(yaw)
        c_r = np.cos(roll)
        s_r = np.sin(roll)
        c_p = np.cos(pitch)
        s_p = np.sin(pitch)
        
        matrix = np.identity(4)
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = c_p * s_y
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        
        matrix[0, 3] = x
        matrix[1, 3] = y
        matrix[2, 3] = z
        
        return matrix

    def load_point_cloud(self, frame_id):
        bin_path = os.path.join(self.points_dir, f"{frame_id}.bin")
        # 假设是 float32
        points = np.fromfile(bin_path, dtype=np.float32)
        
        # 尝试 reshape，CARLA通常是 XYZ 或 XYZI (4通道)
        # 这里的 DataRecorder 保存时可能是 (N, 3) 或 (N, 4)
        if points.size % 4 == 0:
            points = points.reshape(-1, 4)
            print(f"Point cloud has {points.shape[0]} points with 4 channels (XYZI).")
            xyz = points[:, :3]
            i = points[:, 3]
            print(f"Intensity range: min {i.min()}, max {i.max()}")
        elif points.size % 3 == 0:
            points = points.reshape(-1, 3)
            xyz = points
        else:
            raise ValueError(f"Point cloud data shape mismatch: {points.size}")
        # --- [新增] 调试代码开始 ---
        print(f"--- Frame {frame_id} Statistics ---")
        print(f"X range: min {xyz[:,0].min():.2f}, max {xyz[:,0].max():.2f}")
        print(f"Y range: min {xyz[:,1].min():.2f}, max {xyz[:,1].max():.2f}")
        print(f"Z range: min {xyz[:,2].min():.2f}, max {xyz[:,2].max():.2f}")
        # --- [新增] 调试代码结束 ---


        lidar_range = [-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf]
        
        mask = (
            (xyz[:, 0] >= lidar_range[0]) & (xyz[:, 0] <= lidar_range[3]) &
            (xyz[:, 1] >= lidar_range[1]) & (xyz[:, 1] <= lidar_range[4]) &
            (xyz[:, 2] >= lidar_range[2]) & (xyz[:, 2] <= lidar_range[5])
        )
        xyz = xyz[mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        return pcd

    def load_boxes(self, frame_id, sensor_matrix):
        """
        读取 json，直接使用存储的传感器坐标系下的 Box 信息
        """
        json_path = os.path.join(self.labels_dir, f"{frame_id}.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        objects = data['objects']
        
        # [修改点]：不再需要计算 world_to_sensor 逆矩阵
        # sensor_world_matrix = ... (删除)
        # world_to_sensor = ... (删除)
        
        lines_sets = []
        
        color_map = {
            "Car": [1, 0, 0],       # Red
            "Cyclist": [0, 1, 0],   # Green
            "Pedestrian": [0, 0, 1] # Blue
        }

        for obj in objects:
            # 1. 直接获取 Box 在传感器坐标系的位姿
            # 因为 DataRecorder 已经转换好了，这里的 xyz 和 rotation 都是相对于 LiDAR 的
            box_pose_dict = obj['location']
            box_pose_dict.update(obj['rotation']) 
            
            # 这个矩阵现在直接代表 T_sensor_to_box
            box_sensor_matrix = self._get_matrix_from_pose(box_pose_dict)
            
            # 2. 获取尺寸
            l = obj['dimensions']['l']
            w = obj['dimensions']['w']
            h = obj['dimensions']['h']
            
            # 3. 创建 Open3D OrientedBoundingBox
            # 直接从矩阵提取平移(Center)和旋转(Rotation)
            center = box_sensor_matrix[:3, 3]
            rotation = box_sensor_matrix[:3, :3]
            
            # extent 对应长宽高
            bbox = o3d.geometry.OrientedBoundingBox(center, rotation, np.array([l, w, h]))
            
            # 4. 可视化生成
            lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
            
            # 设置颜色
            c = color_map.get(obj['class'], [1, 1, 0]) 
            lines.paint_uniform_color(c)
            
            lines_sets.append(lines)
            
        return lines_sets

    def update_vis(self, vis):
        if self.current_idx >= len(self.frames):
            self.current_idx = 0
            
        frame_id = self.frames[self.current_idx]
        print(f"Showing frame: {frame_id}")
        
        vis.clear_geometries()
        
        # 1. Load Points
        pcd = self.load_point_cloud(frame_id)
        vis.add_geometry(pcd)
        
        # 2. Load Boxes
        # 空矩阵作为 placeholder，实际在 load_boxes 内部读取 json 里的 pose
        lines = self.load_boxes(frame_id, None)
        for l in lines:
            vis.add_geometry(l)
            
        # 设置视角（仅第一帧重置，后面保持用户拖动后的视角）
        if self.current_idx == 0:
            vis.reset_view_point(True)
            
    def next_frame(self, vis):
        self.current_idx = (self.current_idx + 1) % len(self.frames)
        self.update_vis(vis)
        return False

    def prev_frame(self, vis):
        self.current_idx = (self.current_idx - 1 + len(self.frames)) % len(self.frames)
        self.update_vis(vis)
        return False

    def run(self):
        self.vis.create_window(window_name='Carla Data Viewer', width=1280, height=720)
        
        # 注册按键回调
        self.vis.register_key_callback(ord('N'), self.next_frame) # 按 N 下一帧
        self.vis.register_key_callback(ord('B'), self.prev_frame) # 按 B 上一帧
        
        # 初始化显示
        self.update_vis(self.vis)
        
        self.vis.run()
        self.vis.destroy_window()

if __name__ == "__main__":
    # 配置你的路径

    ROOT_DIR = "/home/yty/mfh/record"
    SEQ_NAME = "seq05"
    VIEW_NAME = "roadside0" # 或者是 "roadside0" "vehicle"
    
    viewer = DataVisualizer(ROOT_DIR, SEQ_NAME, VIEW_NAME)
    viewer.run()