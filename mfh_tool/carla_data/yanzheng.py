import os
import json
import math
import numpy as np
import open3d as o3d

class DataVisualizer:
    def __init__(self, root_dir, seq="seq01", view="roadside0"):
        self.base_path = os.path.join(root_dir, seq, view)
        
        # 指向转换后的目录
        self.points_dir = os.path.join(self.base_path, "new_points_virt")
        self.labels_dir = os.path.join(self.base_path, "new_labels_virt")
        
        self.frames = sorted([f.split('.')[0] for f in os.listdir(self.points_dir) if f.endswith('.bin')])
        print(f"Found {len(self.frames)} frames in {self.points_dir}")
        
        self.current_idx = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

    def get_box_matrix(self, location, rotation_yaw, dimensions):
        """
        DAIR-V2X / KITTI 风格的 Box 矩阵构建 (只有 Yaw 旋转)
        location: {x, y, z} (底面中心 或 几何中心，取决于你的转换脚本，这里假设是中心)
        rotation_yaw: float (radians)
        dimensions: {h, w, l}
        """
        x = location['x']
        y = location['y']
        z = location['z']
        print(f"Box Location: x={x}, y={y}, z={z}, yaw={rotation_yaw}, dim={dimensions}")
        # 只需要绕 Z 轴旋转 (Yaw)
        c = np.cos(rotation_yaw)
        s = np.sin(rotation_yaw)
        
        # 旋转矩阵 Rz
        rotation_matrix = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        center = np.array([x, y, z])
        extent = np.array([dimensions['l'], dimensions['w'], dimensions['h']])
        
        return center, rotation_matrix, extent

    def load_point_cloud(self, frame_id):
        bin_path = os.path.join(self.points_dir, f"{frame_id}.bin")
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        # --- 🎨 染色方案 A: 按高度 (Z) 染色 (强烈推荐！方便看平不平) ---
        # 地面低 -> 蓝色/紫色，高处 -> 红色/黄色
        z = xyz[:, 2]
        # 归一化 Z (掐头去尾，防止噪声影响颜色分布)
        min_z = np.percentile(z, 5)  # 忽略底部 5% 的噪点
        max_z = np.percentile(z, 95) # 忽略顶部 5% 的噪点
        norm_z = (z - min_z) / (max_z - min_z + 1e-6)
        norm_z = np.clip(norm_z, 0, 1)
        
        # 使用 matplotlib 的 colormap (手动实现简易版: 蓝->红)
        colors = np.zeros((xyz.shape[0], 3))
        # R通道: 高处红
        colors[:, 0] = norm_z 
        # G通道: 中间绿 (可选，增加对比度)
        colors[:, 1] = np.sin(norm_z * np.pi) * 0.5
        # B通道: 低处蓝
        colors[:, 2] = 1 - norm_z
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # --- 🎨 染色方案 B: 纯白色 (如果你只想看轮廓，取消下面这行的注释) ---
        # pcd.paint_uniform_color([1, 1, 1]) 
            
        return pcd

    def load_boxes(self, frame_id):
        json_path = os.path.join(self.labels_dir, f"{frame_id}.json")
        if not os.path.exists(json_path):
            print(f"Label not found: {json_path}")
            return []

        with open(json_path, 'r') as f:
            data = json.load(f)
            
        objects = data.get('objects', [])
        geometries = []
        
        color_map = {
            "Car": [1, 0, 0],       # Red
            "Cyclist": [0, 1, 0],   # Green
            "Pedestrian": [0, 0, 1] # Blue
        }

        for obj in objects:
            # 适配新的 DAIR 格式 key
            loc = obj.get('3d_location')
            dim = obj.get('3d_dimensions')
            rot = obj.get('rotation', 0.0) # 这是一个 float (yaw 弧度)
            
            if loc is None or dim is None:
                continue

            center, R, extent = self.get_box_matrix(loc, rot, dim)
            
            bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
            lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
            
            c = color_map.get(obj.get('type', 'Car'), [1, 1, 0]) 
            lines.paint_uniform_color(c)
            geometries.append(lines)
            
        return geometries

    def update_vis(self, vis):
        vis.clear_geometries()
        
        # 添加坐标轴 (原点)，红色X，绿色Y，蓝色Z
        # 如果数据转平了，蓝色轴应该垂直向上
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
        vis.add_geometry(axis)

        frame_id = self.frames[self.current_idx]
        
        # 加载点云
        pcd = self.load_point_cloud(frame_id)
        vis.add_geometry(pcd)
        
        # 加载框
        boxes = self.load_boxes(frame_id)
        for b in boxes:
            vis.add_geometry(b)

        # 首次加载设置视角
        if self.current_idx == 0:
            vis.reset_view_point(True)
            print(f"Showing Frame: {frame_id}. Press 'N' for next, 'B' for prev.")
            
    def next_frame(self, vis):
        self.current_idx = (self.current_idx + 1) % len(self.frames)
        self.update_vis(vis)
        return False

    def prev_frame(self, vis):
        self.current_idx = (self.current_idx - 1 + len(self.frames)) % len(self.frames)
        self.update_vis(vis)
        return False

    def run(self):
        self.vis.create_window(window_name='DAIR-V2X Viewer', width=1280, height=720)
        
        self.vis.register_key_callback(ord('N'), self.next_frame)
        self.vis.register_key_callback(ord('B'), self.prev_frame)
        
        self.update_vis(self.vis)
        
        # 开启渲染循环
        self.vis.run()
        self.vis.destroy_window()

if __name__ == "__main__":
    # 配置你的路径
    ROOT_DIR = "/home/yty/mfh/record"
    SEQ_NAME = "seq07"
    # 既然我们要看转平的效果，这里看 roadside0
    VIEW_NAME = "roadside0"  # 也可以改成 "roadside0"
    
    viewer = DataVisualizer(ROOT_DIR, SEQ_NAME, VIEW_NAME)
    viewer.run()