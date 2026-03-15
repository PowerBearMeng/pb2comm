import open3d as o3d
import numpy as np
import os
import time

# ================= 配置区域 =================
ROOT_DIR = "/home/yty/mfh/record"
SEQ_NAME = "seq12"   # 你的序列名
LIDAR_FOLDER = "new_points_virt" 

PLAY_DELAY = 0.5     # 播放延迟
RANSAC_THRESH = -0.0  # RANSAC 距离阈值 (米)，小于此距离被认为是地面
RANSAC_ITER = 100    # RANSAC 迭代次数
# ===========================================

class ColoredVisualizer:
    def __init__(self, pcd_files):
        self.pcd_files = pcd_files
        self.total = len(pcd_files)
        self.idx = 0
        self.is_paused = False
        self.first_frame_rendered = False
        
        # 初始化窗口
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=f"Colored Ground - {SEQ_NAME}", width=1280, height=720)
        
        # 注册按键
        self.vis.register_key_callback(32, self.toggle_pause) # Space
        self.vis.register_key_callback(78, self.next_frame)   # N
        self.vis.register_key_callback(81, self.quit_app)     # Q
        
        # 渲染选项：深灰背景
        opt = self.vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1]) 
        
        # 初始化点云对象
        self.pcd = o3d.geometry.PointCloud()

        print(f"\n[操作说明]")
        print(f"  > 空格: 暂停/播放")
        print(f"  > N键:  下一帧")
        print(f"  > Q键:  退出")
        print(f"  > 颜色: [蓝色]=地面, [红色]=障碍物")
        print("-" * 40)

    def toggle_pause(self, vis):
        self.is_paused = not self.is_paused
        print(f" >> {'暂停' if self.is_paused else '播放'}")
        return False

    def next_frame(self, vis):
        if self.is_paused:
            self.idx = (self.idx + 1) % self.total
            self.draw_frame()
        return False
        
    def quit_app(self, vis):
        vis.close()
        return False

    def read_bin(self, file_path):
        if not os.path.exists(file_path):
            return np.array([])
        try:
            # 尝试 N*4 (x,y,z,i) -> 取前3列
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            return points[:, :3]
        except ValueError:
            try:
                # 尝试 N*3 (x,y,z)
                points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 3)
                return points
            except Exception:
                return np.array([])

    def draw_frame(self):
        file_path = self.pcd_files[self.idx]
        file_name = os.path.basename(file_path)
        
        # 1. 读取原始数据
        points = self.read_bin(file_path)
        
        if len(points) == 0:
            print(f"[{self.idx}] {file_name} | 数据为空！")
            return

        # 2. 准备 Open3D 点云
        # 为了使用 segment_plane，我们需要先构建一个临时的 pcd 对象
        self.pcd.points = o3d.utility.Vector3dVector(points)

        # 3. RANSAC 地面分割
        # segment_plane 返回: [a,b,c,d], inliers(索引列表)
        # distance_threshold: 点到平面的距离阈值
        if len(points) > 50:
            plane_model, inliers = self.pcd.segment_plane(distance_threshold=RANSAC_THRESH,
                                                          ransac_n=3,
                                                          num_iterations=RANSAC_ITER)
            
            print(f"Plane model: {plane_model[0]:.3f}x + {plane_model[1]:.3f}y + {plane_model[2]:.3f}z + {plane_model[3]:.3f} = 0")
            # 4. 设置颜色
            # 默认全部设为红色 (障碍物) [1, 0, 0]
            colors = np.tile([1.0, 0.0, 0.0], (len(points), 1))
            
            # 将地面点 (inliers) 设为蓝色 [0, 0, 1]
            if len(inliers) > 0:
                # colors[inliers] = [0.0, 0.0, 1.0]
                pass 
            # 删除z以下的点云
            z = points[:, 2]
            colors[z < 0.57] = [0.0, 1.0, 0.0]
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 简单的调试信息
            min_z = np.min(points[:, 2])
            print(f"[{self.idx}/{self.total}] {file_name} | Pts: {len(points)} | MinZ: {min_z:.2f}m")
            
        else:
            # 点太少，不分割，直接全白
            self.pcd.paint_uniform_color([1, 1, 1])

        # 5. 刷新画面
        self.vis.remove_geometry(self.pcd, reset_bounding_box=False)
        self.vis.add_geometry(self.pcd, reset_bounding_box=not self.first_frame_rendered)
        
        if not self.first_frame_rendered:
            self.vis.reset_view_point(True)
            self.first_frame_rendered = True

    def run(self):
        self.draw_frame()
        while True:
            self.vis.poll_events()
            self.vis.update_renderer()
            
            if not self.is_paused:
                time.sleep(PLAY_DELAY)
                self.idx = (self.idx + 1) % self.total
                self.draw_frame()

if __name__ == "__main__":
    target_dir = os.path.join(ROOT_DIR, SEQ_NAME, "roadside0", LIDAR_FOLDER)
    
    if not os.path.exists(target_dir):
        print(f"[错误] 路径不存在: {target_dir}")
        exit()
        
    files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.bin')])
    
    if not files:
        print("[错误] 没找到 .bin 文件")
        exit()
        
    print(f"加载了 {len(files)} 帧，准备播放...")
    app = ColoredVisualizer(files)
    app.run()