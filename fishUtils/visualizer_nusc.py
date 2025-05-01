import cv2
import numpy as np
import itertools
import colorsys
import time
import os

from .geometry_utils import *
from .box_utils import get_3d_box, get_2d_box, get_3d_box_8corner, euler_from_quaternion
from .utils import encodeCategory, decodeCategory, get_trk_colormap, cal_func_time
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from concurrent.futures import ThreadPoolExecutor

class TrackVisualizer:
    def __init__(
        self, 
        nusc_cfg: dict,
        viz_cat: list,
        windowName: str = "track",
        range_: tuple = (100, 100),
        windowSize: tuple = (800, 800),
        imgSize: tuple = (1600, 1600), 
        duration: float = 0.5,
        background_color: tuple = ((249, 249, 249), (50, 50, 50)),
        frameRate: int = 2,
        **kwargs,
    ):
        self.nusc_cfg = nusc_cfg
        self.nusc = NuScenes(**self.nusc_cfg)
        self.viz_cat = viz_cat
        self.trk_colorMap = get_trk_colormap()
        self.range = range_
        self.height = imgSize[0]
        self.width = imgSize[1]
        self.resolution = self.range[0] / self.height
        self.play = True
        self.duration = duration
        self.windowName = windowName
        self.window = cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        self.windowSize = np.array(windowSize, dtype=np.uint8)
        self.grid_cfg = kwargs.get('grid_cfg', None) 
        self.draw_grid_or_not = self.grid_cfg.get('draw', False) if self.grid_cfg else False
        self.border_cfg = kwargs.get('border', None)
        self.frameRate = frameRate
        self._grid_cache, self._grid_ready = None, False
        self._car_cache = None
        self._lidar_cache = None
        self._cam_cache = None
        self._executor = ThreadPoolExecutor(max_workers=6)  # one thread per camera view
        self.DAYNIGHT = kwargs.get('daynight', 'night')   # True: day, False: night
        self.DAYNIGHT_MAP = {'day': 0, 'night': 1}
        self.background_color = np.array(background_color, dtype=np.uint8)
        self.reset()
        
        cv2.resizeWindow(self.windowName, windowSize)
        print(f"window: {self.windowName}")
        print(f"Visualize category: {self.viz_cat}")
        print(f"Visualize range: {self.range}")
        print(f"res: {self.resolution}")
        print(f"Image size: {self.height, self.width}")
        print(f"duration: {self.duration}")

    def reset(self):
        self.cam_height, self.cam_width = 0, 0
        self.image = np.empty((self.height, self.width, 3), dtype=np.uint8)
        bg_color = self.background_color[self.DAYNIGHT_MAP[self.DAYNIGHT]]
        cv2.rectangle(self.image, (0,0), (self.width, self.height), bg_color.tolist(), thickness=cv2.FILLED)
        self._cam_cache = None
        self._lidar_cache = None

    def clear_cache(self):
        self._grid_cache, self._grid_ready = None, False
        self._car_cache = None
        self._cam_cache = None
        self._lidar_cache = None

    def switch_daynight(self, setting=None):
        if setting is not None:
            self.DAYNIGHT = setting
        elif self.DAYNIGHT == 'day':
            self.DAYNIGHT = 'night'
        else:
            self.DAYNIGHT = 'day'
        self.clear_cache()

    def read_lidar_pc(self, token):
        sample_record = self.nusc.get('sample', token)
        lidar_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        point_cloud = np.fromfile(os.path.join(self.nusc.dataroot, lidar_record['filename']), dtype=np.float32).reshape(-1,5)
        point_cloud = np.array(point_cloud[:, [0,1,2]])
        point_cloud = self.PCsensor2car(point_cloud, token, 'LIDAR_TOP', inverse=False)
        return point_cloud

    def get_4f_transform(self, pose, inverse=False):
        return transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=inverse)

    def PCsensor2car(self, pointcloud, token, name='LIDAR_TOP', inverse=False):
        '''
        Input pointcloud shape: (n, 3)
        '''
        pointcloud = np.array(pointcloud)
        sample_record = self.nusc.get('sample', token)
        sensor_record = self.nusc.get('sample_data', sample_record['data'][name])
        ego_pose = self.nusc.get('ego_pose', sensor_record['ego_pose_token'])
        cs_record = self.nusc.get('calibrated_sensor', sensor_record['calibrated_sensor_token'])
        sensor2car = self.get_4f_transform(cs_record, inverse=inverse)
        point_cloud = np.hstack([pointcloud, np.ones((pointcloud.shape[0], 1))])
        point_cloud = point_cloud @ sensor2car.T
        point_cloud = point_cloud.reshape(-1, 4)[:, :3]
        return point_cloud

    def draw_lidar_pts(self, token, cache=None, **kwargs):
        lidar_pc = None
        if cache is None:
            lidar_pc = self.read_lidar_pc(token)
            lidar_pc = lidar_pc[:, :2]  # Only take the x and y coordinates
            lidar_pc = self.ego2bev_points(lidar_pc) / self.resolution
            lidar_pc = lidar_pc + np.array([self.height // 2, self.width // 2])
            lidar_pc = lidar_pc.astype(int)
        _, overlay = self.draw_points_fast(lidar_pc, cache=cache, **kwargs)
        self._lidar_cache = overlay

    def draw_points_fast(self, pts_px, color=(200,200,200), radius=2, alpha=0.8, cache=None, **kwargs):
        if cache is None:
            radius = max(1, int(radius * self.height / 1600.0))
            H, W = self.image.shape[:2]
            # 1) 建空 mask
            mask = np.zeros((H, W), np.uint8)
            # 2) 以 NumPy 高速散點 (超過邊界自動裁掉)
            xs, ys = pts_px[:, 0].clip(0, W-1), pts_px[:, 1].clip(0, H-1)
            mask[ys, xs] = 255  # O(N) 純 C 向量化
            # 3) 若要 radius > 0，可一次 dilate：
            if radius > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1,)*2)
                mask = cv2.dilate(mask, k, iterations=1)    # 單次形態學
            # 4) 把 mask 染成 BGR
            overlay = self.image.copy()
            overlay[mask == 255] = color
        else:
            overlay = cache
        cv2.addWeighted(overlay, alpha, self.image, 1-alpha, 0, self.image)   # 只混一次
        return self.image, overlay

    def draw_points(self, pts: np.ndarray, BGRcolor=(50, 50, 255), radius=4, alpha=0.8, **kwargs):
        base_radius = 4.0 if radius is None else radius
        radius = max(1, int(base_radius * self.height / 1600.0))
        overlay = self.image.copy()
        for p in pts:
            p = np.round(p).astype(int)
            cv2.circle(overlay, p, radius, BGRcolor, -1, lineType=cv2.LINE_AA)
        self.image = cv2.addWeighted(self.image, 1-alpha, overlay, alpha, 0)

    def draw_bboxes(self, corners: np.ndarray, BGRcolor=(255, 150, 150), thickness=2, alpha=1.0, **kwargs):
        boxes_image = self.image.copy()
        for box in corners:
            box = np.round(box).astype(int)
            cv2.drawContours(boxes_image, [box], 0, BGRcolor, thickness=thickness, lineType=cv2.LINE_AA)
        self.image = cv2.addWeighted(self.image, 1-alpha, boxes_image, alpha, 0)

    def world2cam4f(self, token, cam_name):
        sample_record = self.nusc.get('sample', token)
        sd_record = self.nusc.get('sample_data', sample_record['data'][cam_name])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        camera_intrinsic = np.array(cs_record['camera_intrinsic'])
        viewpad = np.eye(4)
        viewpad[:camera_intrinsic.shape[0], :camera_intrinsic.shape[1]] = camera_intrinsic

        # world to ego
        world2car = self.get_4f_transform(pose_record, inverse=True)
        # ego to camera
        car2cam = self.get_4f_transform(cs_record, inverse=True)
        # camera to image
        cam2img = viewpad
        # world to image
        world2img = cam2img @ car2cam @ world2car
        return world2img
    
    def car2cam4f(self, token, cam_name):
        sample_record = self.nusc.get('sample', token)
        sd_record = self.nusc.get('sample_data', sample_record['data'][cam_name])
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        camera_intrinsic = np.array(cs_record['camera_intrinsic'])
        viewpad = np.eye(4)
        viewpad[:camera_intrinsic.shape[0], :camera_intrinsic.shape[1]] = camera_intrinsic

        # ego to camera
        car2cam = self.get_4f_transform(cs_record, inverse=True)
        # camera to image
        cam2img = viewpad
        # car to image
        car2img = cam2img @ car2cam
        return car2img

    def get_camera_images(self, token):
        sample_record = self.nusc.get('sample', token)
        name_list = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
            'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
        ]
        images = {}
        for sensor in name_list:
            sensor_record = self.nusc.get('sample_data', sample_record['data'][sensor])
            img = cv2.imread(os.path.join(self.nusc.dataroot, sensor_record['filename']))
            images[sensor] = img
        for i, (name, img) in enumerate(images.items()):
            cv2.putText(img, name, (10, 80), cv2.FONT_HERSHEY_TRIPLEX, 3, (20, 100, 160), 3, cv2.LINE_AA)
            images.update({name: img})
        img_front = np.hstack([images[name_list[0]], images[name_list[1]], images[name_list[2]]])
        img_back = np.hstack([images[name_list[3]], images[name_list[4]], images[name_list[5]]])
        return img_front, img_back
    
    def _load_img(self, path):
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

    def load_cams_parallel(self, sample_record, name_list):
        # prepare absolute paths -------------------------------------------------
        paths = []
        for sensor in name_list:
            sd_record = self.nusc.get('sample_data', sample_record['data'][sensor])
            filename = sd_record['filename']
            filename = os.path.join(self.nusc_cfg['dataroot'], filename)
            paths.append(filename)

        # submit tasks -----------------------------------------------------------
        futs = [self._executor.submit(self._load_img, p) for p in paths]

        # gather (keeps order) ---------------------------------------------------
        imgs = {cam: fut.result() for cam, fut in zip(name_list, futs)}

        # cache images ---------------------------------------------------
        self._cam_cache = imgs
        return imgs

    def draw_camera_images(self, token, boxes=[], BGRcolor=(255, 50, 50), colorID=False, imgs=None, **kwargs):
        """
        在當前畫布的上下分別添加前相機和後相機的圖像，並進行等比例調整。
        
        :param front_image: 前相機圖像 (height, width, 3)。
        :param rear_image: 後相機圖像 (height, width, 3)。
        """
        def resize_to_match_width(image, target_width):
            """
            等比例調整圖像的寬度以匹配目標寬度。
            
            :param image: 要調整的圖像。
            :param target_width: 目標寬度。
            :return: 調整後的圖像。
            """
            h, w, _ = image.shape
            scale_factor = target_width / w
            new_height = int(h * scale_factor)
            return cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_LINEAR)

        color = np.array(BGRcolor, dtype=np.float32) / 255.0
        base_color = np.clip(color, a_min=0, a_max=1)
        color = deepcopy(base_color)
        bboxes_corners = []
        ids = []
        for box in boxes:
            x, y, z = box['translation']
            w, l, h = box['size']
            vx, vy = box['velocity'][:2]
            Quaternion = q_to_xyzw(box['rotation'])
            roll, pitch, yaw = euler_from_quaternion(Quaternion)
            corners = get_3d_box_8corner((x, y, z), (l, w, h), yaw)
            bboxes_corners.append(corners)
            if 'tracking_id' in box:
                ids.append(int(box['tracking_id']))
        ids = np.array(ids, dtype=np.int32)
        bboxes_corners = np.array(bboxes_corners)

        sample_record = self.nusc.get('sample', token)
        name_list = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
            'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
        ]
        if imgs is None:
            imgs = self.load_cams_parallel(sample_record, name_list)

        for sensor in name_list:

            transform = self.world2cam4f(token, sensor)
            img = imgs[sensor]

            num_bboxes = bboxes_corners.shape[0]
            coords = np.concatenate(
                [bboxes_corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
            )
            transform = transform.reshape(4, 4)
            coords = coords @ transform.T
            coords = coords.reshape(-1, 8, 4)

            indices = np.all(coords[..., 2] > 0, axis=1)
            coords = coords[indices]
            # labels = labels[indices]
            if boxes and 'tracking_id' in boxes[0]:
                obj_ids = ids[indices]

            indices = np.argsort(-np.min(coords[..., 2], axis=1))
            coords = coords[indices]
            # labels = labels[indices]
            if boxes and 'tracking_id' in boxes[0]:
                obj_ids = obj_ids[indices]

            coords = coords.reshape(-1, 4)
            coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
            coords[:, 0] /= coords[:, 2]
            coords[:, 1] /= coords[:, 2]

            coords = coords[..., :2].reshape(-1, 8, 2)

            for index in range(coords.shape[0]):
                # Set colors (BGR)
                if colorID and 'tracking_id' in boxes[0]:
                    id = obj_ids[index]
                    color[0] = 100                                                  # B 100
                    color[1] = int((base_color[1] * 255 - (id) * 20) % 155 + 100)   # G 100~255
                    color[2] = int((base_color[2] * 255 + (id) * 15) % 155 + 100)   # R 100~255
                else:
                    color = (base_color * 255)
                    color = [int(color[0]), int(color[1]), int(color[2])]

                for start, end in [
                    (0, 1),
                    (0, 3),
                    (0, 4),
                    (1, 2),
                    (1, 5),
                    (3, 2),
                    (3, 7),
                    (4, 5),
                    (4, 7),
                    (2, 6),
                    (5, 6),
                    (6, 7),
                ]:
                    cv2.line(
                        img,
                        coords[index, start].astype(np.int32),
                        coords[index, end].astype(np.int32),
                        color=(int(color[0]), int(color[1]), int(color[2])),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
            imgs.update({sensor: img})

        font_scale = 3
        for name, img in imgs.items():
            cv2.putText(img, name, (10, 80), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (20, 100, 160), 3, cv2.LINE_AA)
            imgs.update({name: img})

        img_front = np.hstack([imgs[name_list[0]], imgs[name_list[1]], imgs[name_list[2]]])
        img_rear = np.hstack([imgs[name_list[3]], imgs[name_list[4]], imgs[name_list[5]]])

        # 等比例調整前相機和後相機圖像的寬度
        front_image_resized = resize_to_match_width(img_front, self.width)
        rear_image_resized = resize_to_match_width(img_rear, self.width)

        # 獲取調整後圖像的高度
        front_height = front_image_resized.shape[0]
        rear_height = rear_image_resized.shape[0]

        # 計算新畫布的高度
        new_height = self.image.shape[0] + front_height + rear_height

        # 使用 vstack 創建新畫布
        new_canvas = np.vstack((
            front_image_resized,
            self.image,
            rear_image_resized
        ))

        # 更新當前畫布
        self.image = new_canvas
        self.cam_height = front_height
        self.cam_width = self.width

    def draw_ego_car(self, img_src,
                    car_width_m=3.0,    # 實車寬 (m)
                    car_len_m =5.0):    # 實車長 (m)
        # ---------- 1. 讀圖 (含 alpha) ----------
        if self._car_cache is None:
            car = cv2.imread(img_src, cv2.IMREAD_UNCHANGED)
            if car is None:
                raise FileNotFoundError(img_src)

            # 強制 BGR+A 格式
            if car.shape[2] == 3:
                # 若沒 alpha，就把非黑像素視為 255（或自行生成）
                alpha = 255 * np.ones(car.shape[:2], np.uint8)
                car   = np.dstack([car, alpha])
            # 依需要旋轉
            car = cv2.rotate(car, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # ---------- 2. 物理尺寸 → 像素 ----------
            w_px = int(round(car_width_m / self.resolution))
            h_px = int(round(car_len_m  / self.resolution))

            # 保持奇數 (左右上下可對稱貼到中心)
            if w_px % 2 == 0: w_px += 1
            if h_px % 2 == 0: h_px += 1

            # ---------- 3. 等比例縮放並透明 padding 到 (h_px, w_px) ----------
            bh, bw = car.shape[:2]
            scale  = min(w_px / bw, h_px / bh)
            new_w, new_h = int(bw * scale), int(bh * scale)
            car_rs = cv2.resize(car, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 建空透明畫布，將 car 貼到中間
            car_pad = np.zeros((h_px, w_px, 4), np.uint8)
            y0 = (h_px - new_h)//2
            x0 = (w_px - new_w)//2
            car_pad[y0:y0+new_h, x0:x0+new_w] = car_rs
            self._car_cache = car_pad
        else:
            car_pad = self._car_cache
            w_px, h_px = car_pad.shape[1], car_pad.shape[0]

        # ---------- 4. 取 RGB 與 mask ----------
        rgb  = car_pad[:,:,:3]
        mask = car_pad[:,:,3]

        # ---------- 5. 貼到 BEV 畫布中心 ----------
        y_c, x_c = self.height//2, self.width//2
        y1 = y_c - h_px//2; y2 = y1 + h_px
        x1 = x_c - w_px//2; x2 = x1 + w_px

        roi = self.image[y1:y2, x1:x2]          # 與 car_pad 同大小
        cv2.copyTo(rgb, mask, roi)              # 透明貼圖 (OpenCV4)

        # 若 cv2<4.1 無 copyTo(dst)，可:
        # inv_mask = cv2.bitwise_not(mask)
        # bg = cv2.bitwise_and(roi, roi, mask=inv_mask)
        # fg = cv2.bitwise_and(rgb, rgb, mask=mask)
        # self.image[y1:y2, x1:x2] = cv2.add(bg, fg)


    def draw_radar_pts(
        self, 
        radar_pc: list, 
        trans: np.ndarray, 
        BGRcolor=(50, 50, 255), 
        radius=2,
        draw_vel=False, 
        thickness=2, 
        alpha=0.5,
        global_draw=False,
        **kwargs
    ):
        """Param :

        radar_pc : list of [x, y, z, vx, vy]
        ego_pose : {'translation': [x, y, z], 'rotation': [w, x, y, z], 'timestamp': t, 'token' : t}

        """
        if len(radar_pc) == 0:
            return

        local_pts = []
        for point in radar_pc:
            local_pts.append(point[:3])

        # radius = max(1, int(radius * self.height / 1600.0))
        thickness = max(1, int(thickness * self.height / 1600.0))

        local_pts = np.array(local_pts, dtype=float)
        local_pts = np.hstack([local_pts, np.ones((local_pts.shape[0], 1), dtype=float)])
        local_pts = (trans @ local_pts.T).T[:, :2]
        local_pts = self.ego2bev_points(local_pts)
        local_pts = local_pts / self.resolution + np.array([self.height // 2, self.width // 2])
        if global_draw:
            self.draw_points_fast(local_pts.astype(int), BGRcolor, radius, alpha, **kwargs)
        else:
            self.draw_points(local_pts.astype(int), BGRcolor, radius, alpha, **kwargs)
        if draw_vel:
            vel = np.array([point[3:5] for point in radar_pc])
            vel = np.hstack([vel, np.zeros((vel.shape[0], 1))])
            vel = (trans[:3, :3] @ vel.T).T[:, :2]
            vel = self.ego2bev_points(vel) / self.resolution
            pt_objs = []
            for p, v in zip(local_pts, vel):
                pt_objs.append({
                    'translation': p,
                    'velocity': v
                })
            self._draw_vel(pt_objs, BGRcolor, thickness, alpha)

    def _cluster_centroid(self, pts_int32: np.ndarray) -> tuple:
        """ Return (x, y) centroid of Nx2 int32 array """
        m = pts_int32.mean(axis=0)  # NumPy 求平均 2d
        return int(m[0]), int(m[1])

    def _draw_cross(self, image, center, size=10, color=(50, 50, 250), thickness=2):
        """ 在 image 上以 X 畫中心點 """
        x, y = center
        d = size // 2
        cv2.line(image, (x-d, y-d), (x+d, y+d), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x-d, y+d), (x+d, y-d), color, thickness, cv2.LINE_AA)

    def draw_radar_seg(
        self, 
        radarSeg: np.ndarray, 
        trans: np.ndarray, 
        colorID=False, 
        colorName=False, 
        alphaPts=0.8,
        alphaMask=0.4,
        contours=True, 
        **kwargs
    ):
        '''
        Input radarSeg shape: (N, 6): [x, y, z, vx, vy, clusterID]
        '''
        if colorID and colorName:
            assert "colorID and colorName can not be True simultaneously"
        radarSeg = sorted(radarSeg, key=lambda x: x[5])
        radarSeg = np.array(radarSeg)
        # Default BGR color
        colors = kwargs.get('color', ((210, 60, 10), (100, 170, 0)))[self.DAYNIGHT_MAP[self.DAYNIGHT]]
        colorPts = colors[0]
        colorMask = colors[1]
        if colorID:
            for k, g in itertools.groupby(radarSeg, lambda x: x[5]):
                g = list(g)
                # BGRcolor = getColorFromID(ID=k, colorRange=(50, 255))
                BGRcolor = getColorFromID_HSV(ID=k, cycle_num=12)
                if k == -1:  # id == -1
                    self.draw_radar_pts(g, trans, BGRcolor=BGRcolor, alpha=alphaPts, **kwargs)
                else:
                    self.draw_radar_pts(g, trans, BGRcolor=BGRcolor, alpha=alphaPts, **kwargs)
                if contours:
                    self.draw_cluster_polygons(np.array(g), trans, BGRcolor=BGRcolor, alpha=alphaMask, **kwargs)
        elif colorName:
            for k, g in itertools.groupby(radarSeg, lambda x: x[5]):
                g = list(g)
                cat_num = int(g[0][6])
                cat_name = decodeCategory([cat_num], self.viz_cat)[0]
                if cat_num == -1:  # id == -1
                    B, G, R = 100, 100, 100 # Gray color
                    self.draw_radar_pts(g, trans, BGRcolor=(B, G, R), alpha=alphaPts, **kwargs)
                    if contours:
                        self.draw_cluster_polygons(np.array(g), trans, BGRcolor=BGRcolor, alpha=alphaMask, **kwargs)
                else:
                    BGRcolor = self.trk_colorMap[cat_name]
                    self.draw_radar_pts(g, trans, BGRcolor=BGRcolor, alpha=alphaPts, **kwargs)
                    if contours:
                        self.draw_cluster_polygons(np.array(g), trans, BGRcolor=BGRcolor, alpha=alphaMask, **kwargs)
        else:
            self.draw_radar_pts(radarSeg, trans, BGRcolor=colorPts, alpha=alphaPts, **kwargs)
            if contours:
                self.draw_cluster_polygons(radarSeg, trans, BGRcolor=colorMask, alpha=alphaMask, **kwargs)

    def draw_cluster_polygons(
        self,
        radarSeg,                       # N×6, labels在 [:,5]
        trans,
        BGRcolor=(90, 150, 0),
        poly_type="hull",               # "hull" | "rect"
        mode="both",                    # "fill" | "outline" | "both"
        thickness=4,                    # 畫邊框的粗細
        alpha=0.4,
        inflate_px=8,                   # ★ 向外擴幾個 pixel (0 = 不膨脹)
        draw_center=False,
        **kwargs
    ):
        if radarSeg.shape[0] == 0:
            return 

        H,W      = self.image.shape[:2]
        img      = self.image
        overlay  = img.copy()
        
        # ---------- 配色 ----------
        labels   = radarSeg[:, 5].astype(np.int32)
        clusters = np.unique(labels)
        clusters = clusters[clusters!=-1]
        if BGRcolor:
            base = np.array(BGRcolor, dtype=int).reshape(1, 3)   # 先變 (1,3)
            colors = np.repeat(base, len(clusters), axis=0)      # → (K,3)
        else:
            colors = (np.random.rand(len(clusters),3)*255).astype(int)

        # ---------- 參數 ----------
        thickness = max(1, int(thickness * self.height / 1600.0))
        inflate_px = int(inflate_px * self.height / 1600.0)
        x_size = max(1, int(12 * self.height / 1600.0))
        x_thickness = max(1, int(2 * self.height / 1600.0))
        x_color = (50, 50, 255)

        # ---------- 座標轉像素 ----------
        local_pts = []
        for point in radarSeg:
            local_pts.append(point[:3])
        local_pts = np.array(local_pts, dtype=float)
        local_pts = np.hstack([local_pts, np.ones((local_pts.shape[0], 1), dtype=float)])
        local_pts = (trans @ local_pts.T).T[:, :2]
        local_pts = self.ego2bev_points(local_pts)
        local_pts = local_pts / self.resolution + np.array([self.height // 2, self.width // 2])
        points = np.array(local_pts, dtype=int)
        
        # ---------- 逐 cluster 繪製 ----------
        if inflate_px > 0:
            ksize  = inflate_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        
        for cid, col in zip(clusters, colors):
            pts = points[labels==cid].astype(np.int32)
            col = tuple(int(x) for x in col)     # 轉 Python int
            # if pts.shape[0] < 3: continue
            
            # ===== 1. 先算凸包  =====
            if poly_type=="hull":                # ▷ 凸包
                hull = cv2.convexHull(pts).reshape(-1,2)
            else:                                # ▷ 最小面積矩形
                hull = cv2.boxPoints(cv2.minAreaRect(pts)).astype(int)
            
            if inflate_px == 0:
                poly = hull
            else:
                # ===== 2. 只在 ROI 做形態學膨脹 =====

                # 1. 以 int 形式取得最小外接方框，並加上 inflate 範圍
                min_xy = hull.min(0) - inflate_px
                max_xy = hull.max(0) + inflate_px

                # 2. clip 到影像邊界後，再轉成純 int
                x0 = int(max(0,        min_xy[0]))
                y0 = int(max(0,        min_xy[1]))
                x1 = int(min(W - 1,    max_xy[0]))
                y1 = int(min(H - 1,    max_xy[1]))

                # 3. 若結果退化成空矩形，就跳過
                roi_w = x1 - x0 + 1
                roi_h = y1 - y0 + 1
                if roi_w <= 2 or roi_h <= 2:
                    continue   # 點距離太小，膨脹後也看不到；跳過

                mask_roi = np.zeros((roi_h, roi_w), np.uint8)
                cv2.fillPoly(mask_roi, [hull - [x0, y0]], 255)  # 先填在 ROI 座標
                cv2.dilate(mask_roi, kernel, mask_roi, iterations=1)

                cnts, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                poly = cnts[0] + [x0, y0]  # 還原到全域座標
            
            # === 畫圖 ===
            if mode in ["fill","both"]:
                cv2.fillPoly(overlay, [poly], col)
            if mode in ["outline","both"]:
                cv2.polylines(img, [poly], True, col, thickness, cv2.LINE_AA)

            # === 畫中心 ===
            if draw_center:
                center = self._cluster_centroid(pts)
                self._draw_cross(img, center, size=x_size, color=col, thickness=x_thickness)
        
        if mode in ["fill","both"]:
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    def world2ego(self, objects, ego_trans):
        ret = []
        for object in objects:
            trans = np.array(object['translation'])
            vel = np.array([object['velocity'][0], object['velocity'][1], 0.0])
            rot = quaternion_rotation_matrix(object['rotation'])
            trans = np.hstack([rot, trans.reshape(-1, 1)])
            trans = np.vstack([trans, np.array([0, 0, 0, 1])]).reshape(-1, 4)
            vel = vel.reshape(-1, 1)
            new_trans = ego_trans @ trans
            new_vel = ego_trans[:3, :3] @ vel
            object['translation'] = new_trans[:3, 3].ravel().tolist()
            object['rotation'] = q_to_wxyz(R.from_matrix(new_trans[:3, :3]).as_quat())
            object['velocity'] = new_vel.ravel()[:2]
            ret.append(object)
        return ret
    
    def ego2bev_points(self, points):
        u = -points[:,1]   # -y → u
        v = -points[:,0]   # -x → v
        return np.c_[u, v]
    
    def ego2bev_boxes(self, boxes):
        ret = []
        for b in boxes:
            # --- 1) 位置 ---
            x, y, z = b['translation']
            b['translation'] = [-y, -x, z]

            # --- 2) 速度 ---
            vx, vy = b['velocity']
            b['velocity'] = [-vy, -vx]

            # --- 3) 朝向 ---
            q = b['rotation']                       # wxyz
            yaw = R.from_quat([q[1],q[2],q[3],q[0]]).as_euler('xyz')[2]
            yaw_bev = -yaw - np.pi/2                # 公式
            q_bev = R.from_euler('z', yaw_bev).as_quat()
            b['rotation'] = [q_bev[3], q_bev[0], q_bev[1], q_bev[2]]
            ret.append(b)
        return ret

    def getBoxCorners2d(self, boxes: list) -> np.ndarray:
        corners = []
        for box in boxes:
            x, y, z = box['translation']
            w, l, h = box['size']
            Quaternion = q_to_xyzw(box['rotation'])
            roll, pitch, yaw = euler_from_quaternion(Quaternion)
            corner = get_2d_box([x, y], [l, w], yaw)
            corners.append(corner)
        return np.array(corners)
    
    def draw_dashed_circle(self, img, center, radius, color=(0, 0, 255), thickness=2, dash_length=6, gap_length=3, alpha=1.0):
        """
        在 img 上以 center 為圓心畫一個虛線圓。
        
        :param img: 要繪製的圖像 (np.array)
        :param center: 圓心座標 (x, y)
        :param radius: 圓的半徑 (單位：像素)
        :param color: BGR 顏色，例如 (255, 150, 150)
        :param thickness: 線寬
        :param dash_length: 每一段線的長度 (像素)
        :param gap_length: 線段之間的間隔 (像素)
        """
        circumference = 2 * np.pi * radius
        # 計算大約需要的線段數
        num_dashes = int(circumference / (dash_length + gap_length))
        if num_dashes < 1:
            num_dashes = 1
        angle_step = 2 * np.pi / num_dashes
        for i in range(num_dashes):
            start_angle = i * angle_step
            # 計算 dash 對應的角度長度
            dash_angle = dash_length / radius  # 注意：dash_length / radius 得到弧度值
            end_angle = start_angle + dash_angle
            pt1 = (int(center[0] + radius * np.cos(start_angle)), int(center[1] + radius * np.sin(start_angle)))
            pt2 = (int(center[0] + radius * np.cos(end_angle)), int(center[1] + radius * np.sin(end_angle)))
            cv2.line(img, pt1, pt2, color, thickness)

    def draw_uncertainty_ellipse(self, image, mean, cov_matrix, color=(0, 255, 0), scale=1.0):
        """
        在圖像上繪製表示不確定性的橢圓。

        :param image: 要繪製的圖像。
        :param mean: 狀態均值（中心點），形狀為 (2,) 的數組。
        :param cov_matrix: 4x4 協方差矩陣。(Only use the first 2x2 part because we viz covariance x, y)
        :param color: 橢圓的顏色，默認為綠色。
        :param scale: 橢圓尺度因子，控制置信區域的大小。 1 for 1 sigma, 2 for 2 sigma, etc.
        """
        if not np.all(np.isfinite(cov_matrix)):      # 含 NaN/Inf
            return
        eigvals, eigvecs = np.linalg.eigh(cov_matrix[:2, :2])
        if np.any(eigvals <= 0):                     # 非正定
            return
        # 計算特徵值和特徵向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 按降序排序特徵值
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

        # 計算橢圓的角度
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # 計算橢圓的半軸長度
        half_major_axis = scale * np.sqrt(eigenvalues[0])
        half_minor_axis = scale * np.sqrt(eigenvalues[1])

        # 繪製橢圓
        cv2.ellipse(
            image,
            (int(mean[0]), int(mean[1])),
            (int(half_major_axis), int(half_minor_axis)),
            angle,
            0,
            360,
            color,
            2
        )
        cv2.circle(image, (int(mean[0]), int(mean[1])), 3, color, -1)

    def draw_det_bboxes(
        self, 
        nusc_det: list, 
        trans: np.ndarray, 
        draw_vel: bool = False,
        draw_id: bool = False,
        draw_name: bool = False,
        draw_score: bool = False,
        draw_matching_circle: bool = False,
        draw_cov: bool = False,
        BGRcolor=(255, 150, 150), 
        thickness=None,
        colorID=False, 
        colorName=False, 
        legend=False,
        **kwargs
    ):
        """Param :

        nusc_det : list of {'translation': [x, y, z], 'rotation': [w, x, y, z], 'size': [x, y, z], 'velocity': [vx, vy], 'detection_name': s, 'detection_score': s, 'sample_token': t}
        ego_pose : {'translation': [x, y, z], 'rotation': [w, x, y, z], 'timestamp': t, 'token' : t}
        legend : bool (default: True) - draw detection name on the top left corner (only effective when colorName is True)

        """
        if len(nusc_det) == 0:
            return
        nusc_det = deepcopy(nusc_det)
        nusc_det = self.world2ego(nusc_det, trans)
        base_thickness = 2.0 if thickness is None else thickness
        thickness = max(1, int(base_thickness * self.height / 1600.0)) if base_thickness > 0 else -1    # -1 for ground truth
        fontScale = 0.6 * self.height / 1600.0
        if 'detection_name' not in nusc_det[0] and 'tracking_name' in nusc_det[0]:
            for det in nusc_det:
                det['detection_name'] = det['tracking_name']
        if 'detection_score' not in nusc_det[0] and 'tracking_score' in nusc_det[0]:
            for det in nusc_det:
                det['detection_score'] = det['tracking_score']

        nusc_det = self.ego2bev_boxes(nusc_det) # 將物體轉換到BEV圖像坐標系 From frontX-leftY to rightX-downY
        for det in nusc_det:

            # Transform to image pixel level
            det['translation'] = np.array(det['translation']) / self.resolution
            det['translation'][:2] = det['translation'][:2] + np.array([self.height // 2, self.width // 2])
            det['size'] = np.array(det['size']) / self.resolution
            det['velocity'] = np.array(det['velocity']) / self.resolution
            
            # 處理 covariance 矩陣翻轉 (flip y-axis)
            if draw_cov and 'covariance' in det:
                A = np.array([
                    [ 0, -1,  0,  0],
                    [-1,  0,  0,  0],
                    [ 0,  0,  0, -1],
                    [ 0,  0, -1,  0]
                ])
                cov_matrix = np.array(det['covariance']).reshape(4, 4)
                cov_matrix = A @ cov_matrix @ A.T
                # 注意 covariance 的單位縮放 (位置單位是 resolution, 速度也是，因此統一用 resolution^2)
                det['covariance'] = cov_matrix / (self.resolution ** 2)

        if colorName:   # Draw boxes by detection_name
            legends = {}
            for cat in self.viz_cat:
                BGRcolor = self.trk_colorMap[cat]
                legends[cat] = (BGRcolor, f"{cat}: 0")
            # a map used to draw matching circles
            matching_range_map = {
                'car': 4.0,
                'pedestrian': 4.0,
                'bicycle': 2.5,
                'motorcycle': 2.0,
                'bus': 4.0,
                'trailer': 4.0,
                'truck': 4.0,
            }
            # Sort the detections by 'detection_name' to ensure proper grouping
            nusc_det = sorted(nusc_det, key=lambda x: x['detection_name'])
            for k, g in itertools.groupby(nusc_det, lambda x: x['detection_name']):
                g_det = list(g)
                cat_num = encodeCategory([k], self.viz_cat)[0]
                BGRcolor = self.trk_colorMap[k]
                corners2d = self.getBoxCorners2d(g_det)
                self.draw_bboxes(corners2d, BGRcolor, thickness, **kwargs)
                # 為該類別的每個偵測框畫出匹配圓圈
                if draw_matching_circle:
                    for det in g_det:
                        # 取偵測框中心
                        center = (int(det['translation'][0]), int(det['translation'][1]))
                        # 根據類別設定匹配範圍半徑
                        radius = matching_range_map.get(k, 3.0) / self.resolution
                        # 畫虛線圓
                        self.draw_dashed_circle(
                            self.image, 
                            center, 
                            radius, 
                            BGRcolor, 
                            thickness=thickness, 
                            dash_length=radius // 2, 
                            gap_length=radius // 4, 
                            # alpha=0.5,
                        )
                if draw_cov:
                    for det in g_det:
                        if 'covariance' not in det:
                            continue
                        mean = det['translation']
                        cov_matrix = det['covariance']
                        self.draw_uncertainty_ellipse(self.image, mean, cov_matrix, scale=2.0)
                    
                if draw_vel:
                    self._draw_vel(g_det, BGRcolor, thickness, **kwargs)
                if draw_id:
                    self._draw_id(g_det, BGRcolor, fontScale, max(1, thickness // 2), **kwargs)
                if draw_name:
                    self._draw_name(g_det, BGRcolor, fontScale, max(1, thickness // 2), **kwargs)
                if draw_score:
                    self._draw_score(g_det, BGRcolor, fontScale, max(1, thickness // 2), **kwargs)
                legends[k] = (BGRcolor, f"{k}: {len(g_det)}")
            if legend:
                scale = self.height / 1600.0  # 假設預設是1600
                legend_x = int(20 * scale)
                legend_y = int(20 * scale)
                legend_spacing = int(40 * scale)
                rect_size = int(30 * scale)
                font_scale = 1.5 * scale
                font_thickness = max(1, int(2 * scale))  # 避免字體太細
                for (color, text) in legends.values():
                    cv2.rectangle(self.image, (legend_x, legend_y), (legend_x + rect_size, legend_y + rect_size), color, thickness)
                    cv2.putText(self.image, text, (legend_x + rect_size + int(20 * scale), legend_y + rect_size), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
                    legend_y += legend_spacing

        elif colorID and ('tracking_id' in nusc_det[0]):
            for det in nusc_det:
                # BGRcolor = getColorFromID(ID=det['tracking_id'], colorRange=(50, 255))
                BGRcolor = getColorFromID_HSV(ID=det['tracking_id'], cycle_num=12)
                corners2d = self.getBoxCorners2d([det])
                self.draw_bboxes(corners2d, BGRcolor, thickness, **kwargs)
                if draw_vel:
                    self._draw_vel([det], BGRcolor, thickness, **kwargs)
                if draw_id:
                    self._draw_id([det], BGRcolor, fontScale, max(1, thickness // 2), **kwargs)
                if draw_name:
                    self._draw_name([det], BGRcolor, fontScale, max(1, thickness // 2), **kwargs)
                if draw_score:
                    self._draw_score([det], BGRcolor, fontScale, max(1, thickness // 2), **kwargs)

        else:   # Draw all boxes using same BGRcolor
            corners2d = self.getBoxCorners2d(nusc_det)
            self.draw_bboxes(corners2d, BGRcolor, thickness, **kwargs)
            if draw_vel:
                self._draw_vel(nusc_det, BGRcolor, thickness, **kwargs)
            if draw_id:
                self._draw_id(nusc_det, BGRcolor, fontScale, max(1, thickness // 2), **kwargs)
            if draw_name:
                self._draw_name(nusc_det, BGRcolor, fontScale, max(1, thickness // 2), **kwargs)
            if draw_score:
                self._draw_score(nusc_det, BGRcolor, fontScale, max(1, thickness // 2), **kwargs)

    def drawTP_FP_FN(
        self, 
        predictions, 
        ground_truths, 
        matched_predictions, 
        matched_gts,
        trans: np.ndarray, 
        thickness=2,
        legend=False,
        **kwargs
        ):
        if len(predictions) == 0 and len(ground_truths) == 0:
            return
        # (TP in green, FP in red, FN in blue)
        TP = []
        FP = []
        FN = []
        tp_color = (0, 255, 0)
        fp_color = (0, 0, 255)
        fn_color = (255, 0, 0)
        for gt in ground_truths:
            if gt['instance_token'] not in {g['instance_token'] for g in matched_gts}:  # Replace 'instance_token' with the appropriate key
                FN.append(gt)

        # get TP and FP boxes 
        for pred in predictions:
            if pred['tracking_id'] in {p['tracking_id'] for p in matched_predictions}:  # Replace 'tracking_id' with the appropriate key
                TP.append(pred)
            else:
                FP.append(pred)

        # draw legend on the top left corner
        if legend:
            legend_x = 20
            legend_y = 20
            legend_spacing = 40
            legends = [
                (tp_color, f"TP: {len(TP)} True Positive"),
                (fp_color, f"FP: {len(FP)} False Positive"),
                (fn_color, f"FN: {len(FN)} False Negative")
            ]
            for color, text in legends:
                cv2.rectangle(self.image, (legend_x, legend_y), (legend_x + 30, legend_y + 30), color, thickness)
                cv2.putText(self.image, text, (legend_x + 50, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
                legend_y += legend_spacing

        self.draw_det_bboxes(TP, trans, BGRcolor=tp_color, thickness=thickness, **kwargs)
        self.draw_det_bboxes(FP, trans, BGRcolor=fp_color, thickness=thickness, **kwargs)
        self.draw_det_bboxes(FN, trans, BGRcolor=fn_color, thickness=thickness, draw_id=False)

    def _draw_vel(self, nusc_det: list, BGRcolor=(255, 255, 255), thickness=1, alpha=1.0, **kwargs):
        if thickness <= 0:
            thickness = 2
        image = self.image.copy()
        for det in nusc_det:
            vel = det['velocity'][:2]
            vel = vel / self.frameRate
            if np.linalg.norm(vel) < 0.2:
                continue
            start_point = det['translation'][:2]
            end_point = start_point + vel
            start_point = np.round(start_point).astype(int)
            end_point = np.round(end_point).astype(int)
            image = cv2.arrowedLine(image, start_point, end_point, BGRcolor, thickness, line_type=cv2.LINE_AA)
        self.image = cv2.addWeighted(self.image, 1-alpha, image, alpha, 0)

    def _draw_id(self, nusc_det: list, BGRcolor=(255, 255, 255), fontScale=0.8, thickness=1, alpha=1.0, **kwargs):
        image = self.image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = fontScale
        for det in nusc_det:
            org = det['translation'][:2]
            org = np.round(org).astype(int)
            ID = int(float(det['tracking_id']))
            text_size, _ = cv2.getTextSize(str(ID), font, fontScale, thickness)
            org = (org[0], org[1] - text_size[1] // 2)
            image = cv2.putText(image, str(ID), org, font, fontScale, BGRcolor, thickness, cv2.LINE_AA)
        self.image = cv2.addWeighted(self.image, 1-alpha, image, alpha, 0)

    def _draw_name(self, nusc_det: list, BGRcolor=(255, 255, 255), fontScale=0.8, thickness=1, alpha=1.0, **kwargs):
        image = self.image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = fontScale
        for det in nusc_det:
            org = det['translation'][:2]
            org = np.round(org).astype(int)
            name = det['detection_name'][:3]
            text_size, _ = cv2.getTextSize(name, font, fontScale, thickness)
            org = (org[0] - text_size[0] // 2, org[1] - text_size[1])
            image = cv2.putText(image, name, org, font, fontScale, BGRcolor, thickness, cv2.LINE_AA)
        self.image = cv2.addWeighted(self.image, 1-alpha, image, alpha, 0)

    def _draw_score(self, nusc_det: list, BGRcolor=(255, 255, 255), fontScale=0.8, thickness=1, alpha=1.0, **kwargs):
        image = self.image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = fontScale
        for det in nusc_det:
            org = det['translation'][:2]
            org = np.round(org).astype(int)
            score = np.round(det['detection_score'], 2).astype(np.float16)
            text_size, _ = cv2.getTextSize(str(score), font, fontScale, thickness)
            org = (org[0], org[1] + text_size[1])
            image = cv2.putText(image, str(score), org, font, fontScale, BGRcolor, thickness, cv2.LINE_AA)
        self.image = cv2.addWeighted(self.image, 1-alpha, image, alpha, 0)

    def _draw_grid(self, img, grid_shape, color=(0, 255, 0), thickness=1):
        h, w, _ = img.shape
        rows, cols = grid_shape
        dy, dx = h / rows, w / cols

        # draw vertical lines
        for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
            x = int(round(x))
            cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

        # draw horizontal lines
        for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
            y = int(round(y))
            cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

        return img
    
    def draw_grid(self, image, diff=10, color=(0, 255, 0), thickness=1):
        """ Draw grid from image center """
        h, w = self.height, self.width

        # draw vertical lines
        x = w // 2
        while(x < w):
            cv2.line(image, (x, 0), (x, h), color=color, thickness=thickness)
            cv2.line(image, (w-x, 0), (w-x, h), color=color, thickness=thickness)
            x += int(round((diff / self.resolution)))
        
        # draw horizontal lines
        y = h // 2
        while(y < h):
            cv2.line(image, (0, y), (w, y), color=color, thickness=thickness)
            cv2.line(image, (0, h-y), (w, h-y), color=color, thickness=thickness)
            y += int(round((diff / self.resolution)))

        return image

    def draw_img_boundary(self, BGRcolor=(255, 255, 255), thickness=4):
        x, y, w, h = 0, 0, self.image.shape[1], self.image.shape[0]
        cv2.rectangle(self.image, (x, y), (x+w, y+h), BGRcolor, thickness)

    def _build_grid_cache(self):
        h, w = self.height, self.width
        cy, cx = h // 2, w // 2
        if self.cam_height > 0:
            grid_img = np.zeros((h, w, 3), dtype=np.uint8)
            grid_img[0 : self.height, :] = self.image[self.cam_height : self.cam_height + self.height, :]
        else:
            grid_img = self.image.copy()
        GRID_STEP_M = self.grid_cfg["GRID_STEP_M"]
        STEP_LARGE = GRID_STEP_M[0]
        STEP_SMALL  = GRID_STEP_M[1]
        GRID_COLOR = self.grid_cfg["GRID_COLOR"][self.DAYNIGHT_MAP[self.DAYNIGHT]]
        GRID_TK = self.grid_cfg["GRID_TK"]
        GRID_ALPHA = self.grid_cfg["GRID_ALPHA"]
        TEXT_COLOR = self.grid_cfg["TEXT_COLOR"][self.DAYNIGHT_MAP[self.DAYNIGHT]]
        thickness_thick = GRID_TK[0] * self.height // 1600
        thickness_light = GRID_TK[1] * self.height // 1600
        # grid_img = self.draw_grid(grid_img, diff=GRID_STEP_M[1], color=GRID_COLOR[1], thickness=thickness_light)
        # grid_img = self.draw_grid(grid_img, diff=GRID_STEP_M[0], color=GRID_COLOR[0], thickness=thickness_thick)
        # --- 兩條距離尺 + 字 ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8 * self.height / 1600.0
        font_tk = max(1, int(2 * self.height / 1600.0))
        cv2.line(grid_img, (cx, cy), (cx, 0), GRID_COLOR[0], thickness_thick)  # y axis
        cv2.line(grid_img, (cx, cy), (0, cy), GRID_COLOR[0], thickness_thick)  # x axis
        for meter in range(0, self.range[0] // 2 + 1, STEP_SMALL):
            if meter == 0:
                continue
            meter_px = int(meter / self.resolution)
            tick_len = int(STEP_LARGE * 0.025 / self.resolution)
            cv2.line(grid_img, (cx - tick_len, cy - meter_px), (cx + tick_len, cy - meter_px), GRID_COLOR[0], thickness_light)  # y axis tick
            cv2.line(grid_img, (cx - meter_px, cy - tick_len), (cx - meter_px, cy + tick_len), GRID_COLOR[0], thickness_light)  # x axis tick
        for meter in range(0, self.range[0] // 2 + 1, STEP_LARGE):
            meter_px = int(meter / self.resolution)
            shift_px = int(font_scale * 15)
            tick_len = int(STEP_LARGE * 0.05 / self.resolution)
            if meter != 0:
                cv2.circle(grid_img, (cx, cy), meter_px, GRID_COLOR[0], thickness_light, lineType=cv2.LINE_AA)
                cv2.line(grid_img, (cx - tick_len, cy - meter_px), (cx + tick_len, cy - meter_px), GRID_COLOR[0], thickness_thick)  # y axis tick
                cv2.line(grid_img, (cx - meter_px, cy - tick_len), (cx - meter_px, cy + tick_len), GRID_COLOR[0], thickness_thick)  # x axis tick
            cv2.putText(grid_img, f'{meter:.0f} m', (cx + shift_px, cy - meter_px + shift_px*2),
                        font, font_scale, TEXT_COLOR, font_tk, cv2.LINE_AA)
            cv2.putText(grid_img, f'{meter:.0f} m', (cx - meter_px + shift_px, cy + shift_px*2),
                        font, font_scale, TEXT_COLOR, font_tk, cv2.LINE_AA)
        self._grid_cache  = grid_img       # BGR
        self._grid_ready  = True           # flag

    def draw_grid_overlay(self):
        """
        將格線貼到 BEV 畫面，不影響 camera 圖層
        ────────────────────────────────────────────
        self._grid_cache : H_bev x W x 3 uint8   事先 build 好的格線圖
        self.GRID_ALPHA  : 0.0 - 1.0  混合權重
        self.cam_height  : camera 圖層高度 (若沒有就 0)
        """
        ALPHA = self.grid_cfg["ALPHA"]
        self._build_grid_cache()

        y0 = getattr(self, 'cam_height', 0)          # camera 區高度，預設 0
        y1 = y0 + self.height                        # BEV 區下界
        x0, x1 = 0, self.width

        # ---- 只取 BEV 影像 ROI ----
        roi_img  = self.image[y0:y1, x0:x1]          # H_bev×W×3
        grid_img = self._grid_cache                  # 同尺寸

        # ---- 貼格線（單次 SIMD）----
        cv2.addWeighted(roi_img, 1.0 - ALPHA,
                        grid_img, ALPHA,
                        0, dst=roi_img)
        
    def show(self):
        """
        show and reset the image
        """
        BORDER_COLOR = self.border_cfg["BGRcolor"][self.DAYNIGHT_MAP[self.DAYNIGHT]]
        BORDER_TK    = self.border_cfg["thickness"] * self.height // 1600
        if self.draw_grid_or_not:
            self.draw_grid_overlay()
        self.draw_img_boundary(BGRcolor=BORDER_COLOR, thickness=BORDER_TK)
        cv2.imshow(self.windowName, self.image)
        # self.reset()

def getColorFromID(baseColor=(100, 100, 100), colorRange=(155, 255), ID=-1) -> tuple:
    if ID == -1:  # id == -1
        B, G, R = baseColor # Gray color
    else:
        B = ((25 + 50*ID) % (255 - colorRange[0]) + colorRange[0]) % colorRange[1]     # colorRange[0]~colorRange[1]
        G = ((50 + 30*ID) % (255 - colorRange[0]) + colorRange[0]) % colorRange[1]     # colorRange[0]~colorRange[1]
        R = ((100 + 20*ID) % (255 - colorRange[0]) + colorRange[0]) % colorRange[1]    # colorRange[0]~colorRange[1]
    return (B, G, R)

def getColorFromID_HSV(baseColor=(100, 100, 100), ID=-1, cycle_num=12):
    # Generate colors using HSV color space
    ID = int(float(ID))
    if ID == -1:
        return baseColor
    else:
        hue = (ID % cycle_num) / cycle_num
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        bgr_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # Convert RGB to BGR
        return bgr_color

    