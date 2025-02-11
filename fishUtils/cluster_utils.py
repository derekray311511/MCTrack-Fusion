from sklearn.metrics import v_measure_score
from nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
from box_utils import is_points_inside_obb
from utils import (
    encodeCategory, 
    decodeCategory,
)
from geometry_utils import (
    euler_from_quaternion,
    q_to_wxyz, 
    q_to_xyzw,
    eucl2D,
    get_4f_transform,
)
from nusc_util import nusc_data
from visualizer import TrackVisualizer
from sklearn.cluster import DBSCAN
import numpy as np
import argparse
import yaml, json
import cv2

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck',
    'background'
]

def get_v_measure_score(ground_truth_labels, predicted_labels) -> float:
    '''
    v = (1 + beta) * homogeneity * completeness
     / (beta * homogeneity + completeness)
    '''
    assert len(ground_truth_labels) == len(predicted_labels)
    v_measure = v_measure_score(ground_truth_labels, predicted_labels)
    return v_measure

def v_meas_test():
    # Ground truth labels and predicted cluster labels
    ground_truth_labels = [-1, -1, -1, 2, 3, 4]
    predicted_labels = [-1, -1, -1, -1, 3, 3]
    v_meas_score = get_v_measure_score(ground_truth_labels, predicted_labels)
    print(f"v measure score: {v_meas_score}")
    exit(0)

def dbscan_test():
    X = np.array([[1, 2], [2, 2], [2, 3],
                  [8, 7], [8, 8], [25, 80]])
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    print(clustering.labels_)

class RadarSegmentor:
    """
    Segment radar targets using LiDAR detection

    param:
        categories: list of nuscenes detection category to be used
    """
    def __init__(self, cfg):
        self.categories = cfg["detCategories"]
        self.track_cat = NUSCENES_TRACKING_NAMES
        self.expand_rotio = cfg["expand_ratio"]
        self.fuse_vel = cfg["fuse_velocity"]
        self.id_count = 0
        self.radarSegmentation = None   # radar segmentation that is already grouped by LiDAR
        
    def reset(self):
        self.id_count = 0
        self.radarSegmentation = None

    def filterDetByCat(self, det):
        filtered_det = []
        for obj in det:
            if obj['detection_name'] in self.categories:
                filtered_det.append(obj)
        return filtered_det

    def runLiDARSeg(self, radarTargets, lidarDet):
        """ (Global pose)
        Returns radar targets with id: list of [x, y, z, vx, vy, id, cat_num]

        radarTargets: list of [x, y, z, vx, vy]
        lidarDet: nusc_det : list of {'translation': [x, y, z], 'rotation': [w, x, y, z], 'size': [x, y, z], 'velocity': [vx, vy], 'detection_name': s, 'detection_score': s, 'sample_token': t}
        """
        radarSegmentation = []

        # Filter LiDAR detection by name
        lidarDet = self.filterDetByCat(lidarDet)

        # Sort the bboxes by detection scores
        lidarDet.sort(reverse=True, key=lambda box:box['detection_score'])

        # Prepare radar center x, y
        pts = np.array([p[:2] for p in radarTargets])  # Extract x, y coordinates
        radarTargets = np.array(radarTargets)

        # Create a mask for filtering points and targets
        remaining_idxs = np.arange(len(pts))

        # Set init id and cat
        radarTargets = np.hstack([radarTargets, -np.ones((len(radarTargets), 2))])

        # Segment radar targets
        for det in lidarDet:
            ratio = self.expand_rotio
            center = det['translation'][:2]
            size = [det['size'][1] * ratio[1] , det['size'][0] * ratio[0]]
            det_vel = det['velocity']
            cat_num = encodeCategory([det['detection_name']], self.track_cat)[0]
            row, pitch, angle = euler_from_quaternion(q_to_xyzw(det['rotation']))
            inbox_idxs = is_points_inside_obb(pts[remaining_idxs], center, size, angle)
            
            # Use boolean masks for filtering instead of indexing
            temp_idxs = remaining_idxs[inbox_idxs]
            if len(temp_idxs) == 0:
                continue

            # Give radar targets segmentation id
            radarTargets[temp_idxs, 5] = self.id_count

            # Give radar targets category name
            radarTargets[temp_idxs, 6] = cat_num

            # Use LiDAR detection velocity
            if self.fuse_vel:
                for idx in temp_idxs:
                    radarTargets[idx, 3:5] = det_vel[:2]

            radarSegmentation.append(radarTargets[temp_idxs])
            self.id_count += 1
            remaining_idxs = remaining_idxs[~inbox_idxs]  # Update remaining_idxs using the opposite mask

        # Mark remaining points as none segmented targets
        # Give none segmented targets -1 as id
        remaining_targets = radarTargets[remaining_idxs]
        self.radarSegmentation = radarTargets

        return radarTargets

    def runDBSCAN(self, radarTargets):
        """ run DBSCAN on radar data, give each target an ID

        Input format: list of [x, y, z, vx, vy]
        Output format: list of [x, y, z, vx, vy, id, cat]

        notice that cat is always -1 in this function
        """
        radarTargets = np.array(radarTargets)
        pts = [p[:2] for p in radarTargets]
        clustering = DBSCAN(eps=2, min_samples=2).fit(pts)
        radarTargets = np.hstack([radarTargets, clustering.labels_.reshape(-1, 1)])
        radarTargets = np.hstack([radarTargets, -np.ones((radarTargets.shape[0], 1))])  # -1 cat
        return radarTargets

class cluster_evaluation:
    def __init__(
        self, 
        dataroot='/data/small_data2', 
        data_version='v1.0-trainval', 
        split='val',
        segmentor=None,
        nusc_data=None,
        radarSegPath='/data/early_fusion_track_results/2023-09-01-20:27:39_radarUncertainty1.2_noFuseMotor/radar_seg_res.json',
        **kwargs,
    ):
        self.dataroot = dataroot
        self.data_version = data_version
        self.split = split
        self.segmentor = segmentor
        self.nusc_data = nusc_data
        if self.split == 'train':
            self.scene_names = splits.train
        elif self.split == 'val':
            self.scene_names = splits.val
        elif self.split == 'test':
            sys.exit("Not support test data yet!")
        else:
            sys.exit(f"No split type {self.split}!")
        self.nusc = nusc_data.nusc
        self.radarSeg = self.readRadarSegRes(radarSegPath)
        self.kwargs = kwargs

    def formatSeg2numpy(self, radarSeg):
        newSeg = []
        info_len = len(radarSeg[0])
        for p in radarSeg:
            point = []
            point.extend(p['translation'])
            point.extend(p['velocity'])
            point.append(p['segment_id'])
            point.append(p['category'])
            point = np.array(point)
            newSeg.append(point)
        return np.array(newSeg)
        
    def getRadarSegAnnos(self, val_cats=None, rm_background=False):
        if self.data_version == 'v1.0-trainval':
            scenes = splits.val
        elif self.data_version == 'v1.0-test':
            scenes = splits.test
        else:
            raise ValueError("unknown")

        trackViz1 = TrackVisualizer(
            windowName='LiDAR seg radar',
            **self.kwargs["VISUALIZER"],
        )
        trackViz2 = TrackVisualizer(
            windowName='Radar DBSCAN',
            **self.kwargs["VISUALIZER"],
        )

        annotations = []
        v_meas_L_total = []
        v_meas_R_total = []
        cal_nums = {}
        for cat in val_cats:
            cal_nums[cat] = 0
        nusc_samples = tqdm(self.nusc.sample)
        for sample in nusc_samples:
            calculate_frame = False
            scene_name = self.nusc.get("scene", sample['scene_token'])['name']
            if scene_name not in scenes:
                continue

            timestamp = sample["timestamp"] * 1e-6
            token = sample["token"]
            frame = {'token': token, 'timestamp': timestamp}
            
            LIDAR_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = self.nusc.get('ego_pose', LIDAR_record['ego_pose_token'])
            ego_trans = get_4f_transform(ego_pose, inverse=True)

            # Load radar pointcloud
            radarPC = self.nusc_data.load_radarPC(token)

            # Load annotation bboxes
            annos = []
            for annotation_token in sample['anns']:
                anno = self.nusc.get('sample_annotation', annotation_token)
                anno['detection_name'] = self.nusc_data.catMapping[anno['category_name']]
                if val_cats is not None and anno['detection_name'] not in val_cats:
                    continue
                if anno['detection_name'] in val_cats:
                    calculate_frame = True
                    cal_nums[anno['detection_name']] += 1
                anno['detection_score'] = 1.0
                anno['velocity'] = [0.0, 0.0]
                if anno['detection_name']:
                    annos.append(anno)

            # Segmentation with annotation bboxes
            annoSeg = self.segmentor.runLiDARSeg(radarPC, annos)
            ann_id_list = []
            
            # Load radar segmentation result
            radarSeg = self.getRadarSeg(token)
            filtered_radarSeg = []
            res_id_list = []

            # DBSCAN segmentation
            clustering = self.segmentor.runDBSCAN(radarPC)
            filtered_clustering = []
            dbscan_id_list = []

            # Filter Radar targets by velocity (use raw velocity)
            for anno, res, res2 in zip(annoSeg, radarSeg, clustering):
                # set vel < ? to background
                if np.linalg.norm(anno[3:5]) < 0.4:
                    anno[5] = res[5] = res2[5] = -1
                # Set none validation category to -1 (background)
                if decodeCategory([int(res[6])], NUSCENES_TRACKING_NAMES)[0] not in val_cats:
                    res[5] = -1
                # Remove background annotation points
                if rm_background and anno[5] == -1:
                    continue
                ann_id_list.append(anno[5])
                res_id_list.append(res[5])
                dbscan_id_list.append(res2[5])
                filtered_radarSeg.append(res)
                filtered_clustering.append(res2)
            
            if not calculate_frame:   # Only calculate validation annotation categories
                continue

            # Calculate v-measurement
            v_meas_L = self.cal_v_meas_for_frame(ann_id_list, res_id_list)
            v_meas_L_total.append(v_meas_L)
            v_meas_R = self.cal_v_meas_for_frame(ann_id_list, dbscan_id_list)
            v_meas_R_total.append(v_meas_R)
            print(f"calculated frames num: {len(v_meas_L_total)}")
            print(f"v measurement (L): {v_meas_L:.3f}, average: {np.mean(np.array(v_meas_L_total)):.3f}")
            print(f"v measurement (R): {v_meas_R:.3f}, average: {np.mean(np.array(v_meas_R_total)):.3f}")
            print(f"calculate numbers: \n{cal_nums}")

            # Visualization
            if trackViz1.play:
                key = cv2.waitKey(int(trackViz1.duration * 1000))
            else:
                key = cv2.waitKey(0)

            if key == 27: # esc
                cv2.destroyAllWindows()
                exit(0)
            elif key == 32: # space
                trackViz1.play = not trackViz1.play
            elif key == 43: # +
                trackViz1.duration *= 2
                print(f"Viz duration set to {trackViz1.duration}")
            elif key == 45: # -
                trackViz1.duration *= 0.5
                print(f"Viz duration set to {trackViz1.duration}")
            elif key == ord('g'):
                trackViz1.grid = not trackViz1.grid
                trackViz2.grid = not trackViz2.grid

            trackViz1.draw_ego_car(img_src="/data/car1.png")
            trackViz2.draw_ego_car(img_src="/data/car1.png")
            trackViz1.draw_radar_seg(radarSeg=filtered_radarSeg, trans=ego_trans, **self.kwargs["VISUALIZER"]["radarSeg"])
            trackViz2.draw_radar_seg(radarSeg=filtered_clustering, trans=ego_trans, **self.kwargs["VISUALIZER"]["radarSeg"])
            trackViz1.draw_det_bboxes(annos, ego_trans, **self.kwargs["VISUALIZER"]["fusionBox"])
            trackViz2.draw_det_bboxes(annos, ego_trans, **self.kwargs["VISUALIZER"]["fusionBox"])
            trackViz1.show()
            trackViz2.show()

            nusc_samples.set_description()
            nusc_samples.refresh()
        
        v_meas_L_average = np.mean(np.array(v_meas_L_total))
        v_meas_R_average = np.mean(np.array(v_meas_R_total))
        print(f"average v measurement (L) = {v_meas_L_average}")
        print(f"average v measurement (R) = {v_meas_R_average}")

    def readRadarSegRes(self, path):
        with open(path, 'rb') as f:
            radarSeg = json.load(f)["results"]
        return radarSeg

    def getRadarSeg(self, token):
        radarSeg = self.radarSeg[token]
        return self.formatSeg2numpy(radarSeg)

    def filterTargetsByVel(self, radarTargets, th=0.1):
        new = []
        for p in radarTargets:
            if np.linalg.norm(p[3:5]) < th:
                continue
            new.append(p)
        return np.array(new)

    def cal_v_meas_for_frame(self, list1, list2):
        return get_v_measure_score(list1, list2)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="cluster_eval")
    parser.add_argument("config", metavar="CONFIG_FILE")
    return parser

if __name__ == "__main__":
    # v_meas_test()
    # dbscan_test()
    parser = get_parser()
    args, opts = parser.parse_known_args()
    cfg = yaml.safe_load(open(args.config))
    cfg["SEGMENTOR"]["fuse_velocity"] = False
    segmentor = RadarSegmentor(cfg["SEGMENTOR"])
    nusc_data = nusc_data(dataroot='/data/nuscenes')
    cluster_eval = cluster_evaluation(
        segmentor=segmentor, 
        nusc_data=nusc_data,
        **cfg,
    )
    val_cats = [
        'bus',
    ]
    cluster_eval.getRadarSegAnnos(val_cats, rm_background=False)

