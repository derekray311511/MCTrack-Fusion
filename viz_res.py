import argparse
import copy
import os
import cv2
import sys
import time
import json
import math
import numpy as np
import shutil
import tf
import yaml

from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from tqdm import tqdm
from copy import deepcopy
from fishUtils.utils import log_parser_args, mkdir_or_exist, cal_func_time, get_current_datetime
from fishUtils.utils import encodeCategory, decodeCategory, npEncoder
from fishUtils.box_utils import get_3d_box_8corner, get_3d_box_2corner
from fishUtils.box_utils import nms, is_points_inside_obb
from fishUtils.geometry_utils import *
from fishUtils.visualizer_nusc import TrackVisualizer
from fishUtils.nusc_eval import nusc_eval
from fishUtils.nusc_util import nusc_dataset
from fishUtils.custom_eval import evaluate_nuscenes, TrackingEvaluation


class res_data:
    def __init__(
        self, 
        det_path=None,
        trk_path_1=None,
        trk_path_2=None,
        radarSeg_path_1=None,
        radarSeg_path_2=None,
        radarTrk_path=None,
    ):
        self.det = self.load_det(det_path)
        self.trk1 = self.load_trk(trk_path_1)
        self.trk2 = self.load_trk(trk_path_2)
        self.radarSeg1 = self.load_trk(radarSeg_path_1)
        self.radarSeg2 = self.load_trk(radarSeg_path_2)
        # self.radarTrk = self.load_trk(radarTrk_path)

    def load_det(self, path):
        with open(path, 'rb') as f:
            data = json.load(f)['results']
        return data

    def load_trk(self, path):
        with open(path, 'rb') as f:
            data = json.load(f)['results']
        return data
    
    def get_radarSeg(self, num, token):
        if num == 1:
            radarSeg =  self.radarSeg1
        elif num == 2:
            radarSeg =  self.radarSeg2
        else:
            return None
        seg_f = radarSeg[token]
        seg = []
        for i in range(len(seg_f)):
            p = np.zeros(6, dtype=np.float32)
            p[:3] = seg_f[i]['translation']
            p[3:5] = seg_f[i]['velocity']
            p[5] = seg_f[i]['segment_id']
            # p[5] = seg_f[i]['frame_id']
            if p[5] == -1:  # Ignore background
                continue
            seg.append(p)
        return np.array(seg)
    
    def get_avgRadarSeg(self, num, token):
        if num == 1:
            radarSeg =  self.radarSeg1
        elif num == 2:
            radarSeg =  self.radarSeg2
        else:
            return None
        seg_f = radarSeg[token]
        if len(seg_f) == 0:
            return np.array([])
        seg = np.zeros((len(seg_f), 6))
        for i in range(len(seg_f)):
            seg[i][:3] = seg_f[i]['translation']
            seg[i][3:5] = seg_f[i]['velocity']
            # seg[i][5] = seg_f[i]['frame_id']
            seg[i][5] = seg_f[i]['segment_id']

        # Get the unique cluster ids.
        cluster_ids = np.unique(np.hstack([p[-1] for p in seg]))
        avg_radar_seg = []
        for cluster_id in cluster_ids:
            cluster_points = np.vstack([p[p[-1] == cluster_id] for p in seg])
            avg_values = np.mean(cluster_points[:, :5], axis=0)
            remainInfo = cluster_points[0, 5:]
            avg_radar_seg.append(np.hstack([avg_values, remainInfo]))
        
        return np.array(avg_radar_seg)

    def get_det_bbox(self, token, threshold=0.0):
        det = self.det[token]
        det_bbox = []
        for obj in det:
            if obj['detection_score'] > threshold:
                det_bbox.append(obj)
        return det_bbox

def setWinPos(screenSize, winList):
    screen_width, screen_height = screenSize  # Replace with your screen resolution
    window_width = screen_width // 2  # Half of the screen width
    window_height = screen_height  # Full height
    winNum = len(winList)
    # Two main windows and other small windows
    if winNum < 2:
        cv2.moveWindow(winList[0].windowName, 0, 0)
    else:
        cv2.moveWindow(winList[0].windowName, 0, 0)
        cv2.moveWindow(winList[1].windowName, window_width, 0)
    
    if winNum > 2:
        for i in range(2, winNum, 1):
            win = winList[i]
            w, h = win.windowSize
            cv2.moveWindow(win.windowName, window_width - int(w/2), (i-2) * h)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LRFusion visualize")
    parser.add_argument("config", metavar="CONFIG_FILE")
    parser.add_argument("--workspace", type=str, default="/home/Student/Tracking")
    # parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--evaluate", type=int, default=0)
    parser.add_argument("--show_delay", action="store_true")
    parser.add_argument("--vizInterrupt", action="store_true")
    return parser

def main(parser) -> None:
    args, opts = parser.parse_known_args()
    cfg = yaml.safe_load(open(args.config))
    print("cv2 version: ", cv2.__version__)

    dataset = nusc_dataset(cfg["DATASET"])
    res_path_1 = "20250304_080824_Mix_threeStage_DBSCAN"
    res_path_2 = "20250428_092608_Mix_threeStage_th-0.25+MixTrack_RadarMinScore"
    trk_res = res_data(
        det_path="/data/Nuscenes/detections/centerpoint/val.json",
        trk_path_1=f"results/nuscenes/{res_path_1}/results.json",
        trk_path_2=f"results/nuscenes/{res_path_2}/results.json",
        radarSeg_path_1=f"results/nuscenes/{res_path_1}/segmentation_result.json",
        radarSeg_path_2=f"results/nuscenes/{res_path_2}/segmentation_result.json",
    )
    winName1 = 'Lidar + Radar DBSCAN'
    winName2 = 'Our Method (Lidar + TrackPrior)'
    cfg["VISUALIZER"]["detBox"] = {
        'draw': False, 
        'draw_id': False,
        'draw_name': False,
        'draw_score': False,
        'legend': False,
        **cfg["VISUALIZER"]["detBox"],
    }
    trackViz = TrackVisualizer(
        windowName=winName1,
        **cfg["VISUALIZER"],
    )
    trackViz2 = TrackVisualizer(
        windowName=winName2,
        **cfg["VISUALIZER"],
    )
    winList = [trackViz, trackViz2]
    setWinPos(cfg["VISUALIZER"]["screenSize"], winList)

    frames = dataset.get_frames_meta()
    frames = dataset.add_key_frames_info(frames)
    gts = dataset.get_groundTruth()
    R_ctl_list = [-1, 0, 1, 7]
    R_stack_num = 1
    if 'velocity' not in gts[0]['anns'][0]:
        for gt in gts:
            for obj in gt['anns']:
                obj['velocity'] = np.array([0.0, 0.0])
    len_frames = len(frames)
    max_idx = len_frames - 1
    idx = -1
    prev_daynight = cfg["VISUALIZER"]["daynight"]
    for win in winList:
        cv2.createTrackbar('Frame', win.windowName, 0, max_idx, lambda x: None)
        cv2.setTrackbarPos('Frame', win.windowName, idx)
        
    while True:

        if trackViz.play:
            idx += 1
            key = cv2.waitKey(int(trackViz.duration * 1000))
        else:
            key = cv2.waitKey(0)

        if key == 27: # esc
            cv2.destroyAllWindows()
            break
        elif key == 100 or key == 83 or key == 54: # d
            idx += 1
        elif key == 97 or key == 81 or key == 52: # a
            idx -= 1
        elif key == 32: # space
            trackViz.play = not trackViz.play
        elif key == 43: # +
            trackViz.duration *= 2
            print(f"Viz duration set to {trackViz.duration}")
        elif key == 45: # -
            trackViz.duration *= 0.5
            print(f"Viz duration set to {trackViz.duration}")
        elif key == ord('g'):
            for win in winList:
                win.grid = not win.grid
        elif key == ord('0'):
            cfg["VISUALIZER"]["daynight"] = not cfg["VISUALIZER"]["daynight"]
        elif key == ord('i'):
            cfg["VISUALIZER"]["trkBox"]["draw_id"] = not cfg["VISUALIZER"]["trkBox"]["draw_id"]
        elif key == ord('c'):
            # cfg["VISUALIZER"]["groundTruth"]["colorName"] = not cfg["VISUALIZER"]["groundTruth"]["colorName"]
            cfg["VISUALIZER"]["camera"]["draw"] = not cfg["VISUALIZER"]["camera"]["draw"]
        elif key == ord('n'):
            cfg["VISUALIZER"]["trkBox"]["draw_name"] = not cfg["VISUALIZER"]["trkBox"]["draw_name"]
            cfg["VISUALIZER"]["detBox"]["draw_name"] = cfg["VISUALIZER"]["trkBox"]["draw_name"]
        elif key == ord('s'):
            cfg["VISUALIZER"]["trkBox"]["draw_score"] = not cfg["VISUALIZER"]["trkBox"]["draw_score"]
            cfg["VISUALIZER"]["detBox"]["draw_score"] = cfg["VISUALIZER"]["trkBox"]["draw_score"]
        elif key == ord('1'):
            cfg["VISUALIZER"]["detBox"]["draw"] = not cfg["VISUALIZER"]["detBox"]["draw"]
        elif key == ord('h'):
            cfg["VISUALIZER"]["trkBox"]["draw_hist"] = not cfg["VISUALIZER"]["trkBox"]["draw_hist"]
        elif key == ord('t'):
            cfg["VISUALIZER"]["trkBox"]["draw"] = not cfg["VISUALIZER"]["trkBox"]["draw"]
        elif key == ord('k'):
            cfg["VISUALIZER"]["trkBox"]["draw_matching_circle"] = not cfg["VISUALIZER"]["trkBox"]["draw_matching_circle"]
        elif key == ord('y'):
            cfg["VISUALIZER"]["groundTruth"]["draw"] = not cfg["VISUALIZER"]["groundTruth"]["draw"]
        elif key == ord('l'):
            cfg["VISUALIZER"]["lidarPts"]["draw"] = not cfg["VISUALIZER"]["lidarPts"]["draw"]
        elif key == ord('v'):
            for name in cfg["VISUALIZER"]:
                if not isinstance(cfg["VISUALIZER"][name], dict):
                    continue
                if "draw_vel" in cfg["VISUALIZER"][name].keys():
                    cfg["VISUALIZER"][name]["draw_vel"] = not cfg["VISUALIZER"][name]["draw_vel"]
        elif key == ord('r'):
            R_stack_num = R_ctl_list[(R_ctl_list.index(R_stack_num) + 1) % len(R_ctl_list)]
            if R_stack_num <= 0:
                cfg["VISUALIZER"]["radarPts"]["draw"] = False
            else:
                cfg["VISUALIZER"]["radarPts"]["draw"] = True
            if R_stack_num == -1:
                cfg["VISUALIZER"]["radarSeg"]["draw"] = True
            else:
                cfg["VISUALIZER"]["radarSeg"]["draw"] = False
        elif key == 13: # enter
            for win in winList:
                winName = win.windowName
                det_idx = cv2.getTrackbarPos('Frame', winName)
                if det_idx != idx:
                    idx = det_idx
                    break
                    

        if idx < 0:
            idx = 0

        if idx < 0 or idx >= len(frames):
            break

        for win in winList:
            cv2.setTrackbarPos('Frame', win.windowName, idx)

        token = frames[idx]['token']
        timestamp = frames[idx]['timestamp']
        ego_pose = frames[idx]['ego_pose']
        det = trk_res.get_det_bbox(token, threshold=cfg["VISUALIZER"]["detBox"]["score_th"])
        trk1 = trk_res.trk1[token]
        trk2 = trk_res.trk2[token]
        gt = gts[idx]['anns']
        
        # filter tracks and gt by category
        # viz_cats = ["bus", "trailer"]
        # trk1 = [obj for obj in trk1 if obj['tracking_name'] in viz_cats]
        # trk2 = [obj for obj in trk2 if obj['tracking_name'] in viz_cats]
        # gt = [obj for obj in gt if obj['detection_name'] in viz_cats]
        # det = [obj for obj in det if obj['detection_name'] in viz_cats]
        # score_th = 0.25
        # trk1 = [trk for trk in trk1 if trk['tracking_score'] > score_th]
        # trk2 = [trk for trk in trk2 if trk['tracking_score'] > score_th]
        # det = [det for det in det if det['detection_score'] > score_th]

        # if len(gt) > 0:
        #     trackViz.play = False

        # Day and night color define
        BGRcolor_nameList = ["lidarPts", "radarPts", "groundTruth", "trkBox", "detBox"]
        if cfg["VISUALIZER"]["daynight"]:
            cfg["VISUALIZER"]["background_color"] = cfg["VISUALIZER"]["day_background_color"]
            for name in BGRcolor_nameList:
                cfg["VISUALIZER"][name]["BGRcolor"] = cfg["VISUALIZER"][name]["day_BGRcolor"]
        else:
            cfg["VISUALIZER"]["background_color"] = cfg["VISUALIZER"]["night_background_color"]
            for name in BGRcolor_nameList:
                cfg["VISUALIZER"][name]["BGRcolor"] = cfg["VISUALIZER"][name]["night_BGRcolor"]
        for win in winList:
            win.background_color = np.array(cfg["VISUALIZER"]["background_color"], dtype=np.uint8)
            win.reset()

        frame_name = "{}-{}".format(timestamp, idx)

        radar_pcs = dataset.get_radar_pcs(token, max_stack=R_stack_num) # Max is 7
        trans = dataset.get_4f_transform(ego_pose, inverse=True)
        viz_start = time.time()
        if cfg["VISUALIZER"]["trkBox"]["draw_hist"]:
            origin_alpha = cfg["VISUALIZER"]["trkBox"]["alpha"]
            hist_num = 4
            for i in reversed(range(hist_num)):
                if idx - i - 1 < frames[idx]['first_frame_idx']:
                    continue
                hist_token = frames[idx - i - 1]['token']
                alpha = 1.0 - 0.9 * (i + 1) / 5
                cfg["VISUALIZER"]["trkBox"]["alpha"] = alpha
                trackViz.draw_det_bboxes(trk_res.trk1[hist_token], trans, **cfg["VISUALIZER"]["trkBox"])
                trackViz2.draw_det_bboxes(trk_res.trk2[hist_token], trans, **cfg["VISUALIZER"]["trkBox"])
            cfg["VISUALIZER"]["trkBox"]["alpha"] = origin_alpha

        car_imgFile = ""
        if cfg["VISUALIZER"]["daynight"]:
            car_imgFile = "/data/Nuscenes/GPT_blackCar1.png" 
        else:
            car_imgFile = "/data/Nuscenes/car1.png"

        start = time.time()
        trackViz.draw_ego_car(img_src=car_imgFile)
        trackViz2.draw_ego_car(img_src=car_imgFile)
        end = time.time()
        print(f"Draw ego time: {(end - start) * 1000:.2f} ms")
        start = time.time()
        if cfg["VISUALIZER"]["lidarPts"]["draw"]:
            trackViz.draw_lidar_pts(token, **cfg["VISUALIZER"]["lidarPts"])
            trackViz2.draw_lidar_pts(token, **cfg["VISUALIZER"]["lidarPts"])
        end = time.time()
        print(f"Lidar pts time: {(end - start) * 1000:.2f} ms")
        if cfg["VISUALIZER"]["radarPts"]["draw"]:
            trackViz.draw_radar_pts(radar_pcs, trans, **cfg["VISUALIZER"]["radarPts"])
            trackViz2.draw_radar_pts(radar_pcs, trans, **cfg["VISUALIZER"]["radarPts"])
        start = time.time()
        if cfg["VISUALIZER"]["groundTruth"]["draw"]:
            trackViz.draw_det_bboxes(gt, trans, **cfg["VISUALIZER"]["groundTruth"])
            trackViz2.draw_det_bboxes(gt, trans, **cfg["VISUALIZER"]["groundTruth"])
        end = time.time()
        print(f"Draw gt time: {(end - start) * 1000:.2f} ms")
        start = time.time()
        if cfg["VISUALIZER"]["trkBox"]["draw"]:
            trackViz.draw_det_bboxes(trk1, trans, **cfg["VISUALIZER"]["trkBox"])
            trackViz2.draw_det_bboxes(trk2, trans, **cfg["VISUALIZER"]["trkBox"])
        end = time.time()
        print(f"Draw trk time: {(end - start) * 1000:.2f} ms")
        start = time.time()
        if cfg["VISUALIZER"]["radarSeg"]["draw"]:
            radarSeg_1 = trk_res.get_radarSeg(1, token)
            radarSeg_2 = trk_res.get_radarSeg(2, token)
            trackViz.draw_radar_seg(radarSeg_1, trans, **cfg["VISUALIZER"]["radarSeg"])
            trackViz2.draw_radar_seg(radarSeg_2, trans, **cfg["VISUALIZER"]["radarSeg"])
            # radarAvg_1 = trk_res.get_avgRadarSeg(1, token)
            # radarAvg_2 = trk_res.get_avgRadarSeg(2, token)
            # trackViz.draw_radar_pts(radarAvg_1, trans, BGRcolor=(3, 169, 252), radius=6)
            # trackViz2.draw_radar_pts(radarAvg_2, trans, BGRcolor=(3, 169, 252), radius=6)
        end = time.time()
        print(f"Draw radarSeg time: {(end - start) * 1000:.2f} ms")
        for win in winList:
            if cfg["VISUALIZER"]["detBox"]["draw"]:
                win.draw_det_bboxes(det, trans, **cfg["VISUALIZER"]["detBox"])
        start = time.time()
        if cfg["VISUALIZER"]["camera"]["draw"]:
            trk1 = trk1 if cfg["VISUALIZER"]["trkBox"]["draw"] else []
            trk2 = trk2 if cfg["VISUALIZER"]["trkBox"]["draw"] else []
            trackViz.draw_camera_images(token, trk1, **cfg["VISUALIZER"]["trkBox"])
            trackViz2.draw_camera_images(token, trk2, **cfg["VISUALIZER"]["trkBox"])
        end = time.time()
        print(f"Draw camera time: {(end - start) * 1000:.2f} ms")
        start = time.time()
        for win in winList:
            win.show()
        end = time.time()
        print(f"Show time: {(end - start) * 1000:.2f} ms (include grid and boundary)")

        viz_end = time.time()
        if args.show_delay:
            print(f"{frame_name:<30} viz delay:{(viz_end - viz_start) / 1e-3: .2f} ms")

        print("---")
        

if __name__ == "__main__":
    main(get_parser())