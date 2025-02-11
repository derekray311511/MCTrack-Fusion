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
from fishUtils.visualizer import TrackVisualizer
from fishUtils.nusc_eval import nusc_eval
from fishUtils.nusc_util import nusc_dataset
from fishUtils.custom_eval import evaluate_nuscenes, TrackingEvaluation


class res_data:
    def __init__(
        self, 
        det_path=None,
        trk_path_1=None,
        trk_path_2=None,
        radarSeg_path=None,
        radarTrk_path=None,
    ):
        self.det = self.load_det(det_path)
        self.trk1 = self.load_trk(trk_path_1)
        self.trk2 = self.load_trk(trk_path_2)
        # self.radarSeg = self.load_trk(radarSeg_path)
        # self.radarTrk = self.load_trk(radarTrk_path)

    def load_det(self, path):
        with open(path, 'rb') as f:
            data = json.load(f)['results']
        return data

    def load_trk(self, path):
        with open(path, 'rb') as f:
            data = json.load(f)['results']
        return data

    def load_radarSeg(self, token):
        seg_f = self.radarSeg[token]
        seg = np.zeros((len(seg_f), 7))
        for i in range(len(seg_f)):
            seg[i][:3] = seg_f[i]['translation']
            seg[i][3:5] = seg_f[i]['velocity']
            seg[i][5] = seg_f[i]['segment_id']
            seg[i][6] = seg_f[i]['category']
        return seg

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
    trk_res = res_data(
        det_path="/data/nuscenes/detections/centerpoint/val.json",
        trk_path_1=f"results/nuscenes/20241014_100616/results.json",
        trk_path_2=f"results/nuscenes/20241014_100616/results.json",
    )
    winName1 = 'LiDAR'
    # winName2 = 'LiDAR'
    cfg["VISUALIZER"]["trk_res"] = {
        'draw': True,
        'colorName': True,
        'colorID': False,
        'draw_vel': True,
        'draw_id': False,
        'draw_name': False,
        'draw_score': False,
        'draw_hist': False, # Could cause slow visualization
        'legend': True, 
        'thickness': 8,
        # 'alpha': 0.7, # slow
    }
    cfg["VISUALIZER"]["analyze"] = {
        'draw_id': False,
        'draw_name': False,
        'draw_score': False,
        'draw_vel': True, 
        'legend': True, 
    }
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
    # trackViz2 = TrackVisualizer(
    #     windowName=winName2,
    #     **cfg["VISUALIZER"],
    # )
    winList = [trackViz]
    setWinPos(cfg["VISUALIZER"]["screenSize"], winList)

    frames = dataset.get_frames_meta()
    frames = dataset.add_key_frames_info(frames)
    gts = dataset.get_groundTruth()
    if 'velocity' not in gts[0]['anns'][0]:
        for gt in gts:
            for obj in gt['anns']:
                obj['velocity'] = np.array([0.0, 0.0])
    len_frames = len(frames)
    max_idx = len_frames - 1
    idx = -1
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
        elif key == ord('i'):
            cfg["VISUALIZER"]["trk_res"]["draw_id"] = not cfg["VISUALIZER"]["trk_res"]["draw_id"]
        elif key == ord('c'):
            # cfg["VISUALIZER"]["groundTruth"]["colorName"] = not cfg["VISUALIZER"]["groundTruth"]["colorName"]
            cfg["VISUALIZER"]["camera"]["draw"] = not cfg["VISUALIZER"]["camera"]["draw"]
        elif key == ord('n'):
            cfg["VISUALIZER"]["trk_res"]["draw_name"] = not cfg["VISUALIZER"]["trk_res"]["draw_name"]
            cfg["VISUALIZER"]["detBox"]["draw_name"] = cfg["VISUALIZER"]["trk_res"]["draw_name"]
        elif key == ord('s'):
            cfg["VISUALIZER"]["trk_res"]["draw_score"] = not cfg["VISUALIZER"]["trk_res"]["draw_score"]
            cfg["VISUALIZER"]["detBox"]["draw_score"] = cfg["VISUALIZER"]["trk_res"]["draw_score"]
        elif key == ord('1'):
            cfg["VISUALIZER"]["detBox"]["draw"] = not cfg["VISUALIZER"]["detBox"]["draw"]
        elif key == ord('h'):
            cfg["VISUALIZER"]["trk_res"]["draw_hist"] = not cfg["VISUALIZER"]["trk_res"]["draw_hist"]
        elif key == ord('t'):
            cfg["VISUALIZER"]["trk_res"]["draw"] = not cfg["VISUALIZER"]["trk_res"]["draw"]
        elif key == ord('y'):
            cfg["VISUALIZER"]["groundTruth"]["draw"] = not cfg["VISUALIZER"]["groundTruth"]["draw"]
        elif key == ord('l'):
            cfg["VISUALIZER"]["lidarPts"]["draw"] = not cfg["VISUALIZER"]["lidarPts"]["draw"]
        elif key == ord('r'):
            cfg["VISUALIZER"]["radarPts"]["draw"] = not cfg["VISUALIZER"]["radarPts"]["draw"]
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
        frame_name = "{}-{}".format(timestamp, token)

        radar_pcs = dataset.get_radar_pcs(token, max_stack=1) # Max is 7
        trans = dataset.get_4f_transform(ego_pose, inverse=True)
        viz_start = time.time()
        if cfg["VISUALIZER"]["trk_res"]["draw_hist"]:
            hist_num = 4
            for i in reversed(range(hist_num)):
                if idx - i - 1 < frames[idx]['first_frame_idx']:
                    continue
                token = frames[idx - i - 1]['token']
                alpha = 1.0 - 0.9 * (i + 1) / 5
                trackViz.draw_det_bboxes(trk_res.trk1[token], trans, **cfg["VISUALIZER"]["trk_res"], alpha=alpha)
                trackViz2.draw_det_bboxes(trk_res.trk2[token], trans, **cfg["VISUALIZER"]["trk_res"], alpha=alpha)
        trackViz.draw_ego_car(img_src="/data/car1.png")
        if cfg["VISUALIZER"]["lidarPts"]["draw"]:
            trackViz.draw_lidar_pts(token, **cfg["VISUALIZER"]["lidarPts"])
        if cfg["VISUALIZER"]["radarPts"]["draw"]:
            trackViz.draw_radar_pts(radar_pcs, trans, **cfg["VISUALIZER"]["radarPts"])
        if cfg["VISUALIZER"]["trk_res"]["draw"]:
            trackViz.draw_det_bboxes(trk1, trans, **cfg["VISUALIZER"]["trk_res"])
        if cfg["VISUALIZER"]["groundTruth"]["draw"]:
            trackViz.draw_det_bboxes(gt, trans, **cfg["VISUALIZER"]["groundTruth"])
        if cfg["VISUALIZER"]["camera"]["draw"]:
            trackViz.draw_camera_images(token)
        # trackViz2.draw_ego_car(img_src="/data/car1.png")
        # trackViz2.draw_det_bboxes(trk2, trans, **cfg["VISUALIZER"]["trk_res"])
        # trackViz2.draw_det_bboxes(gt, trans, **cfg["VISUALIZER"]["groundTruth"])
        for win in winList:
            if cfg["VISUALIZER"]["detBox"]["draw"]:
                win.draw_det_bboxes(det, trans, **cfg["VISUALIZER"]["detBox"])
            win.show()

        viz_end = time.time()
        if args.show_delay:
            print(f"viz delay:{(viz_end - viz_start) / 1e-3: .2f} ms")

        print("---")
        

if __name__ == "__main__":
    main(get_parser())