# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import json, yaml
import logging
import copy
import argparse
import os
import time
import multiprocessing
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime
from functools import partial
from tracker.base_tracker import Base3DTracker
from dataset.baseversion_dataset import BaseVersionTrackingDataset
from evaluation.static_evaluation.kitti.evaluation_HOTA.scripts.run_kitti import (
    eval_kitti,
)
from evaluation.static_evaluation.nuscenes.eval import eval_nusc
from evaluation.static_evaluation.waymo.eval import eval_waymo
from utils.kitti_utils import save_results_kitti
from utils.nusc_utils import save_results_nuscenes, save_results_nuscenes_for_motion
from utils.waymo_utils.convert_result import save_results_waymo
from fishUtils import TrackVisualizer
from utils.utils import transform_yaw2quaternion
import shutil

total_index = 0

def run(scene_id, scenes_data, radar_data, cfg, args, tracking_results, visualizer=None, gts=None):
    """
    Info: This function tracks objects in a given scene, processes frame data, and stores tracking results.
    Parameters:
        input:
            scene_id: ID of the scene to process.
            scenes_data: Dictionary with scene data.
            cfg: Configuration settings for tracking.
            args: Additional arguments.
            tracking_results: Dictionary to store results.
        output:
            tracking_results: Updated tracking results for the scene.
    """
    global total_index
    
    scene_data = scenes_data[scene_id]
    scene_radar = radar_data[scene_id]
    dataset = BaseVersionTrackingDataset(scene_id, scene_data, scene_radar, cfg=cfg)
    tracker = Base3DTracker(cfg=cfg)
    all_trajs = {}

    for index in tqdm(range(len(dataset)), desc=f"Processing {scene_id}"):
        frame_gt = gts[total_index]['anns']
        frame_info = dataset[index]
        frame_id = frame_info.frame_id
        cur_sample_token = frame_info.cur_sample_token
        all_traj, segRadar, radar_only_match = tracker.track_single_frame_fusion(frame_info)
        result_info = {
            "frame_id": frame_id,
            "cur_sample_token": cur_sample_token,
            "trajs": copy.deepcopy(all_traj),
            "transform_matrix": frame_info.transform_matrix,
        }
        all_trajs[frame_id] = copy.deepcopy(result_info)
        
        if visualizer:
            if visualizer.play:
                key = cv2.waitKey(int(visualizer.duration * 1000))
            else:
                key = cv2.waitKey(0)
            if key == 27: # esc
                cv2.destroyAllWindows()
                exit(0)
            elif key == 32: # space
                visualizer.play = not visualizer.play
            elif key == 43: # +
                visualizer.duration *= 2
                print(f"Viz duration set to {visualizer.duration}")
            elif key == 45: # -
                visualizer.duration *= 0.5
                print(f"Viz duration set to {visualizer.duration}")
            elif key == ord('g'):
                visualizer.grid = not visualizer.grid
            elif key == ord('l'):
                cfg["VISUALIZER"]["lidarPts"]["draw"] = not cfg["VISUALIZER"]["lidarPts"]["draw"]
            elif key == ord('r'):
                cfg["VISUALIZER"]["radarPts"]["draw"] = not cfg["VISUALIZER"]["radarPts"]["draw"]
            elif key == ord('s'):
                cfg["VISUALIZER"]["radarSeg"]["draw"] = not cfg["VISUALIZER"]["radarSeg"]["draw"]
            elif key == ord('c'):
                cfg["VISUALIZER"]["camera"]["draw"] = not cfg["VISUALIZER"]["camera"]["draw"]
            elif key == ord('i'):
                cfg["VISUALIZER"]["trkBox"]["draw_id"] = not cfg["VISUALIZER"]["trkBox"]["draw_id"]
            elif key == ord('q'): # next scene
                total_index += len(dataset) - index
                break
            # Get trajs and transform matrix
            global2ego = np.array(frame_info.transform_matrix["global2ego"]).reshape(4, 4)
            ego2lidar = np.array(frame_info.transform_matrix["ego2lidar"]).reshape(4, 4)
            trans = global2ego
            # Transform trajs to visualize type
            sample_results = []
            for track_id, bbox in all_traj.items():
                global_orientation = transform_yaw2quaternion(bbox.global_yaw)
                box_result = {
                    "sample_token": cur_sample_token,
                    "translation": [
                        float(bbox.global_xyz_lwh_yaw_fusion[0]),
                        float(bbox.global_xyz_lwh_yaw_fusion[1]),
                        float(bbox.global_xyz_lwh_yaw_fusion[2]),
                    ],
                    "size": [
                        float(bbox.lwh_fusion[1]),
                        float(bbox.lwh_fusion[0]),
                        float(bbox.lwh_fusion[2]),
                    ],
                    "rotation": [
                        float(global_orientation[0]),
                        float(global_orientation[1]),
                        float(global_orientation[2]),
                        float(global_orientation[3]),
                    ],
                    "velocity": [
                        float(bbox.global_velocity[0]),
                        float(bbox.global_velocity[1]),
                    ],
                    "tracking_id": str(track_id),
                    "tracking_name": bbox.category,
                    "tracking_score": bbox.det_score,
                }
                sample_results.append(box_result)
            # Visualize
            cv2.setTrackbarPos('Frame', visualizer.windowName, total_index)
            visualizer.draw_ego_car(img_src="/data/Nuscenes/car1.png")
            if cfg["VISUALIZER"]["lidarPts"]["draw"]:
                visualizer.draw_lidar_pts(cur_sample_token, **cfg["VISUALIZER"]["lidarPts"])
            if cfg["VISUALIZER"]["radarPts"]["draw"]:
                # visualizer.draw_radar_pts(np.vstack(frame_info.radar), trans, **cfg["VISUALIZER"]["radarPts"])
                visualizer.draw_radar_pts(radar_only_match, trans, **cfg["VISUALIZER"]["radarPts"])
            if cfg["VISUALIZER"]["radarSeg"]["draw"]:
                visualizer.draw_radar_seg(np.delete(segRadar, 5, axis=1), trans, **cfg["VISUALIZER"]["radarSeg"])
            visualizer.draw_det_bboxes(nusc_det=sample_results, trans=trans, **cfg["VISUALIZER"]["trkBox"])
            visualizer.draw_det_bboxes(frame_gt, trans, **cfg["VISUALIZER"]["groundTruth"])
            if cfg["VISUALIZER"]["camera"]["draw"]:
                visualizer.draw_camera_images(cur_sample_token)
            visualizer.show()
        
        total_index += 1

    if cfg["TRACKING_MODE"] == "GLOBAL":
        trajs = tracker.post_processing()
        for index in tqdm(
            range(len(dataset)), desc=f"Trajectory Postprocessing {scene_id}"
        ):
            frame_id = dataset[index].frame_id
            for track_id in sorted(list(trajs.keys())):
                for bbox in trajs[track_id].bboxes:
                    if (
                        bbox.frame_id == frame_id
                        and bbox.is_interpolation
                        and track_id not in all_trajs[frame_id]["trajs"].keys()
                    ):
                        all_trajs[frame_id]["trajs"][track_id] = bbox

        for index in tqdm(
            range(len(dataset)), desc=f"Trajectory Postprocessing {scene_id}"
        ):
            frame_id = dataset[index].frame_id
            for track_id in sorted(list(trajs.keys())):
                det_score = 0
                for bbox in trajs[track_id].bboxes:
                    det_score = bbox.det_score
                    break
                if (
                    track_id in all_trajs[frame_id]["trajs"].keys()
                    and det_score <= cfg["THRESHOLD"]["GLOBAL_TRACK_SCORE"]
                ):
                    del all_trajs[frame_id]["trajs"][track_id]

    tracking_results[scene_id] = all_trajs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTrack")
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti",
        help="Which Dataset: kitti/nuscenes/waymo",
    )
    parser.add_argument("--eval", "-e", action="store_true", help="evaluation")
    parser.add_argument("--load_image", "-lm", action="store_true", help="load_image")
    parser.add_argument("--load_point", "-lp", action="store_true", help="load_point")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument("--mode", "-m", action="store_true", help="online or offline")
    parser.add_argument("--process", "-p", type=int, default=1, help="multi-process!")
    parser.add_argument("--viz", "-v", action="store_true", help="visualize")
    args = parser.parse_args()

    if args.dataset == "kitti":
        cfg_path = "./config/kitti.yaml"
    elif args.dataset == "nuscenes":
        cfg_path = "./config/nuscenes.yaml"
    elif args.dataset == "waymo":
        cfg_path = "./config/waymo.yaml"
    if args.mode:
        cfg_path = cfg_path.replace(".yaml", "_offline.yaml")

    cfg = yaml.load(open(cfg_path, "r"), Loader=yaml.Loader)

    additional_name = "_Mix_threeStage_th-0.25+MixTrack_MixUpdate"
    save_path = os.path.join(
        os.path.dirname(cfg["SAVE_PATH"]),
        cfg["DATASET"],
        datetime.now().strftime("%Y%m%d_%H%M%S") + additional_name,
    )
    os.makedirs(save_path, exist_ok=True)
    cfg["SAVE_PATH"] = save_path

    # Copy config file to save path
    shutil.copy(cfg_path, os.path.join(save_path, os.path.basename(cfg_path)))

    start_time = time.time()

    if args.viz:
        trackViz = TrackVisualizer(
            windowName='Track',
            **cfg["VISUALIZER"],
        )
        cv2.createTrackbar('Frame', trackViz.windowName, 0, 6018, lambda x: None)
    else:
        trackViz = None

    with open(cfg["GT_PATH"], 'rb') as f:
        gts = json.load(f)['samples']
        print(f"GroundTruth loaded successfully. {len(gts)} samples.")

    with open(cfg["RADAR_PATH"], "r", encoding="utf-8") as file:
        radar_data = json.load(file)
        print("Radar data loaded successfully.")

    detections_root = os.path.join(
        cfg["DETECTIONS_ROOT"], cfg["DETECTOR"], cfg["SPLIT"] + ".json"
    )
    with open(detections_root, "r", encoding="utf-8") as file:
        print(f"Loading data from {detections_root}...")
        data = json.load(file)
        print("Data loaded successfully.")

    if args.debug:
        if args.dataset == "kitti":
            scene_lists = [str(scene_id).zfill(4) for scene_id in cfg["TRACKING_SEQS"]]
        elif args.dataset == "nuscenes":
            scene_lists = [scene_id for scene_id in data.keys()][:2]
        else:
            scene_lists = [scene_id for scene_id in data.keys()][:2]
    else:
        scene_lists = [scene_id for scene_id in data.keys()]

    manager = multiprocessing.Manager()
    tracking_results = manager.dict()
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        func = partial(
            run, scenes_data=data, radar_data=radar_data, cfg=cfg, args=args, tracking_results=tracking_results, visualizer=trackViz, gts=gts
        )
        pool.map(func, scene_lists)
        pool.close()
        pool.join()
    else:
        for scene_id in tqdm(scene_lists, desc="Running scenes"):
            run(scene_id, data, radar_data, cfg, args, tracking_results, visualizer=trackViz, gts=gts)
    tracking_results = dict(tracking_results)

    if args.dataset == "kitti":
        save_results_kitti(tracking_results, cfg)
        if args.eval:
            eval_kitti(cfg)
    if args.dataset == "nuscenes":
        save_results_nuscenes(tracking_results, save_path)
        save_results_nuscenes_for_motion(tracking_results, save_path)
        if args.eval:
            eval_nusc(cfg)
    elif args.dataset == "waymo":
        save_results_waymo(tracking_results, save_path)
        if args.eval:
            eval_waymo(cfg, save_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
