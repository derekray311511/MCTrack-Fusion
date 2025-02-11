# Modified from 
# nuScenes dev-kit. 
# https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/tests/test_evaluate.py

import json
import os
import random
import shutil
from typing import Dict

import numpy as np
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.constants import DETECTION_NAMES
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.utils import category_to_detection_name, detection_name_to_rel_attributes
from nuscenes.utils.splits import create_splits_scenes

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def nusc_eval_det():
    
    # # Detection evaluation
    # dataroot = "/data/nuscenes/"
    # result_path = "/data/nuscenes/detections/"
    # detector = "BEVFusion2"
    # detection_path = os.path.join(result_path, detector, 'val.json')
    # out_dir = os.path.join(result_path, detector, 'eval')

    # Detection eval for detection fusion
    dataroot = "/data/nuscenes/"
    result_path = "/data/early_fusion_track_results/2024-11-13-18:48:4_noFusePedTruck(LRVelRefined_wTh)/"
    detection_path = os.path.join(result_path, 'detection_fusion_res.json')
    out_dir = os.path.join(result_path, 'detection_fusion_eval')

    # # temp
    # dataroot = "/data/nuscenes/"
    # result_path = "/data/track_result_bboxth-0.0/"
    # detection_path = os.path.join(result_path, 'detection_result.json')
    # out_dir = os.path.join(result_path, 'detection_eval')

    # # Tracking evaluation
    # dataroot = "/data/nuscenes/"
    # result_path = "/data/early_fusion_track_results/2024-11-08-12:15:6_noFusePedTruck(LRVelRefined)"
    # detection_path = os.path.join(result_path, 'tracking_result.json')
    # new_detection_path = os.path.join(result_path, 'tracking_result_detectionFormat.json')
    # out_dir = os.path.join(result_path, 'detection_eval')

    # # Format for eval
    # new_detection_path = os.path.join(result_path, 'temp_detection_result.json')
    # with open(detection_path, 'r') as f:
    #     data = json.load(f)
    #     results = data['results']
    #     new_results = {}
    #     # use tqdm for iterating over the results
    #     for token, tracks in tqdm(results.items(), desc="Processing tracks format"):
    #         for t in tracks:
    #             t.update({
    #                 'detection_name': t['tracking_name'],
    #                 'detection_score': t['tracking_score'],
    #                 'attribute_name': 'vehicle.moving',
    #             })
    #         sorted_tracks = sorted(tracks, key=lambda x: x['detection_score'], reverse=True)
    #         tracks = sorted_tracks
    #         scores = [t['detection_score'] for t in tracks]
    #         assert scores == sorted(scores, reverse=True)
    #         new_results[token] = tracks
    #     new_data = {
    #         'meta': data['meta'],
    #         'results': new_results,
    #     }
    # with open(new_detection_path, 'w') as f:
    #     json.dump(new_data, f)


    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

    assert os.path.exists(detection_path), f"Detection file {detection_path} does not exist."
    mkdir_or_exist(out_dir)

    cfg = config_factory('detection_cvpr_2019')
    nusc_eval = DetectionEval(
        nusc=nusc, 
        config=cfg, 
        result_path=detection_path, 
        eval_set='val', 
        output_dir=out_dir,
        verbose=True,
    )
    metrics_summary = nusc_eval.main()

if __name__ == '__main__':
    nusc_eval_det()