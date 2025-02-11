import argparse
import os, sys
import cv2
import time
import json
import math
import numpy as np
import yaml

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from tqdm import tqdm
from copy import deepcopy
from .box_utils import is_points_inside_obb
from .geometry_utils import *

class LRSegmentator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.min_samples = cfg["min_samples"]
        self.lidar_score_th = cfg["lidar_score_th"]
        self.expand_ratio = cfg["expand_ratio"]
        self.id_count = 0

    def reset(self):
        self.id_count = 0

    def BBox2nuscFormat(self, bbox):
        '''
        Convert bbox to nusc format.
        Args:
            bbox: class, bbox information
        return:
            nusc_bbox: dict, bbox information in nusc format
        '''
        nusc_bbox = {
            'translation': bbox.global_xyz,
            'size': bbox.lwh,
            'rotation': bbox.global_orientation,
            'velocity': bbox.global_velocity,
            'detection_name': bbox.category,
            'detection_score': bbox.det_score,
            'sample_token': None
        }
        return nusc_bbox

    def stackRadar(self, radar_data_list):
        '''
        Stack Radar data from multiple frames.
        Add frame id to radar data so that we can know which frame the stack radar data belongs to.
        Args:
            radar_data_list: list of np.array, shape: (n, 5), [x, y, z, vx, vy]
        return:
            radar_data_list: list of np.array, shape: (n, 6), [x, y, z, vx, vy, frame_id]
        
        radar_data_list[-1] is the latest frame. radar_data_list[0] is the oldest frame.
        '''
        for i, radar_data in enumerate(radar_data_list):
            radar_data_list[i] = np.hstack([radar_data, np.ones((len(radar_data), 1)) * i])

        return np.vstack(radar_data_list)

    def segmentRadar(self, radar_data, lidar_det):
        '''
        Run the segmentation algorithm.
        Args:
            radar_data: np.array, shape: (n, 6), [x, y, z, vx, vy, frame_id]
            lidar_det: class bbox, lidar detection
        return:
            # radar_segmentation: np.array, shape: (n, 7), [x, y, z, vx, vy, frame_id, cluster_id]
            # assignment: np.array, shape: (n,), cluster assignment for each point in radar_data.
            radar_lidar_match: list of dict, each containing a LiDAR detection and its matched radar points.
                [{'lidar_det': bbox, 'radar_segment': np.array of shape (m, 7)}]
            lidar_only: list of bbox, LiDAR detections not matched with any radar point.
            radar_only: np.array, shape: (k, 7), [x, y, z, vx, vy, frame_id, cluster_id(-1)]
        '''
        # Prepare data
        raw_lidar_det = deepcopy(lidar_det)
        lidar_det = [self.BBox2nuscFormat(det) for det in lidar_det]
        lidar_det.sort(reverse=True, key=lambda x: x["detection_score"])
        radar_pts = np.array([p[:2] for p in radar_data])
        radar_targets = np.array(radar_data)

        # Initialize cluster assignments
        radar_targets = np.hstack([radar_targets, -np.ones((len(radar_targets), 1))])
        remaining_idxs = np.arange(len(radar_pts))

        matched_lidar_ids = {}  # Map cluster_id to its corresponding LiDAR detection

        for i, det in enumerate(lidar_det):
            if det['detection_score'] < self.lidar_score_th:
                continue
            center = det['translation'][:2]
            size = [det['size'][0] * self.expand_ratio[0], det['size'][1] * self.expand_ratio[1]]
            row, pitch, yaw = euler_from_quaternion(q_to_xyzw(det['rotation']))
            inbox_idxs = is_points_inside_obb(radar_pts[remaining_idxs], center, size, yaw)
            
            if np.sum(inbox_idxs) < self.min_samples:
                continue

            # Assign cluster ID to matched radar points
            cluster_id = self.id_count
            radar_targets[remaining_idxs[inbox_idxs], -1] = cluster_id
            remaining_idxs = remaining_idxs[~inbox_idxs]
            matched_lidar_ids[cluster_id] = i  # Link cluster_id to the LiDAR detection index
            self.id_count += 1

        # Split results
        radar_lidar_match = []
        for cluster_id, bbox_idx in matched_lidar_ids.items():
            matched_lidar_bbox = raw_lidar_det[bbox_idx]
            matched_radar_points = radar_targets[radar_targets[:, -1] == cluster_id]  # Include cluster ID
            radar_lidar_match.append({
                'lidar_det': matched_lidar_bbox,
                'radar_segment': matched_radar_points
            })

        radar_only = radar_targets[radar_targets[:, -1] == -1]  # Include cluster ID
        lidar_only = [det for i, det in enumerate(raw_lidar_det) if i not in matched_lidar_ids.values()]

        return radar_lidar_match, lidar_only, radar_only, radar_targets