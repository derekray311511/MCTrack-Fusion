# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import re
import os

from tracker.frame import Frame
from tracker.bbox import BBox
from utils.nusc_utils import obtain_box_bottom_corners, filter_bboxes_with_nms


class BaseVersionTrackingDataset:
    def __init__(self, scene_id, scene_data, scene_radar, cfg):
        self.scene_id = scene_id
        self.scene_data = scene_data
        self.scene_radar = scene_radar
        self.cfg = cfg

    def __len__(self):
        return len(self.scene_data)

    def __getitem__(self, index):
        frame_info = self.scene_data[index]
        frame_id = frame_info["frame_id"]
        timestamp = frame_info["timestamp"]
        cur_sample_token = frame_info["cur_sample_token"]
        transform_matrix = frame_info["transform_matrix"]
        bboxes = frame_info["bboxes"]
        frame_radar = self.scene_radar[index]

        cur_frame = Frame(
            frame_id=int(frame_id),
            cur_sample_token=cur_sample_token,
            timestamp=timestamp,
            transform_matrix=transform_matrix,
        )

        bboxes = np.array(
            [
                bbox
                for bbox in bboxes
                if bbox["category"] in self.cfg["CATEGORY_MAP_TO_NUMBER"]
            ]
        )
        if self.cfg["TRACKING_MODE"] == "ONLINE":
            input_score = self.cfg["THRESHOLD"]["INPUT_SCORE"]["ONLINE"]

        else:
            input_score = self.cfg["THRESHOLD"]["INPUT_SCORE"]["OFFLINE"]
        filtered_bboxes = [
            bbox
            for bbox in bboxes
            if bbox["detection_score"]
            > input_score[self.cfg["CATEGORY_MAP_TO_NUMBER"][bbox["category"]]]
        ]
        if self.cfg["DATASET"] == "nuscenes":
            if len(filtered_bboxes) != 0:
                filtered_bboxes = filter_bboxes_with_nms(filtered_bboxes, self.cfg)
        for bbox in filtered_bboxes:
            cur_frame.bboxes.append(BBox(frame_id, bbox))

        cur_frame.radar = frame_radar # [[-n+1_rpc], [-n+2_rpc], [-n+3_rpc], ..., [0_rpc]]
        
        return cur_frame
