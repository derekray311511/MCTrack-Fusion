# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np

from tracker.matching import *
from tracker.trajectory import Trajectory
from utils.utils import norm_realative_radian
from fishUtils import RadarDBSCAN, LRSegmentator

class Base3DTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.current_frame_id = None
        self.all_trajs = {}
        self.all_dead_trajs = {}
        self.id_seed = 0
        self.cache_size = 3
        self.track_id_counter = 0
        if cfg["RadarSegmentation"]["method"] == "DBSCAN":
            self.radar_segmentor = RadarDBSCAN(**cfg["RadarSegmentation"]["DBSCAN"])
        elif cfg["RadarSegmentation"]["method"] == "LRSegmentor":
            self.radar_segmentor = LRSegmentator(cfg["RadarSegmentation"]["LRSegmentor"])
        elif cfg["RadarSegmentation"]["method"] == "Mix":
            self.radar_segmentor_1 = LRSegmentator(cfg["RadarSegmentation"]["LRSegmentor"])
            self.radar_segmentor_2 = RadarDBSCAN(**cfg["RadarSegmentation"]["DBSCAN"])

    def get_trajectory_bbox(self, all_trajs):
        track_ids = sorted(all_trajs.keys())
        trajs = []
        for i in track_ids:
            trajs.append(all_trajs[i])
        return trajs

    def predict_before_associate(self):
        for track_id, traj in self.all_trajs.items():
            traj.predict()

    def track_single_frame_fusion(self, frame_info):
        """
        Info: This function is the Radar-Lidar fusion version of track_single_frame()
        Parameters:
            input:
                frame_info: Object containing information about the current frame.
            output:
                output_trajs: Updated trajectories after performing tracking and matching for the current frame.
        """
        self.predict_before_associate()

        trajs = self.get_trajectory_bbox(self.all_trajs)
        trajs_cnt = len(trajs)
        # trajs_cnt, dets_cnt = len(trajs), len(frame_info.bboxes)
        
        # Read Radar data and perform segmentation
        if self.cfg["RadarSegmentation"]["method"] == "DBSCAN":
            radar_data = self.radar_segmentor.stackRadar(frame_info.radar)
            segRadar, assignment = self.radar_segmentor.segmentRadar(radar_data)
        elif self.cfg["RadarSegmentation"]["method"] == "LRSegmentor":
            radar_data = self.radar_segmentor.stackRadar(frame_info.radar)
            RL_match, L_only, R_only, segRadar = self.radar_segmentor.segmentRadar(radar_data, frame_info.bboxes)
            R_only = R_only[R_only[:, -2] == (len(frame_info.radar) - 1)]
        elif self.cfg["RadarSegmentation"]["method"] == "Mix":
            radar_data = self.radar_segmentor_1.stackRadar(frame_info.radar)
            RL_match, L_only, R_only, segRadar_1 = self.radar_segmentor_1.segmentRadar(radar_data, frame_info.bboxes)
            radar_data = R_only[:, :6] # Remove cluster id -1 because R_only is not segmented
            segRadar_2, assignment = self.radar_segmentor_2.segmentRadar(radar_data)
            # Concat two segmentations
            segRadar = np.vstack((segRadar_1, segRadar_2))
            # Preserve only current frame radar points
            segRadar = segRadar[segRadar[:, -2] == (len(frame_info.radar) - 1)]
            # print(f"segRadar shape: {segRadar.shape}")
            R_only = self.radar_segmentor_2.avgSegmentation(segRadar_2)
            if len(R_only) != 0:
                R_only = R_only[R_only[:, -2] == (len(frame_info.radar) - 1)]

        # ======================== First matching ========================
        # Trajectory / LR-match lidar bboxes
        lidar_bboxes_1 = [bbox["lidar_det"] for bbox in RL_match]
        match_res, cost_matrix = match_trajs_and_dets(
            trajs, lidar_bboxes_1, self.cfg
        )
        matched_det_indices_1 = set(match_res[:, 1])

        unmatched_det_indices_1 = np.array(
            [i for i in range(len(lidar_bboxes_1)) if i not in matched_det_indices_1]
        )
        unmatched_det_1 = [lidar_bboxes_1[i] for i in unmatched_det_indices_1]

        unmatched_trajs = {}
        remain_trajs = []
        # Update trajectories with matched detections
        for i in range(trajs_cnt):
            track_id = trajs[i].track_id
            if i in match_res[:, 0]:
                indexes = np.where(match_res[:, 0] == i)[0]
                self.all_trajs[track_id].update(
                    lidar_bboxes_1[match_res[indexes, 1][0]], cost_matrix[indexes][0]
                )
            else:
            #     unmatched_trajs[track_id] = self.all_trajs[track_id]
            #     if not self.cfg["IS_RV_MATCHING"]:
            #         self.all_trajs[track_id].unmatch_update(frame_info.frame_id)
                remain_trajs.append(trajs[i])

        # ======================== Second matching ========================
        # Trajectory / Lidar-only lidar bboxes
        lidar_bboxes_2 = L_only
        remain_trajs_cnt = len(remain_trajs)
        match_res, cost_matrix = match_trajs_and_dets(
            remain_trajs, lidar_bboxes_2, self.cfg
        )
        matched_det_indices_2 = set(match_res[:, 1])

        unmatched_det_indices_2 = np.array(
            [i for i in range(len(lidar_bboxes_2)) if i not in matched_det_indices_2]
        )
        unmatched_det_2 = [lidar_bboxes_2[i] for i in unmatched_det_indices_2]

        remain_trajs_2 = []
        # Update trajectories with matched detections
        for i in range(remain_trajs_cnt):
            track_id = remain_trajs[i].track_id
            if i in match_res[:, 0]:
                indexes = np.where(match_res[:, 0] == i)[0]
                self.all_trajs[track_id].update(
                    lidar_bboxes_2[match_res[indexes, 1][0]], cost_matrix[indexes][0]
                )
                # # if match is successful, remove the trajectory from unmatched_trajs
                # del unmatched_trajs[track_id]
            else:
                # unmatched_trajs[track_id] = self.all_trajs[track_id]
                # if not self.cfg["IS_RV_MATCHING"]:
                #     self.all_trajs[track_id].unmatch_update(frame_info.frame_id)
                remain_trajs_2.append(remain_trajs[i])

        # print(f"bbox: {len(frame_info.bboxes)}")
        # print(f"RL_match: {len(RL_match)}, L_only: {len(L_only)}, R_only: {R_only.shape[0]}")
        # print(f"traj_cnt: {trajs_cnt}, remain_trajs: {remain_trajs_cnt}")

        # ======================== Third matching ========================
        # Trajectory / Radar-only radar points
        radar_match_res, radar_costs = match_trajs_and_radars(remain_trajs_2, R_only, self.cfg)
        # For each match, update the trajectory accordingly.
        radar_only_match = []
        remain_trajs_2_cnt = len(remain_trajs_2)
        for i in range(remain_trajs_2_cnt):
            track_id = remain_trajs_2[i].track_id
            if i in radar_match_res[:, 0]:
                indexes = np.where(radar_match_res[:, 0] == i)[0]
                traj = remain_trajs_2[i]
                radar_point = R_only[radar_match_res[indexes, 1][0]]
                radar_only_match.append(radar_point)
                self.all_trajs[track_id].radar_update(radar_point, frame_info.frame_id, radar_costs[indexes])
            else:
                unmatched_trajs[track_id] = self.all_trajs[track_id]
        # for i, match in enumerate(radar_match_res):
        #     traj_idx, radar_idx = match
        #     traj = remain_trajs_2[traj_idx]
        #     track_id = traj.track_id
        #     radar_point = R_only[radar_idx]
        #     radar_only_match.append(radar_point)
        #     self.all_trajs[track_id].radar_update(radar_point, frame_info.frame_id, radar_costs[i])

        # ======================== Birth and death ========================

        # Create new trajectories for unmatched detections
        unmatched_det = unmatched_det_1 + unmatched_det_2
        # print(f"new Traj cnt: {len(unmatched_det)}")
        for det in unmatched_det:
            self.all_trajs[self.track_id_counter] = Trajectory(
                track_id=self.track_id_counter,
                init_bbox=det,
                cfg=self.cfg,
            )
            self.track_id_counter += 1

        # Remove dead trajectories
        for track_id in list(self.all_trajs.keys()):
            if self.all_trajs[track_id].status_flag == 4:
                self.all_dead_trajs[track_id] = self.all_trajs[track_id]
                del self.all_trajs[track_id]

        output_trajs = self.get_output_trajs(frame_info.frame_id)

        return output_trajs, segRadar, radar_only_match

    def track_single_frame(self, frame_info):
        """
        Info: This function tracks objects in a single frame, performing association between predicted trajectories and detected objects.
        Parameters:
            input:
                frame_info: Object containing information about the current frame.
            output:
                output_trajs: Updated trajectories after performing tracking and matching for the current frame.
        """
        self.predict_before_associate()

        trajs = self.get_trajectory_bbox(self.all_trajs)
        trajs_cnt, dets_cnt = len(trajs), len(frame_info.bboxes)
        
        # Read Radar data and perform segmentation
        if self.cfg["RadarSegmentation"]["method"] == "DBSCAN":
            radar_data = self.radar_segmentor.stackRadar(frame_info.radar)
            segRadar, assignment = self.radar_segmentor.segmentRadar(radar_data)
        elif self.cfg["RadarSegmentation"]["method"] == "LRSegmentor":
            radar_data = self.radar_segmentor.stackRadar(frame_info.radar)
            segRadar, assignment = self.radar_segmentor.segmentRadar(radar_data, frame_info.bboxes)
        elif self.cfg["RadarSegmentation"]["method"] == "Mix":
            radar_data = self.radar_segmentor_1.stackRadar(frame_info.radar)
            segRadar_1, assignment = self.radar_segmentor_1.segmentRadar(radar_data, frame_info.bboxes)
            # Remove radar points that are already assigned to a cluster
            radar_data = radar_data[assignment == -1]
            segRadar_2, assignment = self.radar_segmentor_2.segmentRadar(radar_data)
            # Concat two segmentations
            segRadar = np.vstack((segRadar_1, segRadar_2))
            # Preserve only current frame radar points
            # segRadar = segRadar[segRadar[:, -2] == (len(frame_info.radar) - 1)]
            # print(f"segRadar shape: {segRadar.shape}")

        match_res, cost_matrix = match_trajs_and_dets(
            trajs, frame_info.bboxes, self.cfg
        )
        matched_det_indices = set(match_res[:, 1])

        unmatched_det_indices = np.array(
            [i for i in range(dets_cnt) if i not in matched_det_indices]
        )

        unmatched_trajs = {}
        for i in range(trajs_cnt):
            track_id = trajs[i].track_id
            if i in match_res[:, 0]:
                indexes = np.where(match_res[:, 0] == i)[0]
                self.all_trajs[track_id].update(
                    frame_info.bboxes[match_res[indexes, 1][0]], cost_matrix[indexes][0]
                )
            else:
                unmatched_trajs[track_id] = self.all_trajs[track_id]
                if not self.cfg["IS_RV_MATCHING"]:
                    self.all_trajs[track_id].unmatch_update(frame_info.frame_id)

        init_bboxes = frame_info.bboxes
        # if self.cfg["IS_RV_MATCHING"]:
        #     unmatched_trajs_inbev = self.get_trajectory_bbox(unmatched_trajs)
        #     trajs_cnt_inbev, dets_cnt_inbev = len(unmatched_trajs_inbev), len(
        #         unmatched_det_indices
        #     )
        #     unmatched_dets_inbev = (
        #         np.array(frame_info.bboxes)[unmatched_det_indices].tolist()
        #         if dets_cnt_inbev > 0
        #         else unmatched_det_indices
        #     )

        #     match_res_inbev, cost_matrix_inbev = match_trajs_and_dets(
        #         unmatched_trajs_inbev,
        #         unmatched_dets_inbev,
        #         self.cfg,
        #         frame_info.transform_matrix,
        #         is_rv=True,
        #     )

        #     for i in range(trajs_cnt_inbev):
        #         track_id = unmatched_trajs_inbev[i].track_id
        #         if i in match_res_inbev[:, 0]:
        #             indexes = np.where(match_res_inbev[:, 0] == i)[0]
        #             trk_bbox = self.all_trajs[track_id].bboxes[-1]
        #             det_bbox = unmatched_dets_inbev[
        #                 match_res_inbev[match_res_inbev[:, 0] == i, 1][0]
        #             ]
        #             diff_rot = (
        #                 abs(
        #                     norm_realative_radian(
        #                         trk_bbox.global_yaw - det_bbox.global_yaw
        #                     )
        #                 )
        #                 * 180
        #                 / np.pi
        #             )
        #             dist = np.linalg.norm(
        #                 np.array(trk_bbox.global_xyz) - np.array(det_bbox.global_xyz)
        #             )
        #             if diff_rot > 90 or dist > 5:
        #                 self.all_trajs[track_id].unmatch_update(frame_info.frame_id)
        #                 continue
        #             self.all_trajs[track_id].update(
        #                 det_bbox, float(cost_matrix_inbev[indexes])
        #             )
        #         else:
        #             self.all_trajs[track_id].unmatch_update(frame_info.frame_id)

        #     matched_det_indices = set(match_res_inbev[:, 1])
        #     unmatched_det_indices = np.array(
        #         [i for i in range(dets_cnt_inbev) if i not in matched_det_indices]
        #     )
        #     init_bboxes = unmatched_dets_inbev

        for i in unmatched_det_indices:
            self.all_trajs[self.track_id_counter] = Trajectory(
                track_id=self.track_id_counter,
                init_bbox=init_bboxes[i],
                cfg=self.cfg,
            )
            self.track_id_counter += 1

        for track_id in list(self.all_trajs.keys()):
            if self.all_trajs[track_id].status_flag == 4:
                self.all_dead_trajs[track_id] = self.all_trajs[track_id]
                del self.all_trajs[track_id]

        output_trajs = self.get_output_trajs(frame_info.frame_id)

        return output_trajs, segRadar

    def get_output_trajs(self, frame_id):
        output_trajs = {}
        for track_id in list(self.all_trajs.keys()):
            if self.all_trajs[track_id].status_flag == 1 or frame_id < 3:
                bbox = self.all_trajs[track_id].bboxes[-1]
                if bbox.det_score == self.all_trajs[track_id]._is_filter_predict_box:
                    continue
                output_trajs[track_id] = bbox
                self.all_trajs[track_id].is_output = True
        return output_trajs

    def post_processing(self):
        trajs = {}
        for track_id in self.all_dead_trajs.keys():
            traj = self.all_dead_trajs[track_id]
            traj.filtering()
            trajs[track_id] = traj
        for track_id in self.all_trajs.keys():
            traj = self.all_trajs[track_id]
            traj.filtering()
            trajs[track_id] = traj
        return trajs
