# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import lap

from tracker.cost_function import *
from utils.utils import mask_tras_dets


def Greedy(cost_matrix, thresholds):
    """
    Refer: https://github.com/lixiaoyu2000/Poly-MOT/blob/main/utils/matching.py
    Info: This function implements the Greedy matching algorithm.
    Parameters:
        input:
            cost_matrix: np.array, either 2D or 3D cost matrix with shape [N_cls, N_det, N_tra] or [N_det, N_tra].
                         - N_cls: Number of classes.
                         - N_det: Number of detections.
                         - N_tra: Number of trajectories.
                         Invalid costs are represented by np.inf.
            thresholds: dict, class-specific matching thresholds to restrict false positive matches.
        output:
            m_det: list, indices of matched detections.
            m_tra: list, indices of matched trajectories.
            um_det: np.array, indices of unmatched detections.
            um_tra: np.array, indices of unmatched trajectories.
            costs: np.array, matching costs for the matched pairs.
    """
    assert cost_matrix.ndim == 2 or cost_matrix.ndim == 3, "cost matrix must be valid."
    if cost_matrix.ndim == 2:
        cost_matrix = cost_matrix[None, :, :]
    assert (
        len(thresholds) == cost_matrix.shape[0]
    ), "the number of thresholds should be egual to cost matrix number."

    # solve cost matrix
    m_det, m_tra = [], []
    costs = []
    num_det, num_tra = cost_matrix.shape[1:]
    for cls_idx, cls_cost in enumerate(cost_matrix):
        for det_idx in range(num_det):
            tra_idx = cls_cost[det_idx].argmin()
            if cls_cost[det_idx][tra_idx] <= thresholds[cls_idx]:
                costs.append(cls_cost[det_idx, tra_idx])
                cost_matrix[cls_idx, :, tra_idx] = 1e18
                m_det.append(det_idx)
                m_tra.append(tra_idx)

    # unmatched tra and det
    if len(m_det) == 0:
        um_det, um_tra = np.arange(num_det), np.arange(num_tra)
    else:
        um_det = np.setdiff1d(np.arange(num_det), np.array(m_det))
        um_tra = np.setdiff1d(np.arange(num_tra), np.array(m_tra))

    return m_det, m_tra, um_det, um_tra, np.array(costs)


def Hungarian(cost_matrix, thresholds):
    """
    Refer: https://github.com/lixiaoyu2000/Poly-MOT/blob/main/utils/matching.py
    Info: This function implements the Hungarian algorithm using the Linear Assignment Problem solver (lapjv).
    Parameters:
        input:
            cost_matrix: np.array, either 2D or 3D cost matrix with shape [N_cls, N_det, N_tra] or [N_det, N_tra].
                         Invalid costs are represented by np.inf.
            thresholds: dict, class-specific matching thresholds to restrict false positive matches.
        output:
            m_det: list, indices of matched detections.
            m_tra: list, indices of matched trajectories.
            um_det: np.array, indices of unmatched detections.
            um_tra: np.array, indices of unmatched trajectories.
            costs: np.array, matching costs for the matched pairs.
    """
    assert cost_matrix.ndim == 2 or cost_matrix.ndim == 3, "cost matrix must be valid."
    if cost_matrix.ndim == 2:
        cost_matrix = cost_matrix[None, :, :]
    assert (
        len(thresholds) == cost_matrix.shape[0]
    ), "the number of thresholds should be equal to cost matrix number."

    # solve cost matrix
    m_det, m_tra = [], []
    costs = []
    for cls_idx, cls_cost in enumerate(cost_matrix):
        _, x, y = lap.lapjv(cls_cost, extend_cost=True, cost_limit=thresholds[cls_idx])
        for ix, mx in enumerate(x):
            if mx >= 0:
                assert (ix not in m_det) and (mx not in m_tra)
                m_det.append(ix)
                m_tra.append(mx)
                costs.append(cls_cost[ix, mx])

    # unmatched tra and det
    num_det, num_tra = cost_matrix.shape[1:]
    if len(m_det) == 0:
        um_det, um_tra = np.arange(num_det), np.arange(num_tra)
    else:
        um_det = np.setdiff1d(np.arange(num_det), np.array(m_det))
        um_tra = np.setdiff1d(np.arange(num_tra), np.array(m_tra))

    return m_det, m_tra, um_det, um_tra, np.array(costs)


def match_trajs_and_dets(trajs, dets, cfg, transform_matrix=None, is_rv=False):
    """
    Info: This function matches trajectories with detections using a cost matrix and a specified matching algorithm (Hungarian or Greedy).
    Parameters:
        input:
            trajs: List of trajectory objects.
            dets: List of detection objects.
            cfg: Configuration dictionary, includes matching and category information.
            transform_matrix: (Optional) Matrix for transforming coordinates (if needed for the matching process).
            is_rv: bool, flag to indicate whether to use re-projected view (RV) or bird's eye view (BEV) for matching.
        output:
            matched_indices: np.array, array of matched trajectory and detection indices (shape [n, 2]).
            costs: np.array, corresponding costs for each match.
    """
    if len(trajs) == 0 or len(dets) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)

    cost_matrix, trajs_category, dets_category = cost_calculate_general(
        trajs, dets, cfg, transform_matrix, is_rv
    )
    match_type = "RV" if is_rv else "BEV"

    category_map = cfg["CATEGORY_MAP_TO_NUMBER"]
    vectorized_map = np.vectorize(category_map.get)
    dets_label = vectorized_map(dets_category)
    trajs_label = vectorized_map(trajs_category)

    cls_num = len(cfg["CATEGORY_LIST"])
    valid_mask = mask_tras_dets(cls_num, trajs_label, dets_label)
    trans_valid_mask = valid_mask.transpose(0, 2, 1)

    trans_cost_matrix = cost_matrix.T
    trans_cost_matrix = trans_cost_matrix[None, :, :].repeat(cls_num, axis=0)
    trans_cost_matrix[np.where(~trans_valid_mask)] = np.inf

    if min(cost_matrix.shape) > 0:
        if cfg["MATCHING"][match_type]["MATCHING_MODE"] == "Hungarian":
            m_det, m_tra, um_det, um_tra, costs = Hungarian(
                trans_cost_matrix,
                cfg["THRESHOLD"][match_type]["COST_THRE"],
            )
            assert len(m_det) == len(m_tra)
            matched_indices = np.column_stack((m_tra, m_det))
        elif cfg["MATCHING"][match_type]["MATCHING_MODE"] == "Greedy":
            m_det, m_tra, um_det, um_tra, costs = Greedy(
                trans_cost_matrix,
                cfg["THRESHOLD"][match_type]["COST_THRE"],
            )
            assert len(m_det) == len(m_tra)
            matched_indices = np.column_stack((m_tra, m_det))
    else:
        matched_indices = np.empty(shape=(0, 2))

    return matched_indices, costs


def cost_calculate_general(trajs, dets, cfg, transform_matrix, is_rv=False):
    cost_matrix = np.zeros((len(trajs), len(dets)))

    def choose_cost_func(is_rv, cost_mode):
        if is_rv:
            if cost_mode == "IOU_2D":
                cal_cost_func = cal_iou_inrv
            elif cost_mode == "GIOU_2D":
                cal_cost_func = cal_giou_inrv
            elif cost_mode == "DIOU_2D":
                cal_cost_func = cal_diou_inrv
            elif cost_mode == "SDIOU_2D":
                cal_cost_func = cal_sdiou_inrv
        else:
            if cost_mode == "RO_GDIOU_3D":
                cal_cost_func = cal_rotation_gdiou_inbev
        return cal_cost_func

    for t, trk in enumerate(trajs):
        for d, det in enumerate(dets):
            # trk_category = cfg["CATEGORY_MAP_TO_NUMBER"][trajs[0].bboxes[-1].category]
            trk_category = cfg["CATEGORY_MAP_TO_NUMBER"][trk.bboxes[-1].category]
            match_type = "BEV"
            if is_rv:
                match_type = "RV"
            cost_mode = cfg["MATCHING"][match_type]["COST_MODE"][trk_category]
            cost_state = cfg["MATCHING"][match_type]["COST_STATE"][trk_category]
            cost_state_predict_ratio = cfg["THRESHOLD"]["COST_STATE_PREDICT_RATION"][
                trk_category
            ]
            cal_cost_func = choose_cost_func(is_rv, cost_mode)
            pred_cost = cal_cost_func(trk, det, cfg, cal_flag="Predict")
            no_pred_cost = cal_cost_func(trk, det, cfg, cal_flag="BackPredict")

            if cost_state == "Predict":
                cost_matrix[t, d] = pred_cost
            elif cost_state == "BackPredict":
                cost_matrix[t, d] = no_pred_cost
            elif cost_state == "Fusion":
                cost_matrix[t, d] = (
                    cost_state_predict_ratio * pred_cost
                    + (1 - cost_state_predict_ratio) * no_pred_cost
                )

    trajs_category = np.array([traj.bboxes[-1].category for traj in trajs])
    dets_category = np.array([det.category for det in dets])
    same_category_mask = (trajs_category[:, np.newaxis] == dets_category).astype(int)
    cost_matrix[same_category_mask == 0] = -np.inf

    return 1 - cost_matrix, trajs_category, dets_category


def match_trajs_and_radars(trajs, radar_points, cfg):
    """
    Matches trajectories with radar points using Global Nearest Neighbor (GNN) matching
    via the Hungarian algorithm.

    Parameters:
        trajs (list): List of trajectory objects. Each trajectory is assumed to have a predicted
                      2D position accessible as traj.bboxes[-1].global_xyz_lwh_yaw_predict.
        radar_points (np.ndarray): Array of radar points (shape: [N, D]), where the first two columns
                                   represent the (x, y) world coordinates.
        cfg (dict): Configuration dictionary. This function looks for a matching threshold under:
                    cfg["THRESHOLD"]["RADAR"]["DIST_THRE"]
                    If not found, a default threshold (e.g., 4.0) is used.

    Returns:
        matched_indices (np.ndarray): An array of shape [M, 2] where each row is [traj_index, radar_index].
        costs (np.ndarray): A 1D array of the matching costs for each matched pair.
    """
    if len(trajs) == 0 or radar_points.shape[0] == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,))

    # -------------------------------------------------------------------------
    # 1. Build the cost matrix between each trajectory and each radar point.
    #    We assume that each trajectory has a Kalman filter (e.g., CV, CA, or CTRA) that
    #    stores the predicted state in .x, with the first two entries being [x, y].
    # -------------------------------------------------------------------------
    pred_positions = np.zeros((len(trajs), 2))
    for i, traj in enumerate(trajs):
        bbox = traj.bboxes[-1]
        pose = bbox.global_xyz_lwh_yaw_predict
        pred_positions[i] = pose[:2]
    radar_positions = radar_points[:, :2]

    # Compute the Euclidean distance for each (trajectory, radar_point) pair.
    # Resulting cost_matrix has shape (n_trajs, n_radars)
    diff = pred_positions[:, None, :] - radar_positions[None, :, :]
    cost_matrix = np.linalg.norm(diff, axis=2)

    # -------------------------------------------------------------------------
    # 2. Retrieve the matching threshold from the configuration.
    #    (You can change the default value as appropriate.)
    # -------------------------------------------------------------------------
    thresholds = cfg.get("THRESHOLD", {}).get("RADAR", {}).get("DIST_THRE", {})

    # -------------------------------------------------------------------------
    # 3. Solve the assignment problem using the Hungarian algorithm.
    #    The lap.lapjv function returns an assignment for each trajectory:
    #      - For each trajectory i, x[i] is the index of the radar point assigned to it,
    #        or -1 if no assignment was made (or if the cost exceeded the threshold).
    # -------------------------------------------------------------------------

    matched_indices = []
    costs = []

    for i, traj in enumerate(trajs):
        category = traj.bboxes[-1].category
        radar_cost_threshold = thresholds.get(category, 4.0)  # 默認閾值為 4.0

        _, x, _ = lap.lapjv(cost_matrix[i:i+1, :], extend_cost=True, cost_limit=radar_cost_threshold)
        
        radar_idx = x[0]
        if radar_idx >= 0 and cost_matrix[i, radar_idx] <= radar_cost_threshold:
            matched_indices.append([i, radar_idx])
            costs.append(cost_matrix[i, radar_idx])

    if len(matched_indices) == 0:
        matched_indices = np.empty((0, 2), dtype=int)
        costs = np.empty((0,))
    else:
        matched_indices = np.array(matched_indices, dtype=int)
        costs = np.array(costs)

    return matched_indices, costs

def match_lidar_and_radar(dets, radar_points, cfg):
    """
    Matches trajectories with radar points using Global Nearest Neighbor (GNN) matching
    via the Hungarian algorithm.

    Parameters:
        trajs (list): List of trajectory objects. Each trajectory is assumed to have a predicted
                      2D position accessible as traj.bboxes[-1].global_xyz_lwh_yaw_predict.
        radar_points (np.ndarray): Array of radar points (shape: [N, D]), where the first two columns
                                   represent the (x, y) world coordinates.
        cfg (dict): Configuration dictionary. This function looks for a matching threshold under:
                    cfg["THRESHOLD"]["RADAR"]["DIST_THRE"]
                    If not found, a default threshold (e.g., 4.0) is used.

    Returns:
        matched_indices (np.ndarray): An array of shape [M, 2] where each row is [traj_index, radar_index].
        costs (np.ndarray): A 1D array of the matching costs for each matched pair.
    """
    if len(dets) == 0 or radar_points.shape[0] == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,))

    # -------------------------------------------------------------------------
    # 1. Build the cost matrix between each trajectory and each radar point.
    #    We assume that each trajectory has a Kalman filter (e.g., CV, CA, or CTRA) that
    #    stores the predicted state in .x, with the first two entries being [x, y].
    # -------------------------------------------------------------------------
    pred_positions = np.zeros((len(dets), 2))
    for i, det in enumerate(dets):
        bbox = det
        pose = bbox.global_xyz_lwh_yaw
        pred_positions[i] = pose[:2]
    radar_positions = radar_points[:, :2]

    # Compute the Euclidean distance for each (trajectory, radar_point) pair.
    # Resulting cost_matrix has shape (n_trajs, n_radars)
    diff = pred_positions[:, None, :] - radar_positions[None, :, :]
    cost_matrix = np.linalg.norm(diff, axis=2)

    # -------------------------------------------------------------------------
    # 2. Retrieve the matching threshold from the configuration.
    #    (You can change the default value as appropriate.)
    # -------------------------------------------------------------------------
    thresholds = cfg.get("THRESHOLD", {}).get("RADAR", {}).get("DIST_THRE", {})

    # -------------------------------------------------------------------------
    # 3. Solve the assignment problem using the Hungarian algorithm.
    #    The lap.lapjv function returns an assignment for each trajectory:
    #      - For each trajectory i, x[i] is the index of the radar point assigned to it,
    #        or -1 if no assignment was made (or if the cost exceeded the threshold).
    # -------------------------------------------------------------------------

    matched_indices = []
    costs = []

    for i, det in enumerate(dets):
        category = det.category
        radar_cost_threshold = thresholds.get(category, 4.0)  # 默認閾值為 4.0

        _, x, _ = lap.lapjv(cost_matrix[i:i+1, :], extend_cost=True, cost_limit=radar_cost_threshold)
        
        radar_idx = x[0]
        if radar_idx >= 0 and cost_matrix[i, radar_idx] <= radar_cost_threshold:
            matched_indices.append([i, radar_idx])
            costs.append(cost_matrix[i, radar_idx])

    if len(matched_indices) == 0:
        matched_indices = np.empty((0, 2), dtype=int)
        costs = np.empty((0,))
    else:
        matched_indices = np.array(matched_indices, dtype=int)
        costs = np.array(costs)

    return matched_indices, costs