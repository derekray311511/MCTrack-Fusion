# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse

from convert_kitti import kitti_main
from convert_nuscenes import nuscenes_main
# from convert_waymo import waymo_main

kitti_cfg = {
    "raw_data_path": "data/kitti/datasets",
    "dets_path": "data/kitti/detectors/",
    "save_path": "data/base_version/kitti/",
    "detector": "virconv",  # virconv / casa / ... /
    "split": "test",  # val / test
}

nuscenes_cfg = {
    "raw_data_path": "/data/nuscenes/",
    "dets_path": "/data/nuscenes/detections/",
    "save_path": "/data/nuscenes/detections/base_version/",
    "detector": "centerpointPainted",  #  centerpoint(val) / largekernel(test) / ....
    "split": "val",  # val / test
}

waymo_cfg = {
    "raw_data_path": "data/waymo/datasets/",
    "dets_path": "data/waymo/detectors/",
    "save_path": "data/base_version/waymo/",
    "detector": "ctrl",
    "split": "val",  # val / test
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="kitti", help="kitti/nuscenes/waymo"
    )
    args = parser.parse_args()

    if args.dataset == "kitti":
        kitti_main(
            kitti_cfg["raw_data_path"],
            kitti_cfg["dets_path"],
            kitti_cfg["detector"],
            kitti_cfg["save_path"],
            kitti_cfg["split"],
        )
    elif args.dataset == "nuscenes":
        nuscenes_main(
            nuscenes_cfg["raw_data_path"],
            nuscenes_cfg["dets_path"],
            nuscenes_cfg["detector"],
            nuscenes_cfg["save_path"],
            nuscenes_cfg["split"],
        )
    elif args.dataset == "waymo":
        waymo_main(
            waymo_cfg["raw_data_path"],
            waymo_cfg["dets_path"],
            waymo_cfg["detector"],
            waymo_cfg["save_path"],
            waymo_cfg["split"],
        )
