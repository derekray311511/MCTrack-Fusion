import json, yaml
import copy
import argparse
import os
from tqdm import tqdm
from dataset.baseversion_dataset import BaseVersionTrackingDataset

def main():
    # ------------------------------------------------------------------------
    # Load radar data
    # ------------------------------------------------------------------------
    radar_data_path = "/data/radar_PC/radar_PC_13Hz_with_vcomp.json"
    save_path = "/data/radar_PC/radar_PC_baseVersion.json"
    with open(radar_data_path, "r") as f:
        print(f"Loading radar data from {radar_data_path}...")
        radar_data = json.load(f)["radar_PCs"]
        print("Radar data loaded successfully.")

    def load_key_radar_pc(radar_PCs):
        count = 0
        key_radar_PCs = {}
        temp_radar_PCs_token = []
        for k, radar_pc in radar_PCs.items():
            if not radar_pc['is_key_frame']:
                temp_radar_PCs_token.append(k)
                continue
            sample_token = radar_pc['sample_token']
            key_radar_pc = {
                'token': sample_token,
                'radar_token': k,
                'prev_radar_tokens': temp_radar_PCs_token,
                'ego_pose_token': radar_pc['ego_pose_token'],
                'points': radar_pc['points'],
            }
            key_radar_PCs.update({sample_token: key_radar_pc})
            temp_radar_PCs_token = []
            count += 1
        print(f"{count} key radar_PCs loaded")
        return key_radar_PCs

    def get_radar_pcs(radar_PCs, key_radar_PCs, key_token, max_stack=7):
        radar_pcs = []
        stack = 1
        for token in reversed(key_radar_PCs[key_token]['prev_radar_tokens']):
            if stack >= max_stack:
                break
            radar_pcs.append(radar_PCs[token]['points'])
            stack += 1
        radar_pcs.append(key_radar_PCs[key_token]['points'])
        return radar_pcs

    # ------------------------------------------------------------------------
    # Convert radar data to BaseVersionTrackingDataset
    # ------------------------------------------------------------------------
    DETECTIONS_ROOT = "/data/nuscenes/detections/base_version/"
    DETECTOR = "centerpoint_author"
    SPLIT = "val"
    detections_root = os.path.join(
        DETECTIONS_ROOT, DETECTOR, SPLIT + ".json"
    )
    with open(detections_root, "r", encoding="utf-8") as file:
        print(f"Loading data from {detections_root}...")
        data = json.load(file)
        # print(data.keys())
        print("Data loaded successfully.")
    
    cfg_path = "./config/nuscenes.yaml"
    cfg = yaml.load(open(cfg_path, "r"), Loader=yaml.Loader)

    key_radar_PCs = load_key_radar_pc(radar_data)

    scene_lists = [scene_id for scene_id in data.keys()]
    scenes_radar_data = {}
    # print(f"scene_lists: {scene_lists}")
    for scene_id in tqdm(scene_lists, desc="Running scenes"):
        scene_radar = []
        scene_data = data[scene_id]
        dataset = BaseVersionTrackingDataset(scene_id, scene_data, cfg=cfg)
        for index in tqdm(range(len(dataset)), desc=f"Processing {scene_id}"):
            frame_info = dataset[index]
            frame_id = frame_info.frame_id
            cur_sample_token = frame_info.cur_sample_token
            radar_pcs = get_radar_pcs(radar_data, key_radar_PCs, cur_sample_token, max_stack=7)
            scene_radar.append(radar_pcs)
        scenes_radar_data.update({scene_id: scene_radar})
        
    # ------------------------------------------------------------------------
    # Save converted radar data
    # ------------------------------------------------------------------------
    with open(save_path, "w") as f:
        json.dump(scenes_radar_data, f)
        print(f"Radar data saved to {save_path}")
    

if __name__ == "__main__":
    main()