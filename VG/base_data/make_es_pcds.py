import os
import numpy as np
import torch
import mmengine
import json
from og3d_src.utils.utils_read import read_es_infos
from utils.utils_3d import is_inside_box, euler_angles_to_matrix
from utils.decorators import mmengine_track_func

path_of_embodiedScan_info = '' # change it to the path of  embodiedScan_info
path_of_point_clouds = '' # change it to the path of point clouds of scenes
dataroot_mp3d = f"{path_of_point_clouds}/matterport3d/scans"
dataroot_3rscan = f"{path_of_point_clouds}//3rscan/scans"
dataroot_scannet = f"{path_of_point_clouds}//scannet/scans"
output_dir = "" '' # change it to the output path


TYPE2INT = np.load(es_info_file, allow_pickle=True)["metainfo"]["categories"] # str2int

def load_pcd_data(scene):
    pcd_file = os.path.join(DATAROOT, scene, "pc_infos.npy")
    pc_infos = np.load(pcd_file)
    nan_mask = np.isnan(pc_infos).any(axis=1)
    pc_infos = pc_infos[~nan_mask]
    pc = pc_infos[:, :3]
    color = pc_infos[:, 3:6]
    label = pc_infos[:, 6].astype(np.uint16) # this do not matter in the current code
    # semantic_ids = pc_infos[:, 7].astype(np.uint16)
    return pc, color, label

def create_scene_pcd(scene, es_anno, overwrite=False):
    if es_anno is None:
        return None
    if len(es_anno["bboxes"]) <= 0:
        return None
    out_file_name = os.path.join(output_dir,"pcd_with_global_alignment", f"{scene}.pth")
    if os.path.exists(out_file_name) and not overwrite:
        return True
    pc, color, label = load_pcd_data(scene)
    label = np.ones_like(label) * -100
    if np.isnan(pc).any() or np.isnan(color).any():
        print(f"nan detected in {scene}")
    instance_ids = np.ones(pc.shape[0], dtype=np.int16) * (-100)
    bboxes =  es_anno["bboxes"].reshape(-1, 9)
    bboxes[:, 3:6] = np.clip(bboxes[:, 3:6], a_min=1e-2, a_max=None)
    object_ids = es_anno["object_ids"]
    object_types = es_anno["object_types"] # str
    sorted_indices = sorted(enumerate(bboxes), key=lambda x: -np.prod(x[1][3:6])) # the larger the box, the smaller the index
    sorted_indices_list = [index for index, value in sorted_indices]

    bboxes = [bboxes[index] for index in sorted_indices_list]
    object_ids = [object_ids[index] for index in sorted_indices_list]
    object_types = [object_types[index] for index in sorted_indices_list]

    for box, obj_id, obj_type in zip(bboxes, object_ids, object_types):
        obj_type_id = TYPE2INT.get(obj_type, -1)
        center, size, euler = box[:3], box[3:6], box[6:]
        R = euler_angles_to_matrix(euler, convention="ZXY")
        R = R.reshape((3,3))
        box_pc_mask = is_inside_box(pc, center, size, R)
        num_points_in_box = np.sum(box_pc_mask)
        # if num_points_in_box == 0:
        #     print(f"do not contain points: {obj_type}, {obj_id}")
        instance_ids[box_pc_mask] = obj_id
        label[box_pc_mask] = obj_type_id

    out_data = (pc, color, label, instance_ids)
    os.makedirs(os.path.dirname(out_file_name), exist_ok=True)
    torch.save(out_data, out_file_name)
    return True



if __name__ == "__main__":
    MODE = "3rscan"
    assert MODE in ["mp3d", "3rscan", "scannet"]
    DATAROOT = eval(f"dataroot_{MODE}")
    scene_list = os.listdir(DATAROOT)
    # scene_list = scene_list[:50]
    # scene_list = ["scene0000_00"]
    embodiedscan_annotation_files = [
        f"{path_of_embodiedScan_info}/embodiedscan_infos_train_full.pkl",
        f"{path_of_embodiedScan_info}/embodiedscan_infos_val_full.pkl"
    ]
    train_split_file = "./mmscan_anno/train_scene_ids.text"
    val_split_file = "./mmscan_anno/val_scene_ids.text"
    anno_train = read_es_infos(embodiedscan_annotation_files[0])
    anno_val = read_es_infos(embodiedscan_annotation_files[1])
    with open(f"{path_of_embodiedScan_info}/3rscan_mapping.json", 'r') as f:
        scene_mappings = json.load(f)
    ####################################################################
    # save splits
    mini_scenes = set(os.listdir(dataroot_mp3d)[:50] + os.listdir(dataroot_3rscan)[:50] + os.listdir(dataroot_scannet)[:50])
    reverse_mapping = {v:k for k,v in scene_mappings.items()}
    with open(train_split_file, 'w') as f:
        for key in anno_train:
            key = reverse_mapping.get(key, key)
            if key in mini_scenes:
                f.write(key + '\n')
    with open(val_split_file, 'w') as f:
        for key in anno_val:
            key = reverse_mapping.get(key, key)
            if key in mini_scenes:
                f.write(key + '\n')
    ####################################################################
    es_annos = {**anno_train, **anno_val}
    tasks = []
    for scene in scene_list:
        # only 3rscan needs mapping. mp3d do not.
        es_anno = es_annos.get(scene_mappings.get(scene, scene), None)
        if es_anno:
            tasks.append((scene, es_anno))
    mmengine.track_parallel_progress(create_scene_pcd, tasks, nproc=8)
