import numpy as np
import torch
from logging import warning
from typing import Tuple, Union
from torch import Tensor
import json
from tqdm import tqdm
import os.path as osp
import os
import pickle
import argparse
import cv2
import gc

def points_img2cam(
        points: Union[Tensor, np.ndarray],
        cam2img: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Project points in image coordinates to camera coordinates.

    Args:
        points (Tensor or np.ndarray): 2.5D points in 2D images with shape
            [N, 3], 3 corresponds with x, y in the image and depth.
        cam2img (Tensor or np.ndarray): Camera intrinsic matrix. The shape can
            be [3, 3], [3, 4] or [4, 4].

    Returns:
        Tensor or np.ndarray: Points in 3D space with shape [N, 3], 3
        corresponds with x, y, z in 3D space.
    """
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys =points[:, :2]
    depths = torch.tensor(points[:, 2]).view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]

    return points3D



def points_cam2img(points_3d: Union[Tensor, np.ndarray],
                   proj_mat: Union[Tensor, np.ndarray],
                   with_depth: bool = False) -> Union[Tensor, np.ndarray]:
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d (Tensor or np.ndarray): Points in shape (N, 3).
        proj_mat (Tensor or np.ndarray): Transformation matrix between
            coordinates.
        with_depth (bool): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        Tensor or np.ndarray: Points in image coordinates with shape [N, 2] if
        ``with_depth=False``, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, \
        'The dimension of the projection matrix should be 2 ' \
        f'instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or \
        (d1 == 4 and d2 == 4), 'The shape of the projection matrix ' \
        f'({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(4,
                                      device=proj_mat.device,
                                      dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

    return point_2d_res

p3d_mapping = json.load(open("/mnt/hwfile/OpenRobotLab/lvruiyuan/embodiedscan_infos/mp3d_mapping.json"))
trscan_mapping = json.load(open("/mnt/hwfile/OpenRobotLab/lvruiyuan/embodiedscan_infos/3rscan_mapping.json"))


embodiedscan_info = pickle.load(open("/mnt/petrelfs/yangshuai1/OpenRobotLab/lvruiyuan/embodiedscan_infos/embodiedscan_infos_test.pkl",'rb'))["data_list"]
embodiedscan_info_dict = {}
for i in embodiedscan_info:
    scene_name = i['images'][0]['img_path'].split("/")[-3]
    if "1mp3d_" in scene_name:
        scene_name = "_".join(scene_name.split("_")[:2])
    to_dict = {}
    for ii  in  i['images']:
        to_dict[ii['img_path'].split("/")[-1].replace(".jpg","")] = ii
    try:
        to_dict['axis_align_matrix'] = i['axis_align_matrix']
        to_dict['cam2img'] = i['cam2img']
    except:
        pass
    try:
        to_dict['depth_cam2img'] = i['depth_cam2img']
    except:
        pass
    if scene_name not in embodiedscan_info_dict:
        embodiedscan_info_dict[scene_name] = to_dict
    else:
        embodiedscan_info_dict[scene_name].update(to_dict)


class AggregateMultiViewPoints():
    """Aggregate points from each frame together.

    The transform steps are as follows:

        1. Collect points from each frame.
        2. Transform points from ego coordinate to global coordinate.
        3. Concatenate transformed points together.

    Args:
        coord_type (str): The type of output point coordinates.
            Defaults to 'DEPTH', corresponding to the global coordinate system
            in EmbodiedScan.
        save_slices (bool): Whether to save point index slices to convert all
            the points into the input for continuous 3D perception,
            corresponding to 1-N frames. Defaults to False.
    """

    def __init__(self,
                 coord_type: str = 'DEPTH',
                 save_slices: bool = False) -> None:
        super().__init__()
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        self.coord_type = coord_type
        self.save_slices = save_slices

    def transform(self, results: dict) -> dict:
        points = [results['points']]
        global_points = []
        points_slice_indices = [0]
        for idx in range(len(points)):
            point = points[idx][..., :3]
            point = torch.cat([point, point.new_ones(point.shape[0], 1)],
                              dim=1)
            global2ego = torch.from_numpy(
                results['depth2img']['extrinsic'][idx]).to(point.device)
            global_point = (torch.linalg.solve(global2ego, point.transpose(
                0, 1).to(torch.float64))).transpose(0, 1)
            points[idx][:, :3] = global_point[:, :3]
            global_points.append(points[idx])
            if self.save_slices:
                points_slice_indices.append(points_slice_indices[-1] +
                                            len(points[idx]))
        td_points = torch.cat(global_points)
        results["3d_points"] = td_points
        return results


point_projector = AggregateMultiViewPoints()

class ConvertRGBDToPoints():
    """Convert depth map to point clouds.

    Args:
        coord_type (str): The type of point coordinates. Defaults to 'CAMERA'.
        use_color (bool): Whether to use color as additional features
            when converting the image to points. Generally speaking, if False,
            only return xyz points. Otherwise, return xyzrgb points.
            Defaults to False.
    """
    def __init__(self,
                 coord_type: str = 'CAMERA',
                 ) -> None:
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        self.coord_type = coord_type

    def transform(self, input_dict: dict) -> dict:
        """Call function to normalize color of points.

        Args:
            input_dict (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
            Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        depth_img = input_dict['depth_img']
        depth_cam2img = input_dict['depth_cam2img']
        ws = np.arange(depth_img.shape[1])
        hs = np.arange(depth_img.shape[0])
        us, vs = np.meshgrid(ws, hs)
        grid = np.stack(
            [us.astype(np.float32),
             vs.astype(np.float32), depth_img], axis=-1).reshape(-1, 3)
        nonzero_indices = depth_img.reshape(-1).nonzero()[0]
        grid3d = points_img2cam(torch.tensor(grid), torch.tensor(depth_cam2img))
        points = grid3d[nonzero_indices]

        # fatch feature
        img = input_dict['img']
        h, w = img.shape[0], img.shape[1]
        cam2img = input_dict['cam2img']
        points2d = np.round(points_cam2img(points,
                                            cam2img))
        us = np.clip(points2d[:, 0], a_min=0, a_max=w - 1)
        vs = np.clip(points2d[:, 1], a_min=0, a_max=h - 1)

        # Target dimensions
        new_h, new_w = 512, 512
        scale_w = new_w / w
        scale_h = new_h / h

        new_us = np.clip((us * scale_w), a_min=0, a_max=new_w - 1).numpy().astype(int)
        new_vs = np.clip((vs * scale_h), a_min=0, a_max=new_h - 1).numpy().astype(int)
        feature = input_dict["blip_feat"][new_vs, new_us]
        
        input_dict['points'] = points
        input_dict['feature'] = feature

        return input_dict
    

converter = ConvertRGBDToPoints()
import random

def parse_info(room_names, feat_dir, depth_dir):
    for room_name in room_names:
        feature_dir = osp.join(feat_dir, room_name)
        if "mp3d" in feature_dir:
            final_path = feature_dir.replace("mp3d_blip","mp3d_final")
        else:
            final_path = feature_dir.replace("3rscan_blip","3rscan_final").replace(room_name,trscan_mapping[room_name])
        try:
            if (
                osp.exists(osp.join(final_path, "pcd_pos.pt"))
            ):
                return
        except:
            pass

        all_dict = []
        max_points =150000//len(os.listdir(feature_dir))
        bar = tqdm(os.listdir(feature_dir))
        for file in bar:
            try:
                if "mp3d" in feature_dir:
                    color_dir = osp.join(depth_dir, room_name,"matterport_color_images")
                    l_depth_dir = os.path.join(depth_dir, room_name,"matterport_depth_images")

                    dep_path = os.path.join(l_depth_dir, file.replace(".pt.npz", ".png"))[::-1].replace("i", "d", 1)[::-1]
                    color_map = cv2.imread(os.path.join(color_dir, file.replace(".pt.npz", ".jpg")))
                    depth_map = np.asarray(cv2.imread(dep_path, cv2.IMREAD_UNCHANGED))/4000.0
                    
                else:
                    # from IPython import embed;embed()
                    color_dir = osp.join(depth_dir, room_name,"sequence")
                    l_depth_dir = osp.join(depth_dir, room_name,"sequence")
                    depth_map = np.asarray(cv2.imread(os.path.join(color_dir, file.replace("color.pt.npz", "depth.pgm")), cv2.IMREAD_UNCHANGED))/1000.0
                    color_map = cv2.imread(os.path.join(color_dir, file.replace(".pt.npz", ".jpg")))
                
                feature_map = np.load(os.path.join(feature_dir, file))['feat']
            except:
                continue
            picture_id = os.path.join(feature_dir, file).split("/")[-1]
            if "mp3d" in feature_dir:
                # room_image_mapping = json.load(open(os.path.join("/mnt/petrelfs/yangshuai1/yangshuai1/3dLLM/mp3d_image_mapping",f"{room_name}.json",)))
                #p3d_mapping[room_name]
                image_info = embodiedscan_info_dict[room_name][picture_id.replace(".pt.npz","")]
                pose_file = image_info['cam2global']
                pose_file = np.linalg.inv(
                    embodiedscan_info_dict[room_name]['axis_align_matrix'] @ pose_file)

                cam2img =  image_info['cam2img']
                
            else:
                image_info = embodiedscan_info_dict[room_name][picture_id.split(".")[0]+".color"]#.replace("frame-","")
                pose_file = image_info['cam2global']
                pose_file = np.linalg.inv(
                    embodiedscan_info_dict[room_name]['axis_align_matrix'] @ pose_file)
                cam2img =  embodiedscan_info_dict[room_name]['cam2img'] # embodiedscan_info_dict[trscan_mapping[room_name]]['cam2img']

            try:
                depth_cam2img = embodiedscan_info_dict[room_name]['depth_cam2img']
            except:
                depth_cam2img = cam2img

            final_result = point_projector.transform(converter.transform({
                'depth2img':{
                    'extrinsic':[pose_file]
                },
                'cam2img':cam2img,
                'depth_img':depth_map,
                'img':color_map,
                'feature':feature_map,
                'depth_cam2img':depth_cam2img,
                'blip_feat':feature_map
            }))
            choose = random.sample(range(final_result["3d_points"].shape[0]),k=min(final_result["3d_points"].shape[0],max_points))
            all_dict.append({
                "pcd":final_result["3d_points"][choose],
                "feat": final_result["feature"][choose]
            })
            gc.collect()

        all_pcd = np.vstack([np.array(i["pcd"]) for i in all_dict])
        all_feat = np.vstack([np.array(i["feat"]) for i in all_dict])
        os.makedirs(final_path,exist_ok=True)
        torch.save(all_feat,osp.join(final_path,"pcd_feat.pt"))
        torch.save(all_pcd,osp.join(final_path,"pcd_pos.pt"))

import threading

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify dirs")
    parser.add_argument("--data_dir_path", default="./masked_rdp_data/", type=str)
    parser.add_argument("--depth_dir_path", default="./masked_rdp_data/", type=str)
    parser.add_argument("--feat_dir_path", default="./maskformer_masks/", type=str)
    parser.add_argument("--sample_num", default=100000, type=int)
    parser.add_argument("--num_threads", default=1, type=int)
    args = parser.parse_args()

    room_list = os.listdir(args.data_dir_path)
    with open(os.path.join(f'embodiedScan/test_scene_ids.txt' ), 'r') as f:
        scan_names = f.read().splitlines()

    print(args.num_threads)

    test_room = []
    if "matter" in args.data_dir_path:

        t_scan_names = []
        for s in scan_names:
            if "1mp3d_" in s:
                t_scan_names.append("_".join(s.split("_")[:2]))
        scan_names = t_scan_names

        for s in room_list:
            if p3d_mapping[s.split("/")[-1]] in scan_names:
                test_room.append(s)
    else:
        t_scan_names = []
        for s in scan_names:
            if "3rscan" in s:
                t_scan_names.append(s)
        scan_names = t_scan_names
        for s in room_list:
            if trscan_mapping[s] in scan_names:
                test_room.append(s)
    
    room_list = test_room
    room_list.reverse()
    print(room_list)

    num_threads = args.num_threads

    segment_size = len(room_list) // num_threads
    threads = []
    for i in range(num_threads):
        start = i * segment_size
        end = start + segment_size if i < num_threads - 1 else len(room_list)
        thread = threading.Thread(target=parse_info, args=(room_list[start:end],args.feat_dir_path, args.depth_dir_path))
        threads.append(thread)
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
