'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import h5py
import json
import pickle
import numpy as np
import multiprocessing as mp
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
# MULTIVIEW_DATA = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")
UNK_CLASS_ID = 165 # for "object" in es. The actural value is 166, but es counts from 1.

import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt')

def tokenize_text(description):
    tokens = word_tokenize(description)
    return tokens    

from .utils_read import read_es_infos, RAW2NUM_3RSCAN, NUM2RAW_3RSCAN, apply_mapping_to_keys, to_scene_id
from .euler_utils import euler_to_matrix_np
        
with open("/path/to/3rscan_mapping.json", "r") as f:
    mapping_3rscan = json.load(f)
mapping_3rscan = {v:k for k, v in mapping_3rscan.items()}

class ScannetReferenceDataset(Dataset):
       
    def __init__(self, es_info_file, vg_raw_data_file, 
        split="train", 
        num_points=40000,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False, 
        augment=False):
        self.es_info_file = es_info_file
        self.vg_raw_data_file = vg_raw_data_file
        self.es_info = read_es_infos(es_info_file)
        self.es_info = apply_mapping_to_keys(self.es_info, NUM2RAW_3RSCAN)
        scanrefer = self._vg_raw_to_scanrefer(vg_raw_data_file)
        self.scanrefer = scanrefer
        self.scanrefer_all_scene = sorted(list(self.es_info.keys()))
        self.split = split
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.debug = False

        # load data
        self._load_data()
        self.multiview_data = {}


    def __len__(self):
        return len(self.scanrefer)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanrefer[idx]["scene_id"]

        object_id = self.scanrefer[idx]["object_id"]
        
        # get language features
        lang_feat = self.lang[idx]
        lang_len = len(self.scanrefer[idx]["token"])
        lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN else CONF.TRAIN.MAX_DES_LEN

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = point_cloud[:,3:6]-MEAN_COLOR_RGB/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]
        
        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((MAX_NUM_OBJ, 9))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        ref_box_label = np.zeros(MAX_NUM_OBJ) # bbox label for reference target
        ref_class_label = np.zeros(DC.num_size_cluster)
        
        target_bboxes_rot_mat = None    # If target bboxes rot mat is not None, then the target bbox euler angle is not trusted

        if self.split != "test":
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox, :] = instance_bboxes[:MAX_NUM_OBJ,0:9]

            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)
            

            # ------------------------------- DATA AUGMENTATION ------------------------------
            if self.augment and not self.debug:
                # TODO
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    point_cloud[:,0] = -1 * point_cloud[:,0]
                    target_bboxes[:,0] = -1 * target_bboxes[:,0]
                    target_bboxes[:,6] = -target_bboxes[:,6] + np.pi
                    target_bboxes[:,8] = -target_bboxes[:,8]
                    
                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:,1] = -1 * point_cloud[:,1]
                    target_bboxes[:,1] = -1 * target_bboxes[:,1]
                    target_bboxes[:,6] = -target_bboxes[:,6]
                    target_bboxes[:,7] = -target_bboxes[:,1] + np.pi

                # Rotation along up-axis/Z-axis
                target_bboxes_rot_mat = euler_to_matrix_np(target_bboxes[:, 6:9])
                rot_angle = np.random.uniform(-0.087266, 0.087266)
                # rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotz(rot_angle / np.pi * 180)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                # target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")
                target_bboxes[:, 0:3] = np.dot(target_bboxes[:, 0:3], np.transpose(rot_mat))
                target_bboxes_rot_mat = np.matmul(target_bboxes_rot_mat, np.transpose(rot_mat))
                
                # Scale
                scale_factor = np.random.uniform(0.9, 1.1)
                target_bboxes[:, 0:6] *= scale_factor
                point_cloud[:,0:3] *= scale_factor

                # Translation
                trans_factor = np.random.normal(scale=np.array([.1, .1, .1], dtype=np.float32), size=3).T
                point_cloud[:, 0:3] += trans_factor
                target_bboxes[:, 0:3] += trans_factor

            # compute votes *AFTER* augmentation
            # generate votes
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered 
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label. 
            for i_instance in np.unique(instance_labels):            
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                if len(ind) < 10: 
                    continue
                # find the semantic label            
                # if semantic_labels[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind,:3]
                center = 0.5*(x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
            point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
            
            
            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]

            # construct the reference target label for each bbox
            if target_bboxes_rot_mat is None:
                target_bboxes_rot_mat = euler_to_matrix_np(target_bboxes[:, 6:9])
            for i, gt_id in enumerate(instance_bboxes[:num_bbox,-1]):
                if gt_id in object_id:
                    ref_box_label[i] = 1
                    ref_class_label[DC.nyu40id2class[int(instance_bboxes[i,-2])]] = 1
        else:
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9]) # make 3 votes identical 
            point_votes_mask = np.zeros(self.num_points)

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
        except KeyError:
            pass
        
        if target_bboxes_rot_mat is None:
            target_bboxes_rot_mat = euler_to_matrix_np(target_bboxes[:, 6:9])
            

        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict["lang_feat"] = lang_feat.astype(np.float32) # language feature vectors
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64) # length of each description
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (MAX_NUM_OBJ,)
        data_dict["target_bbox"] = target_bboxes.astype(np.float32)
        data_dict["target_rot_mat"] = target_bboxes_rot_mat.astype(np.float32)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (MAX_NUM_OBJ, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (MAX_NUM_OBJ,) semantic class index
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        data_dict["ref_box_label"] = ref_box_label.astype(bool) # 0/1 reference labels for each object bbox
        data_dict['ref_class_label'] = ref_class_label.astype(np.float32)
        data_dict["pcl_color"] = pcl_color
        data_dict["sub_class"] = self.scanrefer[idx]["sub_class"]
        data_dict["load_time"] = time.time() - start

        return data_dict

    def _vg_raw_to_scanrefer(self, vg_raw_data_file):
        ret = []
        data = json.load(open(vg_raw_data_file))
        tmp_save = {}
        # tmp_save[scene_id][object_id] records num annos
        for i, d in enumerate(data):
            # if i > 100:
            #     break
            out_dict = {}
            scene_id = to_scene_id(d["scan_id"])
            out_dict["scene_id"] = NUM2RAW_3RSCAN.get(scene_id, scene_id) # a hack here. The pcd files are saved differently. 3rscan are in raw, scannet and mp3d are in num.
            tmp_save[d["scan_id"]] = tmp_save.get(d["scan_id"], {})
            if len(d['target_id']) <= 0:
                continue
            target_id = d['target_id']
            out_dict["object_id"] = target_id
            out_dict["ann_id"] = i
            out_dict["description"] = d["text"].lower()
            out_dict["token"] = tokenize_text(d["text"].lower())
            out_dict["sub_class"] = d.get("sub_class", "other")
            ret.append(out_dict)
        del tmp_save
        return ret

    def _get_raw2label(self, count_type_from_zero=True):
        """
            Designed to map raw object names to nyu40 ids (int)
            es_mod: map object names to label ids (int)
        """
        raw2label = np.load(self.es_info_file, allow_pickle=True)["metainfo"]["categories"] # str2int
        raw2label = {k: v-count_type_from_zero for k, v in raw2label.items()}
        return raw2label


    def _tranform_des(self):
        with open(GLOVE_PICKLE, "rb") as f:
            glove = pickle.load(f)

        lang = []
        for data in self.scanrefer:

            # tokenize the description
            tokens = data["token"]
            embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN, 300))
            for token_id in range(CONF.TRAIN.MAX_DES_LEN):
                if token_id < len(tokens):
                    token = tokens[token_id]
                    if token in glove:
                        embeddings[token_id] = glove[token]
                    else:
                        embeddings[token_id] = glove["unk"]

            # store
            lang.append(embeddings)

        return lang


    def _load_data(self):
        print("loading data...")
        # load language features
        self.lang = self._tranform_des()
        self.raw2label = self._get_raw2label()

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))
        if self.debug:
            self.scene_list = self.scene_list[:10]
            self.scanrefer = [sr for sr in self.scanrefer if sr["scene_id"] in self.scene_list]
        # load scene data
        self.scene_data = {}
        for scene_id in tqdm(self.scene_list):
            if scene_id not in self.es_info:
                print(f"drop due to {scene_id} not found in es info")
                self.scene_list.remove(scene_id)
                continue
            es_info = self.es_info[scene_id]
            if len(es_info["bboxes"]) == 0:
                continue
            file_path = os.path.join("/path/to/pcd_data/pcd_with_global_alignment", scene_id + ".pth")
            if not os.path.exists(file_path):
                self.scene_list.remove(scene_id)
                print(f"drop {scene_id} due to unfound pcd file ")
                continue
            self.scene_data[scene_id] = {}
            data = torch.load(file_path)
            pcds, colors, semantic_labels, instance_labels = data
            self.scene_data[scene_id]["mesh_vertices"] = np.hstack([pcds, colors])
            assert self.scene_data[scene_id]["mesh_vertices"].shape[1] == 6, self.scene_data[scene_id]["mesh_vertices"].shape
            self.scene_data[scene_id]["instance_labels"] = instance_labels
            self.scene_data[scene_id]["semantic_labels"] = semantic_labels
            boxes = np.zeros((MAX_NUM_OBJ, 8+3))  # x y z w h l yaw roll pitch class id
            for i, (bbox, obj_type) in enumerate(zip(es_info["bboxes"], es_info["object_types"])):
                if i >= MAX_NUM_OBJ:
                    break
                boxes[i, :6] = bbox[:6]
                boxes[i, 6:9] = bbox[6:]
                boxes[i, 9] = self.raw2label[obj_type]
                boxes[i, 10] = i
                
            self.scene_data[scene_id]["instance_bboxes"] = boxes

        print(f"successfully loaded {len(self.scene_data.items())} for {self.split} set")

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox
