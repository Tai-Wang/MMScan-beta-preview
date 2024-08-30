import os, json, random
import torch
import numpy as np
import random
from copy import deepcopy
from typing import Dict, List
from datasets.scannet_base_dataset import BASE, DatasetConfig, ScanNetBaseDataset
from transformers import AutoTokenizer
from eval_utils.evaluate_embodied import evaluate
from datasets.task_prompts import TASK_PROPMT, BOX_FORMAT
import pickle
from glob import glob
import os.path as osp
from datasets.mmscan_config import *

class Dataset(ScanNetBaseDataset):
    
    def __init__(
        self,
        args,
        dataset_config,
        split_set="train",
        num_points=40000,
        use_color=False,
        use_normal=False,
        use_multiview=False,
        use_height=False,
        augment=False,
    ):
        super().__init__(
            args,
            dataset_config,
            split_set=split_set,
            num_points=num_points,
            use_color=use_color,
            use_normal=use_normal,
            use_multiview=use_multiview,
            use_height=use_height,
            augment=augment,
            use_random_cuboid=False,
            random_cuboid_min_points=None,
        )
        self.task_name = 'embodied_scan'
        self.grid_size_3d = args.grid_size_3d
        self.max_prompts = args.max_prompts
        self.split = split_set
        self.dataset_config = dataset_config
        self.max_des_len = args.max_des_len
        self.eval_func = evaluate
        
        ## initialize tokenizer and set tokenizer's `padding token` to `eos token`
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab, add_bos_token=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.qtokenizer = AutoTokenizer.from_pretrained(args.qformer_vocab)
        self.qtokenizer.pad_token = self.tokenizer.eos_token
        self.qtokenizer.padding_side = 'right'
        
        ## load annotations
        assert split_set in ["train", "val"]
        if split_set !="train":
            self.embodied_scan = pickle.load(open(osp.join(embodied_scan_info_base,"embodiedscan_infos_val_full.pkl"),'rb'))['data_list']
            with open(os.path.join(embodied_scan_data_base,'test_scene_ids.txt' ), 'r') as f:
                self.scan_names = f.read().splitlines()
        else:
            self.embodied_scan = pickle.load(open(osp.join(embodied_scan_info_base,"embodiedscan_infos_train_full.pkl"),'rb'))['data_list']
            with open(os.path.join(embodied_scan_data_base,'train_scene_ids.txt' ), 'r') as f:
                self.scan_names = f.read().splitlines()
        collect_dict = {}
        for i in self.embodied_scan:
            bbox_dict = {}
            for ii in i[ 'instances']:
                bbox_dict.update({
                    ii['bbox_id']:ii
                })
            collect_dict.update({
                i['images'][0]['img_path'].split("/")[-2]:bbox_dict # map to readable ID
            })
        
        self.embodied_scan = collect_dict # scan_id ==> bbox_id ==> bbox info
        
        
        anno_files = [
            osp.join(embodied_scan_data_base,'object_caption.json'),
            osp.join(embodied_scan_data_base,'region_caption.json'),
        ]

        scan_caps = []
        for anno_file in anno_files:
            # anno_file = os.path.join(anno_dir, f'attribute_QA.json') # TODO: use split `self.split`
            with open(anno_file, 'r') as f:
                json_data = json.load(f)
            for item in json_data:
                if item['scan_id'] not in self.scan_names:
                    continue
                if "object_id_list" in item:
                    obj_id = item['object_id_list'] # is a list
                    cap = item["region_caption"]
                    bboxs = item["object_bbox_list"]
                elif "object_id" in item:
                    obj_id = [item['object_id']]
                    cap = item["object_caption"]
                    bboxs = [item["object_bbox"]]
                else:
                    print(" ============ error ! ==============")
                # add all answers

                scan_caps.append({
                    'scan_id':item['scan_id'],
                    'obj_id': obj_id, # many have multi bbox
                    'bbox':bboxs,
                    'caption': cap,
                    'question':"describe the region"if "object_id_list" in item else "describe the object",
                    'ID':item['ID']
                })

        self.annotations = scan_caps
        # if self.split != 'train':
        #     self.annotations = [{'scene_id': scene_id} for scene_id in self.scan_names]
        self._tag_dataset(self.annotations, 'embodied_cap')
        
        ## super configuration
        self.tokenizer_config = dict(
            max_length=self.max_des_len, 
            padding='max_length', 
            truncation='longest_first', 
            return_tensors='np'
        )
        print(f"kept {len(self.annotations)} annotations in {len(self.scan_names)} scans...")

    
    def _tag_dataset(self, corpus, task_name): 
        for anno in corpus:
            anno['task_name'] = task_name
        return 
    
    def _encode_box_coords(self, ret_dict,l):
        center_normalized = ret_dict['embodied_scan_box_centers_normalized'][-l:]
        size_normalized = ret_dict['embodied_scan_box_sizes_normalized'][-l:]
        box_normalized = np.hstack((center_normalized, size_normalized))    # (-1, 6)
        # <cx, cy, cz, w, h, l>
        box_normalized = (box_normalized * self.grid_size_3d).astype(np.int64)
        return ' '.join(BOX_FORMAT.format(*box) for box in box_normalized)
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        
        scan_name = self.annotations[idx]['scan_id']
        task_name = self.annotations[idx]['task_name']
        question = self.annotations[idx]['question'].lower()
        embodied_scan_bbox = []
        if self.annotations[idx]["bbox"] is not None:
            for bbox in self.annotations[idx]["bbox"]:
                embodied_scan_bbox.append(torch.tensor(bbox))

        if embodied_scan_bbox:
            try:
                # TODO: only input first bbox
                embodied_scan_bbox= torch.vstack(embodied_scan_bbox)[0,:6]
            except:
                print(embodied_scan_bbox)
                embodied_scan_bbox = None
        else:
            embodied_scan_bbox = None
        ret_dict = self._get_scan_data(scan_name,
                                       embodied_scan_bbox = embodied_scan_bbox)
        
        if self.split == 'train':
            prompt = deepcopy(random.choice(TASK_PROPMT[task_name]))
        else:
            prompt = deepcopy(TASK_PROPMT[task_name][0])
        
        boxes = None
        if self.annotations[idx]['bbox'] is not None:
            boxes = self._encode_box_coords(ret_dict,len(self.annotations[idx]['bbox']))
        prompt['instruction'] = prompt['instruction'].format(locations=boxes, question=question)
        # print(scan_name,str(prompt).replace("</s>",""))


        prompt_inputs = self.tokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        qformer_inputs = self.qtokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)

        if self.split == 'train':
            caption = self.annotations[idx]['caption']
        else:
            caption = ""

        if self.split == 'train':
            response = prompt['answer'].format(answer=caption)
        else:
            response = ""
            
        ## input_ids as labels for LLM
        llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((prompt['instruction'], response, self.tokenizer.eos_token))],
            **self.tokenizer_config
        )
        
        box_query = np.zeros((self.max_prompts, 8, 3))
        box_mask = np.zeros((self.max_prompts,))
        click_query = np.zeros((self.max_prompts, 3))
        click_mask = np.zeros((self.max_prompts,))
        
        # use first object to refer:
        if self.annotations[idx]['bbox'] is not None:
            if random.random() > 0.5:
                # use box to identify an object
                ref_gt_box_corner = \
                    ret_dict["embodied_scan_box_corners"][0].reshape(8, 3).astype(np.float32)
                box_query[0] = ref_gt_box_corner
                box_mask[0] = 1
            else:
                # use click to identify an object(for embodied scan, some empty bboxes do not contain any points, thus we simply use center point)
                click_query[0] = ret_dict["embodied_scan_box_centers"][0].reshape(3,).astype(np.float32)
                click_mask[0] = 1
        else:
            box_query = np.zeros((self.max_prompts, 8, 3))
            box_mask = np.zeros((self.max_prompts,))
            click_query = np.zeros((self.max_prompts, 3))
            click_mask = np.zeros((self.max_prompts,))
        
        ret_dict['box_query'] = box_query.astype(np.float32)
        ret_dict['box_mask'] = box_mask.astype(np.float32)
        ret_dict['click_query'] = click_query.astype(np.float32)
        ret_dict['click_mask'] = click_mask.astype(np.float32)
        
        ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
        # print(scan_name,self.tokenizer.decode(ret_dict['input_ids']).replace("</s>",""))
        ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
        ret_dict['gradient_mask'] = \
            (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
        
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
        ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
        ret_dict['qformer_input_ids'] = qformer_inputs['input_ids'][0].astype(np.int64)
        ret_dict['qformer_attention_mask'] = qformer_inputs['attention_mask'][0].astype(np.float32)

        keys_to_remove = [k for k in ret_dict.keys() if "embodied" in str(k)]
        for k in keys_to_remove:
            ret_dict.pop(k)
        return ret_dict
   

