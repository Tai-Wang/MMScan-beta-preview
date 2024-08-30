import numpy as np
from tqdm import tqdm
import json
from utils.file_read_check import read_annotation_pickle,NpEncoder
from utils.data_path import VG_root, QA_root,SPLIT_ROOT
import os

class VG_STATIC():
    def __init__(self):
        self.count_dict = {}
            
    def update(self,class_name):
        if class_name in self.count_dict:
            self.count_dict[class_name] += 1
        else:
            self.count_dict[class_name] = 1
        return self.count_dict[class_name]
    
    def get_result(self):
        sum = np.sum(list(self.count_dict.values()))
        return {class_name:float(self.count_dict[class_name]/sum) for class_name in self.count_dict}
def get_id(json_root):
    
    for json_path in tqdm(os.listdir(os.path.join(json_root))):
        with open(os.path.join(json_root, json_path), 'r') as f:
            data = json.load(f)
        sub_cnt = VG_STATIC()
        for item in data:
            item['ID'] = item['sub_class'] + '__' + item['scan_id'] +'__'+ str(sub_cnt.update(item['sub_class']))
        with open(os.path.join(json_root, json_path), 'w') as f:
            json.dump(data, f, indent=4)
def collect_QA(QA_root):
    large_dict = {}
    for json_path in tqdm(os.listdir(os.path.join(QA_root, 'QA_jsons'))):
        with open(os.path.join(QA_root, 'QA_jsons', json_path), 'r') as f:
            data = json.load(f)
        for item in data:
            large_dict[item["ID"]] = item
    print(len(large_dict))
    with open(os.path.join(QA_root, 'QA_full_dict.json'), 'w') as f:
        json.dump(large_dict, f)
        
def collect_VG(VG_root):
    large_dict = {}
    for json_path in tqdm(os.listdir(os.path.join(VG_root, 'VG_jsons'))):
        with open(os.path.join(VG_root, 'VG_jsons', json_path), 'r') as f:
            data = json.load(f)
        for item in data:
            large_dict[item["ID"]] = item
    print(len(large_dict))
    with open(os.path.join(VG_root, 'VG_full_dict.json'), 'w') as f:
        json.dump(large_dict, f)
        
def split_VG(VG_root):
    with open(VG_root+'/processVG_full_dict.json', 'r') as f:
        large_dict = json.load(f)
    scene_dict = {}
    for item in tqdm(large_dict.values()):
        if item["scan_id"] not in scene_dict:
            scene_dict[item["scan_id"]] = []
        scene_dict[item["scan_id"]].append(item)
    os.makedirs(VG_root+'/process_VG_jsons', exist_ok=True)
    for scene_id in tqdm(scene_dict):
        with open(VG_root+'/process_VG_jsons/'+scene_id+'.json', 'w') as f:
            json.dump(scene_dict[scene_id], f)
def split_QA(QA_root):
    with open(QA_root+'/processQA_full_dict.json', 'r') as f:
        large_dict = json.load(f)
    scene_dict = {}
    for item in tqdm(large_dict.values()):
        if item["scan_id"] not in scene_dict:
            scene_dict[item["scan_id"]] = []
        scene_dict[item["scan_id"]].append(item)
    os.makedirs(QA_root+'/process_QA_jsons', exist_ok=True)
    for scene_id in tqdm(scene_dict):
        with open(QA_root+'/process_QA_jsons/'+scene_id+'.json', 'w') as f:
            json.dump(scene_dict[scene_id], f)


if __name__ == '__main__':
    collect_QA(QA_root)