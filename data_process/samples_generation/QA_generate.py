import os
import random
import json
import math
from tqdm import tqdm
import numpy as np
import mmengine
from copy import deepcopy


from utils.file_read_check import numberToWords,read_annotation_pickle,refine_text,NpEncoder
from utils.qa_utils import generate_exist_quantity_QA,generate_single_attribute_QA,generate_diff_relation_QA,generate_space_relation_reference,generate_function_relation_QA,generate_region_QA
from utils.data_path import *
from utils.object_attr_template import *


Target_root =f"{QA_root}/QA_jsons"

DEBUG_mode = False

if not DEBUG_mode:
    with open(SR3D_file_path, 'r', encoding='UTF-8') as f:
        sr3d_dict = json.load(f)
        
class QA_STATIC():
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
def filter_excluded_type(annotation):
    '''
        Here we filter out the excluded objects from the annotation.
    '''
    fix_anno = {}
    fix_anno['object_ids'] = []
    fix_anno['object_types'] = []
    fix_anno['bboxes'] = []
    for _index in range(len(annotation['object_ids'])):
        if annotation['object_types'][_index] not in EXCLUDED_OBJECTS:
            fix_anno['object_ids'].append(annotation['object_ids'][_index])
            fix_anno['object_types'].append(annotation['object_types'][_index])
            fix_anno['bboxes'].append(annotation['bboxes'][_index])
    return fix_anno
def get_object_attribute_dict(scene_id,annotation_data=None,update = True):
    '''
        return a dict describing the attributes of an object in the {scene_id}.
        such as:
            {
                object_id1,object_type1:
                {
                    attribute:{}
                    common_attribute:{}
                }
            }
    '''
    object_attribute_dict = dict()
    
    scene_dir = os.path.join(OBJ_DATA_ROOT, scene_id)
    object_text_annos_dir = os.path.join(scene_dir, "corpora_object", "user_shujutang_czc")
    common_attribute_annos_dir = os.path.join(GPT_EXTRACT_PATH, "common_extract")
    if not os.path.exists(object_text_annos_dir):
        return {}
    object_text_anno_paths = os.listdir(object_text_annos_dir)
    object_text_anno_paths = [os.path.join(object_text_annos_dir, p) for p in object_text_anno_paths if
                              p.endswith(".json")]
    
    
    
    for json_path in object_text_anno_paths:
        base_name = os.path.basename(json_path)
        object_id = int(base_name.split("_")[0])
        object_type = base_name.split("_")[1]
        if object_type in EXCLUDED_OBJECTS:
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        attribute_dict = json_data.get("unique_description", {})

        
        new_attribute_dict = dict()
        for attribute in attribute_dict:
            if len(attribute_dict[attribute]) > 0 and attribute!='weight':
                new_attribute_dict[attribute.lower().replace("_"," ")] = attribute_dict[attribute]
                
        if len(new_attribute_dict) == 0:
            continue
        

        object_attribute_dict[(object_id, object_type)] = dict()
        object_attribute_dict[(object_id, object_type)]["attributes"] = new_attribute_dict
        
        if os.path.exists(os.path.join(common_attribute_annos_dir,f"{scene_id}__{object_id}.json")):
            
            with open(os.path.join(common_attribute_annos_dir,f"{scene_id}__{object_id}.json"), "r") as f:
                common_attribute_dict = json.load(f)
            object_attribute_dict[(object_id, object_type)]["common_attributes"] = common_attribute_dict
    return object_attribute_dict

def filter_repeat_utterance(sr3d_list):
    '''
        for repeat space relation, only choose one of them randomly.
    '''
    from copy import deepcopy
    store_flag = None
    store_list = []
    output_list = []
    for l in sr3d_list:

        if store_flag is not None:
            if store_flag['target_id'] == l['target_id'] and store_flag['distractor_ids'] == l['distractor_ids'] and \
                    store_flag['anchor_ids'] == l['anchor_ids']:
                store_list.append(l)
            else:
                output_list.append(random.sample(store_list, 1)[0])
                store_flag = deepcopy(l)
                store_list.append(l)
        else:
            store_flag = deepcopy(l)
            store_list.append(l)

    return output_list


def filter_excluded_type(annotation):
    '''
        Here we filter out the excluded objects from the annotation.
    '''
    fix_anno = {}
    fix_anno['object_ids'] = []
    fix_anno['object_types'] = []
    fix_anno['bboxes'] = []
    for _index in range(len(annotation['object_ids'])):
        if annotation['object_types'][_index] not in EXCLUDED_OBJECTS:
            fix_anno['object_ids'].append(annotation['object_ids'][_index])
            fix_anno['object_types'].append(annotation['object_types'][_index])
            fix_anno['bboxes'].append(annotation['bboxes'][_index])
    return fix_anno



def generate_sample(question, answers, object_list, scan_id,input_bboxes_id=None,input_bboxes_=None,output_bboxes_id=None, output_bboxes_=None, sub_class='',gen_class='',need_refine = False):
    qa_dict = {
        "sub_class": sub_class,
        "gen_class": gen_class,
        "scan_id": scan_id,
        "question": question,
        "answers": answers,
        "object_ids": [t[0] for t in object_list],
        "object_names": [t[1] for t in object_list],
        "input_bboxes_id": input_bboxes_id,
        "input_bboxes": input_bboxes_,
        "output_bboxes_id": output_bboxes_id,
        "output_bboxes": output_bboxes_,
        "need_refine": need_refine
    }

    return qa_dict

def generate_common_dict(object_attribute_dict):
    object_common_dict ={}
    for attribute in ["color","material","placement","shape"]:
        object_common_dict[attribute] ={}
    for object_id,object_type in object_attribute_dict.keys():
        if "common_attributes" in object_attribute_dict[(object_id,object_type)].keys():
            for attribute in ["color","material","placement","shape"]:
                assert attribute in object_attribute_dict[(object_id,object_type)]["common_attributes"].keys(),f"{attribute} not in common_attributes"
                content = object_attribute_dict[(object_id,object_type)]["common_attributes"][attribute]
                if isinstance(content,str):
                    if len(content)>0:
                        content = [content]
                    else:
                        content = []
                for class_name in content:
                    
                    if class_name not in object_common_dict[attribute].keys():
                        object_common_dict[attribute][class_name] = []
                    object_common_dict[attribute][class_name].append((object_id,object_type))
    return object_common_dict


def generate_Single_Target_QA(scene_id, annotation_data, object_attribute_dict):
    """
        Single-target QA generation.
        1. Exist-quantity QA generation.
        2. Single-attribute/Space QA generation.
    
    """
    QA_list = []
    QA_list += generate_exist_quantity_QA(scene_id, annotation_data)
    QA_list += generate_single_attribute_QA(scene_id, annotation_data,object_attribute_dict)
    
    return QA_list

def generate_Inter_Target_QA(scene_id, annotation_data, object_attribute_dict,region_anno_dict,scene_sr3d_dict):
    
    """
        Inter-target QA generation.
        1. Comparing-attribute/Space QA generation. (based on GPT conclude)
        2. Space-relation QA generation (based on sr3d_dict)
        3. Region-level: function-relation/roles/Advanced QA generation (based on region_anno_dict)
    
    """
    QA_list = []
    
    QA_list += generate_diff_relation_QA(scene_id, annotation_data)
    
    if scene_id in scene_sr3d_dict.keys():
  
        QA_list += generate_space_relation_reference(scene_sr3d_dict[scene_id],annotation_data,scene_id,object_attribute_dict)

    QA_list += generate_function_relation_QA(scene_id, annotation_data, object_attribute_dict, region_anno_dict)

    QA_list += generate_region_QA(scene_id, annotation_data, region_anno_dict)
    
    return QA_list

def check_QA_sample(QA_sample):

    assert isinstance(QA_sample['sub_class'],str)
    assert isinstance(QA_sample['gen_class'],str)
    assert isinstance(QA_sample['scan_id'],str)
    assert isinstance(QA_sample['question'],str)
    assert isinstance(QA_sample['answers'],list)
    for answer in QA_sample['answers']:
        assert isinstance(answer,str)
    assert isinstance(QA_sample['object_ids'],list)
    for object_id in QA_sample['object_ids']:
        assert isinstance(object_id,int),f"{object_id} is {type(object_id)}, {QA_sample['gen_class']}"
    assert isinstance(QA_sample['object_names'],list)
    for object_name in QA_sample['object_names']:
        assert isinstance(object_name,str)
    if QA_sample['input_bboxes_id'] is not None:
        assert isinstance(QA_sample['input_bboxes_id'],list)
        for id_ in QA_sample['input_bboxes_id']:
            assert isinstance(id_,int),f"{id_} is {type(object_id)}, {QA_sample['gen_class']}"
        assert isinstance(QA_sample['input_bboxes'],list)
        for bbox in QA_sample['input_bboxes']:
            assert isinstance(bbox,np.ndarray)

def generate_one_scene(ex_scene_id):
   # try:
    annotation_data = read_annotation_pickle(f'{SPLIT_ROOT}/{ex_scene_id}.pkl', show_progress=False)
    filter_annotation_data = {}
    filter_annotation_data[ex_scene_id] = filter_excluded_type(annotation_data[ex_scene_id])
    object_ids = filter_annotation_data[ex_scene_id]['object_ids']
    object_types = filter_annotation_data[ex_scene_id]['object_types']
    # anno data loading and extracting
    object_attribute_dict = get_object_attribute_dict(ex_scene_id,annotation_data=filter_annotation_data,update = True)
    
    # region anno
    scene_dir = os.path.join(REG_DATA_ROOT, ex_scene_id)
    region_text_annos_dir = os.path.join(scene_dir, "region_views")
    region_anno_dict = dict()
    if os.path.exists(region_text_annos_dir):
        for region_id in os.listdir(region_text_annos_dir):
            if region_id[-4:] == '.png':
                continue
            if ex_scene_id + '__' + region_id in wrong_:
                continue
            if os.path.exists(os.path.join(region_text_annos_dir, region_id, 'struction_shujutang_czc.npy')):
                region_anno_dict[region_id] = {
                    "annotation": np.load(
                        os.path.join(region_text_annos_dir, region_id, 'struction_shujutang_czc.npy'),
                        allow_pickle=True),
                    "objects_filter": np.load(os.path.join(region_text_annos_dir, region_id, 'object_filter.npy'))}
                with open(os.path.join(region_text_annos_dir, region_id, 'object_total.json')) as f:
                    region_anno_dict[region_id]["objects_total"] = json.load(f)
                for special_ in region_anno_dict[region_id]["annotation"][1].keys():
                    object_type,object_id = special_[1:-1].split('_')
                    object_id = int(object_id)
                    raw_text = region_anno_dict[region_id]["annotation"][1][special_].lower().replace(f"the {special_}","it").replace(f"{special_}","it")
                    if object_type in EXCLUDED_OBJECTS:
                        continue
                    if (object_id, object_type) not in object_attribute_dict.keys():
                    
                        object_attribute_dict[(object_id, object_type)] = {"attributes":{"region_role":(raw_text,region_id)}}
                    else:
                        object_attribute_dict[(object_id, object_type)]["attributes"]["region_role"] = (raw_text,region_id)
    object_common_dict = generate_common_dict(object_attribute_dict)

    
    ST_QA = generate_Single_Target_QA(ex_scene_id, filter_annotation_data, object_attribute_dict)
    IT_QA = generate_Inter_Target_QA(ex_scene_id, filter_annotation_data, object_attribute_dict,region_anno_dict,sr3d_dict)
    
    
    for QA_sample in ST_QA+IT_QA:
        QA_sample['object_ids'] = [int(id_) for id_ in QA_sample['object_ids']]
        
        if QA_sample['input_bboxes_id'] is not None:
            QA_sample['input_bboxes_id'] = [int(id_) for id_ in QA_sample['input_bboxes_id']]
        check_QA_sample(QA_sample)
        
        QA_sample["question"] = refine_text(QA_sample["question"])
        QA_sample['answers'] = [refine_text(answer) for answer in QA_sample['answers']]
    os.makedirs(Target_root,exist_ok=True)
    with open(Target_root+f"/{ex_scene_id}.json","w") as f:
        json.dump(ST_QA+IT_QA,f,indent=4,cls=NpEncoder)
  
    
if __name__== "__main__":
    if not DEBUG_mode:
        tasks = [scene_pkl.split('.pkl')[0] for scene_pkl in os.listdir(SPLIT_ROOT)]
        mmengine.track_parallel_progress(generate_one_scene, tasks, nproc=30)
        sub_class_cnt = QA_STATIC()
        gen_class_cnt = QA_STATIC()
        cnt = 0
        for json_file in tqdm(os.listdir(Target_root)):
            with open(os.path.join(Target_root,json_file),'r') as f:
                QA_list = json.load(f)
            for QA_sample in QA_list:
                sub_class_cnt.update(QA_sample['sub_class'])
                gen_class_cnt.update(QA_sample['gen_class'])
            cnt+=len(QA_list)
        print(sub_class_cnt.get_result())
        print(gen_class_cnt.get_result())
        print(cnt)
    
    else:
        total_cnt = 0
        with open(SR3D_file_path, 'r', encoding='UTF-8') as f:
            sr3d_dict = json.load(f)
    
        common_dict = {}
        sample_scene_list = [scene_pkl.split('.json')[0] for scene_pkl in os.listdir(SPLIT_ROOT)]
        # base data loading
        for ex_scene_id in tqdm(sample_scene_list):
        
            annotation_data = read_annotation_pickle(f'{SPLIT_ROOT}/{ex_scene_id}.pkl', show_progress=False)
            filter_annotation_data = {}
            filter_annotation_data[ex_scene_id] = filter_excluded_type(annotation_data[ex_scene_id])
            object_ids = filter_annotation_data[ex_scene_id]['object_ids']
            object_types = filter_annotation_data[ex_scene_id]['object_types']
            # anno data loading and extracting
            object_attribute_dict = get_object_attribute_dict(ex_scene_id,annotation_data=filter_annotation_data,update = True)
            
            # region anno
            scene_dir = os.path.join(REG_DATA_ROOT, ex_scene_id)
            region_text_annos_dir = os.path.join(scene_dir, "region_views")
            region_anno_dict = dict()
            if os.path.exists(region_text_annos_dir):
                for region_id in os.listdir(region_text_annos_dir):
                    if region_id[-4:] == '.png':
                        continue
                    if ex_scene_id + '__' + region_id in wrong_:
                        continue
                    if os.path.exists(os.path.join(region_text_annos_dir, region_id, 'struction_shujutang_czc.npy')):
                        region_anno_dict[region_id] = {
                            "annotation": np.load(
                                os.path.join(region_text_annos_dir, region_id, 'struction_shujutang_czc.npy'),
                                allow_pickle=True),
                            "objects_filter": np.load(os.path.join(region_text_annos_dir, region_id, 'object_filter.npy'))}
                        with open(os.path.join(region_text_annos_dir, region_id, 'object_total.json')) as f:
                            region_anno_dict[region_id]["objects_total"] = json.load(f)
                        for special_ in region_anno_dict[region_id]["annotation"][1].keys():
                            object_type,object_id = special_[1:-1].split('_')
                            object_id = int(object_id)
                            raw_text = region_anno_dict[region_id]["annotation"][1][special_].lower().replace(f"the {special_}","it").replace(f"{special_}","it")
                            if object_type in EXCLUDED_OBJECTS:
                                continue
                            if (object_id, object_type) not in object_attribute_dict.keys():
                            
                                object_attribute_dict[(object_id, object_type)] = {"attributes":{"region_role":(raw_text,region_id)}}
                            else:
                                object_attribute_dict[(object_id, object_type)]["attributes"]["region_role"] = (raw_text,region_id)
            object_common_dict = generate_common_dict(object_attribute_dict)

            
            ST_QA = generate_Single_Target_QA(ex_scene_id, filter_annotation_data, object_attribute_dict)
            IT_QA = generate_Inter_Target_QA(ex_scene_id, filter_annotation_data, object_attribute_dict,region_anno_dict,sr3d_dict)
            
            
            for QA_sample in ST_QA+IT_QA:
                QA_sample['object_ids'] = [int(id_) for id_ in QA_sample['object_ids']]
                
                if QA_sample['input_bboxes_id'] is not None:
                    QA_sample['input_bboxes_id'] = [int(id_) for id_ in QA_sample['input_bboxes_id']]
                check_QA_sample(QA_sample)
                
                QA_sample["question"] = refine_text(QA_sample["question"])
                QA_sample['answers'] = [refine_text(answer) for answer in QA_sample['answers']]
          
            total_cnt +=len(ST_QA+IT_QA)
        
        print(total_cnt)
   