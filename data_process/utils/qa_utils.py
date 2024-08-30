import os
import random
import json
import math
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from .object_attr_template import EXCLUDED_OBJECTS,PLURAL_DICT,SPACE_TYPES,REFERING_TYPES,object_template,EXCLUDED_OBJECTS,wrong_,function_base_category
from .file_read_check import numberToWords,read_annotation_pickle
object_have_state = ['cabinet', 'plate', 'door', 'light', 'window', 'tv', 'socket', 'stove', 'lamp', 'backpack', 'case', 'shelf', 'vase', 'wardrobe', 'purse', 'bin', 'refrigerator', 'book', 'container', 'switch', 'box', 'bucket', 'sack', 'bag', 'range hood', 'toilet', 'piano', 'printer', 'laptop', 'bathtub', 'monitor', 'basket', 'flowerpot', 'microwave', 'computer', 'air conditioner', 'package', 'washing machine', 'clothes dryer', 'umbrella', 'notebook', 'tablet', 'jar', 'faucet', 'oven', 'bowl', 'dishwasher', 'screen', 'drawer', 'crate', 'dish rack', 'magazine', 'treadmill', 'copier', 'toaster', 'file', 'pot', 'pack', 'hair dryer', 'radio', 'hamper', 'garage door', 'jalousie', 'bidet', 'projector', 'mailbox', 'humidifier', 'letter', 'can', 'player', 'pool']
EXTRACT_PATH = '/mnt/petrelfs/linjingli/mmscan_db/mmscan_data/gpt_extract'
# making more diversity
DEFAULT_NAME = ['items','entities', 'things','elements']
DEFAULT_COMMAND = ['find','select','choose','locate']
COMMON_USE_DICT = ["material","color","placement","shape"]
BEGIN_STATE = ["Look carefully at the room, ", "Take a close look at the room, ","Inspect every corner of the room, ","Survey the room meticulously, ","Examine the room thoroughly, "]

size_ratio = 0.35
anchor_OR_sample = 5
OBJ_DATA_ROOT = "/mnt/petrelfs/linjingli/mmscan_db/mmscan_data/modified_objects_7_29"
REG_DATA_ROOT = "/mnt/petrelfs/linjingli/mmscan_db/mmscan_data/modified_regions_8_9"
GPT_EXTRACT_PATH = "/mnt/petrelfs/linjingli/mmscan_db/mmscan_data/gpt_extract"
# all VG classes
class_name =  All_VG_SUB_CLASS = ['VG_Single_Attribute_Unique', 'VG_Single_Attribute_Common', 'VG_Single_EQ',
              'VG_Inter_OR', 'VG_Inter_Space_OO', 'VG_Inter_Attribute_OO', 'VG_Single_Space']
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
    
    # prepare for the size output, this is only use in the QA
    if update:  
        Size_info = np.load(f"/mnt/petrelfs/linjingli/mmscan_db/anno_process/tmp/size_average.npy", allow_pickle=True).item()
        object_ids = annotation_data[scene_id]['object_ids']
        object_types = annotation_data[scene_id]['object_types']
        bboxes = annotation_data[scene_id]['bboxes']
    
    
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
        
        if update and object_type in Size_info and object_type!='object':
            average_size = Size_info[object_type]
            object_bbox = bboxes[object_ids.index(object_id)]
            object_size = object_bbox[3]*object_bbox[4]*object_bbox[5]
           
            if size_ratio<(object_size/average_size)<1.0/size_ratio:
                new_attribute_dict["relative_size"] = "Standard in the same type."
            else:
                if object_size<average_size:
                    new_attribute_dict["relative_size"] = "Small in the same type."
                else:
                    new_attribute_dict["relative_size"] = "Large in the same type."

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

def generate_exist_quantity_QA(scene_id, annotation_data):
    
    '''
        ask for existence and quantity of objects
    '''
    from .object_attr_template import PLURAL_DICT
    object_ids = annotation_data[scene_id]['object_ids']
    object_types = annotation_data[scene_id]['object_types']
    QA_list = []

    all_types = list(PLURAL_DICT.keys())
    object_exist_dict = {}
    for index_ in range(len(object_ids)):
        if object_types[index_] == 'object':
            continue
        if object_types[index_] in object_exist_dict.keys():
            object_exist_dict[object_types[index_]].append(object_ids[index_])
        else:
            object_exist_dict[object_types[index_]] = [object_ids[index_]]
    object_type_ids_list = []
    for key_ in object_exist_dict.keys():
        object_type_ids_list.append((key_, object_exist_dict[key_]))
   
    # ask for existence and quantity
    pos_types = list(object_exist_dict.keys())
    neg_types = [type_ for type_ in all_types if type_ not in pos_types and type_ != 'object']
    neg_types = random.sample(neg_types, min(len(neg_types),len(pos_types)))
    
    for object_type in pos_types:
        if random.choice([True,False]):
            question = random.choice(BEGIN_STATE)+f'is there a {object_type} in the room? '
            answers = ['Yes, there is. ']
            object_list = [(object_id, object_type) for object_id in object_exist_dict[object_type]]
            scan_id = scene_id
            QA_list.append(generate_sample(question, answers, object_list, scan_id, sub_class='QA_Single_EQ', gen_class='basedata-pos-exist'))
        else:
            question = random.choice(BEGIN_STATE)+f'how many {PLURAL_DICT[object_type]} are there in the room? '
            answers = [f'There are {numberToWords(len(object_exist_dict[object_type]))}.',f'The number is {len(object_exist_dict[object_type])}. ']
            object_list = [(object_id, object_type) for object_id in object_exist_dict[object_type]]
            scan_id = scene_id
            QA_list.append(generate_sample(question, answers, object_list, scan_id, sub_class='QA_Single_EQ', gen_class='basedata-pos-quantity'))
        
    for object_type in neg_types:
        if random.choice([True,False]):
            question = random.choice(BEGIN_STATE)+f'is there a {object_type} in the room? '
            answers = ["No, there isn't. "]
            object_list = []
            scan_id = scene_id
            QA_list.append(generate_sample(question, answers, object_list, scan_id, sub_class='QA_Single_EQ', gen_class='basedata-neg-exist'))
        else:
            question = random.choice(BEGIN_STATE)+f'how many {PLURAL_DICT[object_type]} are there in the room? '
            answers = [f'There are zero.',f'The number is 0. ']
            object_list = []
            scan_id = scene_id
            QA_list.append(generate_sample(question, answers, object_list, scan_id, sub_class='QA_Single_EQ', gen_class='basedata-neg-quantity'))
    
    return QA_list
           


def generate_single_attribute_QA(scene_id, annotation_data,object_attribute_dict):
    '''
        ask for single attribute of objects
    '''
    QA_list = []
    object_ids = annotation_data[scene_id]['object_ids']
    object_types = annotation_data[scene_id]['object_types']
    object_bboxes = annotation_data[scene_id]['bboxes']

    
    
    # 1. with box
    for object_id, object_type in object_attribute_dict.keys():

        object_bbox = object_bboxes[list(object_ids).index(object_id)]
        object_dict = deepcopy(object_attribute_dict[(object_id, object_type)]["attributes"])
        if ("type" not in object_dict.keys() or object_dict["type"] == "object") and "fine grained category" in object_dict.keys():
            object_dict["type"] = object_dict["fine grained category"]
        for attribute_name in object_dict.keys():
            if attribute_name not in ["color","material","shape","placement","type","function"]:
                continue
            
            question = random.choice(["Examine the object closely, ","Take a close look at the object, ","Inspect the object carefully, ","Have a closer look at the object, ","Study the object closely, "])+f" what is the {attribute_name} of it? "
            answer = [f"Its {attribute_name} is {object_dict[attribute_name]}. "]
            QA_list.append(generate_sample(question, answer, [(object_id, object_type)], scene_id, input_bboxes_id=[object_id], input_bboxes_=[object_bbox], sub_class='QA_Single_Attribute' if attribute_name in ["color","material","type","function"] else 'QA_Single_Space', gen_class=f'unique-normal-attribute-{attribute_name}',need_refine=True))
            
        if object_type in object_have_state and "state" in object_dict.keys():
            question = random.choice(["Examine the object closely, ","Take a close look at the object, ","Inspect the object carefully, ","Have a closer look at the object, ","Study the object closely, "])+f" what is the state of it? "
            answer = [f"Its state is {object_dict['state']}. "]
            QA_list.append(generate_sample(question, answer, [(object_id, object_type)], scene_id, input_bboxes_id=[object_id], input_bboxes_=[object_bbox], sub_class='QA_Single_Attribute', gen_class=f'unique-specail-attribute-state',need_refine=True))
            
        if "relative_size" in object_dict.keys():
            question = random.choice(["Examine the object closely, ","Take a close look at the object, ","Inspect the object carefully, ","Have a closer look at the object, ","Study the object closely, "])+f" what is the size of it compared to the same type? "
            answer = [object_dict["relative_size"]] 
            QA_list.append(generate_sample(question, answer, [(object_id, object_type)], scene_id, input_bboxes_id=[object_id], input_bboxes_=[object_bbox], sub_class='QA_Single_Space', gen_class=f'unique-specail-attribute-size',need_refine=False))
            
        if "region_role" in object_dict.keys():
            question = random.choice(["Examine the object closely, ","Take a close look at the object, ","Inspect the object carefully, ","Have a closer look at the object, ","Study the object closely, "])+f" what is the special role of it in the {object_dict['region_role'][1].split('_')[1]} of the room? "
            answer = [object_dict['region_role'][0]]
            QA_list.append(generate_sample(question, answer, [(object_id, object_type)], scene_id, input_bboxes_id=[object_id], input_bboxes_=[object_bbox], sub_class='QA_Inter_OR', gen_class=f'unique-specail-attribute-region_role',need_refine=True))
    
    # 2.without box
    anno_type_dict = {}
    for object_id, object_type in object_attribute_dict.keys():
        if object_type not in anno_type_dict.keys():
            anno_type_dict[object_type] = []
        anno_type_dict[object_type].append(object_id)
        
    for object_type in anno_type_dict.keys():
        
        if len(anno_type_dict[object_type]) > 1 or object_type=='object':
            continue
        object_id = anno_type_dict[object_type][0]
        object_bbox = object_bboxes[list(object_ids).index(object_id)]
        object_dict = deepcopy(object_attribute_dict[(object_id, object_type)]["attributes"])
        if ("type" not in object_dict.keys() or object_dict["type"] == "object") and "fine grained category" in object_dict.keys():
            object_dict["type"] = object_dict["fine grained category"]
        for attribute_name in object_dict.keys():
            if attribute_name not in ["color","material","shape","function"]:
                continue
            
            question = random.choice(["Examine the object closely, ","Take a close look at the object, ","Inspect the object carefully, ","Have a closer look at the object, ","Study the object closely, "])+f" what is the {attribute_name} of it? "
            question = question.replace("object",object_type)
            answer = [f"Its {attribute_name} is {object_dict[attribute_name]}. "]
            QA_list.append(generate_sample(question, answer, [(object_id, object_type)], scene_id, input_bboxes_id=[object_id], input_bboxes_=[object_bbox], sub_class='QA_Single_Attribute' if attribute_name in ["color","material","type","function"] else 'QA_Single_Space', gen_class=f'unique-nonbox-attribute-{attribute_name}',need_refine=True))
            
        if object_type in object_have_state and "state" in object_dict.keys():
            question = random.choice(["Examine the object closely, ","Take a close look at the object, ","Inspect the object carefully, ","Have a closer look at the object, ","Study the object closely, "])+f" what is the state of it? "
            question = question.replace("object",object_type)
            answer = [f"Its state is {object_dict['state']}. "]
            QA_list.append(generate_sample(question, answer, [(object_id, object_type)], scene_id, input_bboxes_id=[object_id], input_bboxes_=[object_bbox], sub_class='QA_Single_Attribute', gen_class=f'unique-nonbox-attribute-state',need_refine=True))
            
        if "relative_size" in object_dict.keys():
            question = random.choice(["Examine the object closely, ","Take a close look at the object, ","Inspect the object carefully, ","Have a closer look at the object, ","Study the object closely, "])+f" what is the size of it compared to the same type? "
            question = question.replace("object",object_type)
            answer = [object_dict["relative_size"]] 
            QA_list.append(generate_sample(question, answer, [(object_id, object_type)], scene_id, input_bboxes_id=[object_id], input_bboxes_=[object_bbox], sub_class='QA_Single_Space', gen_class=f'unique-nonbox-attribute-size',need_refine=False))
            
        if "region_role" in object_dict.keys():
            question = random.choice(["Examine the object closely, ","Take a close look at the object, ","Inspect the object carefully, ","Have a closer look at the object, ","Study the object closely, "])+f" what is the special role of it in the {object_dict['region_role'][1].split('_')[1]} of the room? "
            question = question.replace("object",object_type)
            answer = [object_dict['region_role'][0]]
            QA_list.append(generate_sample(question, answer, [(object_id, object_type)], scene_id, input_bboxes_id=[object_id], input_bboxes_=[object_bbox], sub_class='QA_Inter_OR', gen_class=f'unique-nonbox-attribute-region_role',need_refine=True))
            
    
    # todo: think of induce if possible. Need to enlarge again.
    return QA_list

def generate_diff_relation_QA(scene_id, annotation_data):
    
    def is_absolute(compare_mode,box1,box2):
        if compare_mode == "bigger":
            x1,y1,z1 = sorted(box1[:3])
            x2,y2,z2 = sorted(box2[:3])
            return x1 > 1.5*x2 and y1 > 1.5*y2 and z1 > 1.5*z2
        elif compare_mode == "higher":
            return box1[2] > box2[2] +0.5* (box2[5]+box1[5])
        return False
    
    QA_list = []
    object_ids = annotation_data[scene_id]['object_ids']
    object_types = annotation_data[scene_id]['object_types']
    object_bboxes = annotation_data[scene_id]['bboxes']
    
    # attribute (color/shape/material/placement)
    differ_pairs = []
    if os.path.exists(os.path.join("/mnt/petrelfs/linjingli/mmscan_db/mmscan_data/gpt_extract/qa_pairs", scene_id)):
        for json_path in os.listdir(os.path.join("/mnt/petrelfs/linjingli/mmscan_db/mmscan_data/gpt_extract/qa_pairs", scene_id)):
            with open(os.path.join("/mnt/petrelfs/linjingli/mmscan_db/mmscan_data/gpt_extract/qa_pairs", scene_id,json_path),"r") as f:
                response_dict = json.load(f)
                object_x_id,object_y_id = json_path[:-5].split("__")
                object_x_id = int(object_x_id)
                object_y_id = int(object_y_id)
                for attribute_name in response_dict.keys():
                    if attribute_name not in ["color", "shape", "material"]:
                        continue
                    if "*" not in response_dict[attribute_name]:
                        continue
                    answer,reason = response_dict[attribute_name].split("*")
                    answer = answer.strip().lower()
                    if answer=='yes':
                        differ_pairs.append({"ids":(object_x_id,object_y_id),"attribute":attribute_name,"pos":True,"reason":reason})
                    if answer=='no':
                        differ_pairs.append({"ids":(object_x_id,object_y_id),"attribute":attribute_name,"pos":False,"reason":reason})
    # need to downsample
    differ_pairs = random.sample(differ_pairs,min(len(differ_pairs),3*10,3*len(object_ids)))                    
    for diff_dict in differ_pairs:
        object_x_id,object_y_id = diff_dict["ids"]
        attribute_name = diff_dict["attribute"]
        pos = diff_dict["pos"]
        reason = diff_dict["reason"]
        if object_x_id not in object_ids or object_y_id not in object_ids:
            continue
        object_x_type = object_types[list(object_ids).index(object_x_id)]
        object_y_type = object_types[list(object_ids).index(object_y_id)]
        object_x_bbox = object_bboxes[list(object_ids).index(object_x_id)]
        object_y_bbox = object_bboxes[list(object_ids).index(object_y_id)]
        question = random.choice(["Carefully compare these two objects, ","Take a careful look at these two objects and compare them, ","Please compare these two objects closely, "])+"are they similar in "+attribute_name+"? Given me the answer and reason. "
        if pos:
            answer = "Yes, "+reason
        else:
            answer = "No, "+reason
        QA_list.append(generate_sample(question, [answer], [(object_x_id,object_x_type),(object_y_id,object_y_type)], scene_id,input_bboxes_id=[object_x_id,object_y_id],input_bboxes_=[object_x_bbox,object_y_bbox], sub_class='QA_Inter_Attribute_OO', gen_class=f'gpt-comparediff-{attribute_name}'))
    
    
    # space (high & size) too easy?
    bigger_list = []
    higher_list = []
    info_list = []
    for _index_1 in range(len(object_ids)):
        for _index_2 in range(_index_1+1,len(object_ids)):
            object_id_1 = object_ids[_index_1]
            object_id_2 = object_ids[_index_2]
            object_type_1 = object_types[_index_1]
            object_type_2 = object_types[_index_2]
            object_bbox_1 = object_bboxes[_index_1]
            object_bbox_2 = object_bboxes[_index_2]
            # not able to refer type.
            if object_type_1 == "object" or object_type_2 == "object" or object_type_1 == object_type_2:
                continue
            if is_absolute("bigger",object_bbox_1,object_bbox_2):
                bigger_list.append((object_id_1,object_id_2,object_type_1,object_type_2))
                info_list.append((object_id_1,object_id_2,object_type_1,object_type_2))
            if is_absolute("higher",object_bbox_1,object_bbox_2):
                higher_list.append((object_id_1,object_id_2,object_type_1,object_type_2))
                info_list.append((object_id_1,object_id_2,object_type_1,object_type_2))
    bigger_list = random.sample(bigger_list,min(len(bigger_list),15))
    higher_list = random.sample(higher_list,min(len(higher_list),15))
    for object_id_1,object_id_2,object_type_1,object_type_2 in bigger_list:
        qusetion_pos = f"Is the {object_type_1} bigger than the {object_type_2}? "
        qusetion_neg = f"Is the {object_type_2} bigger than the {object_type_1}? "
        answer_pos = "Yes. "
        answer_neg = "No. "
        object_bbox_1,object_bbox_2 = object_bboxes[list(object_ids).index(object_id_1)],object_bboxes[list(object_ids).index(object_id_2)]
        QA_list.append(generate_sample(qusetion_pos, [answer_pos], [(object_id_1,object_type_1),(object_id_2,object_type_2)], scene_id,input_bboxes_id=[object_id_1,object_id_2],input_bboxes_=[object_bbox_1,object_bbox_2], sub_class='QA_Inter_Attribute_OO', gen_class='gpt-comparediff-space-bigger'))
        QA_list.append(generate_sample(qusetion_neg, [answer_neg], [(object_id_1,object_type_1),(object_id_2,object_type_2)], scene_id,input_bboxes_id=[object_id_1,object_id_2],input_bboxes_=[object_bbox_1,object_bbox_2], sub_class='QA_Inter_Attribute_OO', gen_class='gpt-comparediff-space-bigger'))
    for object_id_1,object_id_2,object_type_1,object_type_2 in higher_list:
        qusetion_pos = f"Is the {object_type_1} higher than the {object_type_2}? "
        qusetion_neg = f"Is the {object_type_2} higher than the {object_type_1}? "
        answer_pos = "Yes. "
        answer_neg = "No. "
        object_bbox_1,object_bbox_2 = object_bboxes[list(object_ids).index(object_id_1)],object_bboxes[list(object_ids).index(object_id_2)]
        QA_list.append(generate_sample(qusetion_pos, [answer_pos], [(object_id_1,object_type_1),(object_id_2,object_type_2)], scene_id,input_bboxes_id=[object_id_1,object_id_2],input_bboxes_=[object_bbox_1,object_bbox_2], sub_class='QA_Inter_Space_OO', gen_class='gpt-comparediff-space-higger'))
        QA_list.append(generate_sample(qusetion_neg, [answer_neg], [(object_id_1,object_type_1),(object_id_2,object_type_2)], scene_id,input_bboxes_id=[object_id_1,object_id_2],input_bboxes_=[object_bbox_1,object_bbox_2], sub_class='QA_Inter_Space_OO', gen_class='gpt-comparediff-space-higger'))
   
    return QA_list

def generate_space_relation_reference(scene_sr3d_dict,annotation_data,scene_id,object_attribute_dict):
    def process_text(text):
        
        '''
            process the sr3d text, getting the target_dict and all_ids
        '''

        _index = 0
        output_list = []
        all_ids = []
        target_dict = {}
        while '[' in text[_index:] and ']' in text[_index:]:
            a = text[_index:].index('[')
            b = text[_index:].index(']')
            output_list.append(text[_index + a:_index + b + 1])
            if '[' in text[_index + a + 1:_index + b] or ']' in text[_index + a + 1:_index + b]:
                return '', {}, []
            _index = _index + b + 1

        for object_name in output_list:
            space_index = []
            for _index in range(len(object_name)):
                if object_name[_index] == ' ':
                    space_index.append(_index)
            num = int(object_name[space_index[-1] + 1:-1])
            object_type_ = object_name[1:space_index[-1]]

            text = text.replace(object_name, object_type_)

            target_dict[object_name.split(' ')[1]] = [num]
            all_ids.append(num)
        return text, target_dict, all_ids
    def filter_repeat_utterance(sr3d_list):

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

    # filter repeat utterance
    raw_space_list = filter_repeat_utterance(scene_sr3d_dict)
    
    def get_tamplate(text):
        '''
            get the template from the sr3d text
        '''
        _index = 0
        output_list = []
        while '[' in text[_index:] and ']' in text[_index:]:
            a = text[_index:].index('[')
            b = text[_index:].index(']')
            output_list.append(text[_index + a:_index + b + 1])
            if '[' in text[_index + a + 1:_index + b] or ']' in text[_index + a + 1:_index + b]:
                return ''
            _index = _index + b + 1
        for object_name in output_list:
            text = text.replace(object_name, '<>')
        return text
    
    
    wrong_list = ['<>stick', '<>way', 'frame', 'knob', '<>sill', 'cutter', '<>t', '<>pet', 'dryer', 'paper', '<>ry']
    allow_to_ask = [' on ',' left ',' right ',' behind ',' back ',' above ',' over ',' below ',' under ', ' beneath ', ' underneath '," supporting "]
    not_allow_to_ask = [' close',' far',' near ',' next to ',' beside '," in front of "," between "," center "," middle " ]
    delete_list = ['select', 'choose', 'pick', 'find', 'that']
    ask_phrases = ["where is this {target_type} in relation to the {anchor_type}?","where is this {target_type} relative to the {anchor_type}'s position?","describe this {target_type}'s position in relation to the {anchor_type}. ","where is this {target_type} situated compared to the {anchor_type}? "]
    
    
    QA_list = []

    object_ids = annotation_data[scene_id]['object_ids']

    object_types = annotation_data[scene_id]['object_types']

    object_bboxes = annotation_data[scene_id]['bboxes']
    
    for raw_space_item in raw_space_list:
        #print(raw_space_item["utterance"])
        template = get_tamplate(raw_space_item["utterance"])
        vg_text, target_dict, all_ids = process_text(raw_space_item["utterance"])
        
        # template valid
        skip_flag = False
        for wrong_word in wrong_list:
            if wrong_word in template:
                skip_flag = True     
        if len(vg_text) == 0 or len(template) == 0:
            skip_flag = True  
        
        # objects valid
        for _id in all_ids:
            if _id not in object_ids:
                skip_flag = True
                continue
            _type = object_types[list(object_ids).index(_id)]
            if _type in EXCLUDED_OBJECTS:
                skip_flag = True
        if not any(allow_to_ask_word in template for allow_to_ask_word in allow_to_ask):
            skip_flag = True
                
        if skip_flag:
            continue
        if len(raw_space_item["anchor_ids"])>1:
            continue
        
        # (1) ask for the space relation between two objects. need to find the anchor object first.
        if any(allow_to_ask_word in template for allow_to_ask_word in allow_to_ask):
            target_id = int(raw_space_item["target_id"])
            target_type = object_types[list(object_ids).index(raw_space_item["target_id"])]
            target_bbox = object_bboxes[list(object_ids).index(target_id)]
            anchor_id = int(raw_space_item["anchor_ids"][0])
            anchor_type = [object_types[list(object_ids).index(id_)] for id_ in raw_space_item["anchor_ids"]][0]
            anchor_bbox = [object_bboxes[list(object_ids).index(id_)] for id_ in raw_space_item["anchor_ids"]][0]
            
            word_dict = {"target_type":"object","anchor_type":anchor_type}
            question = random.choice(ask_phrases).format(**word_dict)
            if any(word in template for word in [' left ',' right ',' behind ',' back ']):
                question = "In horizontal direction, "+question
            else:
                question = "In vertical direction, "+question
            answer = vg_text
            if (template.count(',') >= 1):
                
                question = template.split(',')[0].replace('<>',f'the {anchor_type}')  +', '+question
                answer = answer.split(',')[1]
            answer_ = ''
            for word in answer.split(" "):
                if word not in delete_list:
                    answer_ += word + " "
            answer = answer_[:-1].replace(f"the {target_type}","this object").replace(" it",f" the {anchor_type}")+". "
            
            QA_list.append(generate_sample(question, [answer], [(target_id,target_type),(anchor_id,anchor_type)], scene_id,input_bboxes_id=[target_id],input_bboxes_=[target_bbox], sub_class='QA_Inter_Space_OO', gen_class='sr3d-asksptial'))
        
        # (2) ask for the target object's attribute in spatial relation to the anchor object.

        target_id = int(raw_space_item["target_id"])
        target_type = object_types[list(object_ids).index(raw_space_item["target_id"])]
        target_bbox = object_bboxes[list(object_ids).index(target_id)]
        anchor_id = int(raw_space_item["anchor_ids"][0])
        anchor_type = [object_types[list(object_ids).index(id_)] for id_ in raw_space_item["anchor_ids"]][0]
        anchor_bbox = [object_bboxes[list(object_ids).index(id_)] for id_ in raw_space_item["anchor_ids"]][0]
        
        if (target_id,target_type) not in object_attribute_dict.keys():
            continue
       
        query_attribute = ["color","material","shape"]
        if len([k for k in query_attribute if k in object_attribute_dict[(target_id,target_type)]["attributes"].keys()])>0:
            attribute_name = random.choice([k for k in query_attribute if k in object_attribute_dict[(target_id,target_type)]["attributes"].keys()])
        else:
            continue
        
        question = vg_text+f". What is the {attribute_name} of it? "
        answer = object_attribute_dict[(target_id,target_type)]["attributes"][attribute_name]
        
        QA_list.append(generate_sample(question, [answer], [(target_id,target_type),(anchor_id,anchor_type)], scene_id,input_bboxes_id=[target_id],input_bboxes_=[target_bbox], sub_class='QA_Inter_Space_OO', gen_class='sr3d-askattribute',need_refine=True))
    
        
    return QA_list

def generate_function_relation_QA(scene_id, annotation_data, object_attribute_dict, region_anno_dict):
    
    
    QA_list = []
    

    object_ids = annotation_data[scene_id]['object_ids']
    object_types = annotation_data[scene_id]['object_types']

    object_bboxes = annotation_data[scene_id]['bboxes']
    
    for _id in region_anno_dict:
        region_id = _id
        
        

        # (1) ask about the joint function of a group of objects
        large_class_dict = region_anno_dict[_id]['annotation'][2]
        for object_name_list in large_class_dict.keys():
            if isinstance(object_name_list, str):
                object_id_list = [int(object_name[1:-1].split('_')[1]) for object_name in eval(object_name_list) if int(object_name[1:-1].split('_')[1]) in object_ids]
            else:
                object_id_list = [int(object_name[1:-1].split('_')[1]) for object_name in object_name_list if int(object_name[1:-1].split('_')[1]) in object_ids]
                
            # directly
            question = random.choice([f"How do these {numberToWords(len(object_name_list))} {random.choice(DEFAULT_NAME)} collectively contribute to the {region_id.split('_')[1]}?", f"What role do these {numberToWords(len(object_name_list))} {random.choice(DEFAULT_NAME)} share in the {region_id.split('_')[1]}?",f"What is the shared purpose of these {numberToWords(len(object_name_list))} {random.choice(DEFAULT_NAME)} within the context of the {region_id.split('_')[1]}?"])

            answer = large_class_dict[object_name_list]
            if answer == '':
                continue
            QA_list.append(generate_sample(question, [answer], [(object_id,object_types[list(object_ids).index(object_id)]) for object_id in object_id_list], scene_id, input_bboxes_id=object_id_list, input_bboxes_=[object_bboxes[list(object_ids).index(object_id)] for object_id in object_id_list], sub_class='QA_Inter_OR',gen_class='regionbigclass-askdirect',need_refine=True))

            # in-directly
            pos_ids = object_id_list
            neg_ids = [id_ for id_ in object_ids if id_ not in pos_ids]
            neg_ids = random.sample(neg_ids, min(len(neg_ids), len(pos_ids)))
            question = random.choice([f"Some {random.choice(DEFAULT_NAME)} collectively contribute to the {region_id.split('_')[1]}: ", f"Some {random.choice(DEFAULT_NAME)} share roles in the {region_id.split('_')[1]}: "])+large_class_dict[object_name_list]+". Is this object belonging to the group? "
            for pos_id in pos_ids:
                answer = ['Yes. ']
                QA_list.append(generate_sample(question, answer, [(pos_id,object_types[list(object_ids).index(pos_id)])], scene_id, input_bboxes_id=[pos_id], input_bboxes_=[object_bboxes[list(object_ids).index(pos_id)]], sub_class='QA_Inter_OR',gen_class='regionbigclass-pos',need_refine=True))
            for neg_id in neg_ids:
                answer = ['No. ']
                QA_list.append(generate_sample(question, answer, [(neg_id,object_types[list(object_ids).index(neg_id)])], scene_id, input_bboxes_id=[neg_id], input_bboxes_=[object_bboxes[list(object_ids).index(neg_id)]], sub_class='QA_Inter_OR',gen_class='regionbigclass-neg',need_refine=True))
               
        # (2) ask about the function relationship in a region
        function_relations = region_anno_dict[_id]['annotation'][0]
        # ask directly
        for object_x_name,object_y_name in function_relations.keys():
            object_x_type, object_x_id = object_x_name[1:-1].split('_')
            object_y_type, object_y_id = object_y_name[1:-1].split('_')

           
            if np.random.uniform() > 0.5:
                object_x_type, object_x_id = object_y_name[1:-1].split('_')
                object_y_type, object_y_id = object_x_name[1:-1].split('_')

            object_x_id = int(object_x_id)
            object_y_id = int(object_y_id)
            if object_x_id not in object_ids or object_y_id not in object_ids:
                print(f"Error! {object_x_name} or {object_y_name} is not in the {region_id} of {scene_id}. ")
                continue

            # (1) ask about the function relationship of two objects
            question = random.choice([f"What is the functional relationship between these two {random.choice(DEFAULT_NAME)} in the {region_id.split('_')[1]}?",f"Describe the functional connection between these two {random.choice(DEFAULT_NAME)} in the {region_id.split('_')[1]}."])
            answers = function_relations[(object_x_name, object_y_name)].replace(object_x_name, object_x_type).replace(
                    object_y_name, object_y_type)
            object_list = [(object_x_id, object_x_type), (object_y_id, object_y_type)]
            input_bboxes_id = [object_x_id, object_y_id]

            input_bboxes = [object_bboxes[list(object_ids).index(object_x_id)],
                            object_bboxes[list(object_ids).index(object_y_id)]]
            QA_list.append(generate_sample(question, [answers], object_list, scene_id, input_bboxes_id=input_bboxes_id, input_bboxes_=input_bboxes, sub_class='QA_Inter_Attribute_OO',gen_class='region-function_relation_direct',need_refine=True))
        
       
        # ask indirectly
        anchor_function_relations = dict()
        for object_tuple in function_relations.keys():

            if object_tuple[0] not in anchor_function_relations.keys():
                anchor_function_relations[object_tuple[0]] = [(object_tuple[1], function_relations[object_tuple])]
            else:
                anchor_function_relations[object_tuple[0]].append((object_tuple[1], function_relations[object_tuple]))
            
            if object_tuple[1] not in anchor_function_relations.keys():
                anchor_function_relations[object_tuple[1]] = [(object_tuple[0], function_relations[object_tuple])]
            else:
                anchor_function_relations[object_tuple[1]].append((object_tuple[0], function_relations[object_tuple]))



        for anchor_name in anchor_function_relations:
            anchor_type, anchor_id = anchor_name.split('_')[0][1:], int(anchor_name.split('_')[1][:-1])
            if anchor_id not in object_ids:
                print("ERROR ID in FunctionOO anchor")
                continue
            anchor_function_text_dict = {}
            for target_name, text in anchor_function_relations[anchor_name]:
                target_type, target_id = target_name.split('_')[0][1:], int(target_name.split('_')[1][:-1])
                if target_id not in object_ids:
                    print("ERROR ID in FunctionOO target")
                    continue
                text = text.replace(target_name, 'X')
                text = text.replace(anchor_name, anchor_type)
                text = text[:-1] + f' in the {region_id.split("_")[1]}. '

                if text not in anchor_function_text_dict:
                    anchor_function_text_dict[text] = [(target_id, target_type)]
                else:
                    anchor_function_text_dict[text].append((target_id, target_type))
                    
            for text in anchor_function_text_dict.keys():
                if len(anchor_function_text_dict[text])>1:
                    continue
                target_id,target_type = anchor_function_text_dict[text][0]
                if (target_id,target_type) not in object_attribute_dict.keys() or "attributes" not in object_attribute_dict[(target_id,target_type)]:
                    continue
                attribute_dict = deepcopy(object_attribute_dict[(target_id,target_type)]["attributes"])
               
                if ("type" not in attribute_dict or target_type=='object') and "fine grained type" in attribute_dict:
                    attribute_dict["type"] = attribute_dict["fine grained type"]
                if "type" in attribute_dict:
                    question = text + f" What is the object-type of X? "
                    QA_list.append(generate_sample(question, [attribute_dict["type"]], [(target_id,target_type), (anchor_id,anchor_type)], scene_id, input_bboxes_id=[anchor_id], input_bboxes_=[object_bboxes[list(object_ids).index(anchor_id)]], sub_class='QA_Inter_Attribute_OO',gen_class='region-function_relation_indirect',need_refine=True))
                
                query_attribute = ["color","material","shape"]
                if len([k for k in query_attribute if k in object_attribute_dict[(target_id,target_type)]["attributes"].keys()])>0:
                    attribute_name = random.choice([k for k in query_attribute if k in object_attribute_dict[(target_id,target_type)]["attributes"].keys()])
                else:
                    continue
                
                question = text+f". What is the {attribute_name} of X? "
                answer = object_attribute_dict[(target_id,target_type)]["attributes"][attribute_name]
                
                QA_list.append(generate_sample(question, [answer], [(target_id,target_type), (anchor_id,anchor_type)], scene_id, input_bboxes_id=[anchor_id], input_bboxes_=[object_bboxes[list(object_ids).index(anchor_id)]], sub_class='QA_Inter_Attribute_OO',gen_class='region-function_relation_indirect',need_refine=True))

   
            
    return QA_list

def generate_region_QA(scene_id, annotation_data, region_anno_dict):
    def strict_check_qa(text):
        angle_list = []
        angle_dict = {}
        for _index1 in range(len(text)):
            if text[_index1] == '<':
                for _index2 in range(_index1 + 1, len(text)):
                    if text[_index2] == '>':
                        break
                angle_list.append(text[_index1:_index2 + 1])
                if text[_index1:_index2 + 1] not in angle_dict.keys():
                    angle_dict[text[_index1:_index2 + 1]] = 1
                else:
                    angle_dict[text[_index1:_index2 + 1]] += 1
        return angle_list, angle_dict
    QA_list = []

    object_ids = annotation_data[scene_id]['object_ids']
    
    object_types = annotation_data[scene_id]['object_types']

    object_bboxes = annotation_data[scene_id]['bboxes']
    
    for _id in region_anno_dict:
        region_id = _id

        region_QA = region_anno_dict[_id]['annotation'][4]
        
        BEGIN_STATE= f"Here is a question about the {region_id.split('_')[1]}, use [] to represent the important objects: "

        for region_Q in region_QA.keys():
            
            if 'Q' in region_Q:
                region_A = region_Q.replace('Q', 'A')
                question = region_QA[region_Q]
                answer = region_QA[region_A]
                if region_QA[region_A] == '' or region_QA[region_Q] == '':
                    continue
                q_list, _ = strict_check_qa(question)
                a_list, _ = strict_check_qa(answer)
                input_id = []
                object_list = []
               

                try:
                    for item_ in q_list:
                        if '_' in item_:
                            o_id, o_name = int(item_[1:-1].split('_')[1]), item_[1:-1].split('_')[0]
                        else:
                            o_id, o_name = int(item_[1:-1].split(' ')[1]), item_[1:-1].split(' ')[0]
                        question = question.replace(item_, '['+o_name+']')
                        input_id.append(o_id)
                        assert o_id in object_ids
                        object_list.append((o_id, o_name))
                
                        
                    for item_ in a_list:
                        if '_' in item_:
                            o_id, o_name = int(item_[1:-1].split('_')[1]), item_[1:-1].split('_')[0]
                        else:
                            o_id, o_name = int(item_[1:-1].split(' ')[1]), item_[1:-1].split(' ')[0]
                        answer = answer.replace(item_, '['+o_name+']')
                        
                        if (o_id, o_name) not in object_list:
                            assert o_id in object_ids
                            object_list.append((o_id, o_name))
                except:
                    continue
                
                if len(input_id)==0:
                    input_id = [region_object_id for region_object_id in region_anno_dict[region_id]["objects_filter"] if region_object_id in object_ids]

                QA_list.append(generate_sample(BEGIN_STATE+question, [answer], object_list, scene_id, input_bboxes_id=input_id, input_bboxes_=[object_bboxes[list(object_ids).index(o_id)] for o_id in input_id], sub_class='QA_Advanced',gen_class='region-QA',need_refine=False))
 
    
    return QA_list        
