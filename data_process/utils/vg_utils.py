import os
import random
import json
import math
from tqdm import tqdm
import numpy as np
from .object_attr_template import EXCLUDED_OBJECTS,PLURAL_DICT,SPACE_TYPES,REFERING_TYPES,object_template,EXCLUDED_OBJECTS,wrong_,function_base_category
from .file_read_check import numberToWords,read_annotation_pickle
from .data_path import GPT_EXTRACT_PATH
EXTRACT_PATH = '/mnt/petrelfs/linjingli/mmscan_db/mmscan_data/gpt_extract'
# making more diversity
DEFAULT_NAME = ['items','entities', 'things','elements']
DEFAULT_COMMAND = ['find','select','choose','locate']
COMMON_USE_DICT = ["material","color","placement","shape"]
BEGIN_STATE = ["Look carefully at the room, ", "Take a close look at the room, ","Inspect every corner of the room, ","Survey the room meticulously, ","Examine the room thoroughly, "]

size_ratio = 0.35
anchor_OR_sample = 5

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


def find_matches(sentence, target_dict):
    """
    Input: {"text1":[id1,id2,...],"text2":[id1,id2,...]}
    output: {id1:[[start1,end1],[start2,end2],...],id2:[[start1,end1],[start2,end2],...]}
    """
    matches = dict()

    for s in target_dict.keys():
        start = 0
        while True:
            start = sentence.find(s, start)
            if start == -1:
                break
            end = start + len(s)
            for object_id in target_dict[s]:
                _id = int(object_id)
                if _id not in matches.keys():
                    matches[_id] = [[start, end]]
                else:
                    matches[_id].append([start, end])
            start += 1
    return matches

def generate_sample(text, scan_id,target_list, anchor_list=[], distractor_ids=[],tokens_dict={}, 
                             sub_class='',gen_class='',need_refine = False):
    reference_dict = {
        "sub_class": sub_class,
        "gen_class": gen_class,
        "need_refine":need_refine,
        "scan_id": scan_id,
        "target_id": [t[0] for t in target_list],
        "distractor_ids":distractor_ids,
        "text": text,
        "target": [t[1] for t in target_list],
        "anchors": [t[1] for t in anchor_list],
        "anchor_ids": [t[0] for t in anchor_list],
        "tokens_positive": find_matches(text, tokens_dict)
    }
    return reference_dict

def generate_easy_unique(object_unqine_dict,object_id,object_type,scan_id):
    '''
        Grounding the single object type without distractors
    '''
    sample_num_dis = {"Attribute":5,"Space":4,"Region-role":2}
    sample_class = {"Attribute":'VG_Single_Attribute_Unique',"Space":'VG_Single_Space',"Region-role":'VG_Inter_OR'}

    object_unqine_dict = {k.lower().replace(" ", "_").replace("-", "_"): v for k, v in object_unqine_dict['attributes'].items()}
    if "fine_grained_category" not in object_unqine_dict or len(object_unqine_dict["fine_grained_category"])==0:
        object_unqine_dict["fine_grained_category"] = object_type
    
    sentence_list = {}
    reference_list = []
    for theme_ in ["Attribute","Space","Region-role"]:
        sentence_list[theme_] = []

        for sentence,text in object_template[theme_]:
            try:
                sentence.format(**object_unqine_dict)
                text.format(**object_unqine_dict)
                sentence_list[theme_].append([sentence,text])
            except KeyError:
                pass

        templates = random.sample(sentence_list[theme_], min(sample_num_dis[theme_], len(sentence_list[theme_])))

        template_choice = [template[0].format(**object_unqine_dict) for template in templates]
        token_phrase = [template[1].format(**object_unqine_dict) for template in templates]
        

        for _index in range(len(templates)):
            sentence, token = template_choice[_index], token_phrase[_index]
     
            assert token in sentence, "token not in sentence"
        
            reference_dict = generate_sample(text=sentence, scan_id=scan_id,target_list=[(object_id,object_type)], anchor_list=[], distractor_ids=[],tokens_dict={token:[object_id]},sub_class=sample_class[theme_],gen_class=f'onetypetemplate-{theme_}',need_refine = True) 
            reference_list.append(reference_dict)

    return reference_list

def generate_hard_unique(distractor_ids,_raw_text,object_id,object_type , scan_id):
    '''
        Grounding the single object type with distractors
    '''
    reference_list = []
    for text in _raw_text:
        if "*" not in text:
            print(text)
            continue
            
        raw_text = text.split("*")[0]
        token_ = raw_text.lower()
        text = random.choice(DEFAULT_COMMAND) + " " + token_
        reference_list.append(generate_sample(text=text, scan_id=scan_id,target_list=[(object_id,object_type)], anchor_list=[], distractor_ids=[id_ for id_ in distractor_ids if id_!=object_id],tokens_dict={token_:[object_id]},sub_class='VG_Single_Attribute_Unique',gen_class=f'multitype-diff',need_refine = False))
    return reference_list
        

def generate_unique_attribute_reference(object_attribute_dict,scan_id):

    reference_list = []
    if os.path.exists(os.path.join(EXTRACT_PATH, "vg_unique_refer3",scan_id+".json")):
        with open(os.path.join(EXTRACT_PATH, "vg_unique_refer3",scan_id+".json"), "r") as f:
            hard_dict = json.load(f)
    else:
        hard_dict = {}
    object_type_id_dict = {}
    for object_id,object_type in object_attribute_dict.keys():
        if object_type not in object_type_id_dict.keys():
            object_type_id_dict[object_type] = []
        object_type_id_dict[object_type].append(object_id)
    for object_id,object_type in object_attribute_dict.keys():
        if object_type not in hard_dict.keys() or isinstance(hard_dict[object_type],str) or object_type=="object":
            reference_list += generate_easy_unique(object_attribute_dict[(object_id,object_type)],object_id,object_type , scan_id=scan_id)
        else:
            # todo: 扩充比例
            if str(object_id) in hard_dict[object_type]:
                reference_list+= generate_hard_unique(object_type_id_dict[object_type],hard_dict[object_type][str(object_id)],object_id,object_type , scan_id)
    return reference_list

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
                    
def generate_common_attribute_reference(object_common_dict,annotation_data,scan_id):
    
    # grounding a lot of object
    reference_list = []
    
    #1. base on color
    all_colors = ['Red','Orange','Yellow','Green','Blue','Purple','Black','White/Beige','Gray/Silver','Brown','Pink','Gold','Transparent']
    
    for class_name in object_common_dict['color'].keys():
       
        if class_name == 'White/Beige':
            color_name = 'white or beige'
        elif class_name == 'Gray/Silver':
            color_name = 'gray or silver'
        else:
            color_name = class_name.lower()
        name_ = random.choice(DEFAULT_NAME)
        token_ = f"all the {name_} that have {color_name} in color"
        text = random.choice(BEGIN_STATE) + random.choice(DEFAULT_COMMAND) + " " + token_+". "
        target_list = object_common_dict['color'][class_name]
        object_id_list = [t[0] for t in target_list]
        if len(object_id_list)<1:
            continue
        reference_list.append(generate_sample(text=text, scan_id=scan_id,target_list=target_list, anchor_list=[], distractor_ids=[],tokens_dict={token_:object_id_list},sub_class='VG_Single_Attribute_Common',gen_class=f'commonrefer-color',need_refine = False))
    
    
    #2. base on material
    
    for class_name in object_common_dict['material'].keys():
        if class_name == 'Fabric/Feather':
            material_name = 'fabric or feather'
        elif class_name == 'Wood/Paper':
            material_name = 'wood or paper'
        elif class_name == 'Concrete/Stone':
            material_name = 'concrete or stone'
        else:
            material_name = class_name.lower()
        name_ = random.choice(DEFAULT_NAME)
        token_ = f"all the {name_} that have {material_name} in material"
        text = random.choice(BEGIN_STATE) + random.choice(DEFAULT_COMMAND) + " " + token_+". "
        target_list = object_common_dict['material'][class_name]
        
        object_id_list = [t[0] for t in target_list]
        if len(object_id_list)<1:
            continue
        reference_list.append(generate_sample(text=text, scan_id=scan_id,target_list=target_list, anchor_list=[], distractor_ids=[],tokens_dict={token_:object_id_list},sub_class='VG_Single_Attribute_Common',gen_class=f'commonrefer-material',need_refine = False))
    
    
    #3. base on placement
    all_placement = ["Standing upright","Lying/Leaning","Hanging/Mounted on the wall"]  
    for class_name in object_common_dict['placement'].keys():
        if class_name == "Lying/Leaning":
            placement_name = "lying or leaning on some surfaces"
        elif class_name == "Standing upright":
            continue
        else:
            placement_name = "hangging or mounted on the wall"
        name_ = random.choice(DEFAULT_NAME)
        token_ = f"all the {placement_name} {name_}"
        text = random.choice(BEGIN_STATE) + random.choice(DEFAULT_COMMAND) + " " + token_+". "
        target_list = object_common_dict['placement'][class_name]
        object_id_list = [t[0] for t in target_list]
        if len(object_id_list)<1:
            continue
        reference_list.append(generate_sample(text=text, scan_id=scan_id,target_list=target_list, anchor_list=[], distractor_ids=[],tokens_dict={token_:object_id_list},sub_class='VG_Single_Attribute_Common',gen_class=f'commonrefer-placement',need_refine = False))
    
    #4. base on shape 
    all_shape = ["Rectanglar","Round(Cylindrical/Spherical/Conial)"]   
    for class_name in object_common_dict['shape'].keys():
        if class_name == "Round(Cylindrical/Spherical/Conial)":
            shape_name = "round"
        else:
            shape_name = class_name.lower()
        name_ = random.choice(DEFAULT_NAME)
        token_ = f"all the {shape_name} {name_}"
        text = random.choice(BEGIN_STATE) + random.choice(DEFAULT_COMMAND) + " " + token_+". "
        target_list = object_common_dict['shape'][class_name]
        object_id_list = [t[0] for t in target_list]
        if len(object_id_list)<1:
            continue
        reference_list.append(generate_sample(text=text, scan_id=scan_id,target_list=target_list, anchor_list=[], distractor_ids=[],tokens_dict={token_:object_id_list},sub_class='VG_Single_Attribute_Common',gen_class=f'commonrefer-shape',need_refine = False))
        
    #5. base on coarse grain category
    for function_class in function_base_category:
        object_type_id = [(object_id,object_type) for object_id,object_type in zip(annotation_data[scan_id]['object_ids'],annotation_data[scan_id]['object_types']) if object_type in function_base_category[function_class]]
        if len(object_type_id)<1:
            continue
        token_ = "all "+function_class
        text = random.choice(BEGIN_STATE) + random.choice(DEFAULT_COMMAND) + " " + token_
        target_list = object_type_id
        object_id_list = [t[0] for t in target_list]
        reference_list.append(generate_sample(text=text, scan_id=scan_id,target_list=target_list, anchor_list=[], distractor_ids=[],tokens_dict={token_:object_id_list},sub_class='VG_Single_Attribute_Common',gen_class=f'commonrefer-coarseclass',need_refine = False))
    #6. base on type
    object_type_id_dict = {}
    for object_id,object_type in zip(annotation_data[scan_id]['object_ids'],annotation_data[scan_id]['object_types']):
        if object_type not in object_type_id_dict.keys() :   
            object_type_id_dict[object_type] = []
        object_type_id_dict[object_type].append(object_id)
    for object_type in object_type_id_dict.keys():
        if object_type == "object":
            continue
        token_ = "all "+PLURAL_DICT[object_type]
        text = random.choice(BEGIN_STATE) + random.choice(DEFAULT_COMMAND) + " " + token_+". "
        target_list = [(id_,object_type) for id_ in object_type_id_dict[object_type]]
        object_id_list = [t[0] for t in target_list]
        reference_list.append(generate_sample(text=text, scan_id=scan_id,target_list=target_list, anchor_list=[], distractor_ids=[],tokens_dict={token_:object_id_list},sub_class='VG_Single_Attribute_Common',gen_class=f'commonrefer-type',need_refine = False))
    
    #7. type distinguish
    color_type_dict = {}
    material_type_dict = {}
    
    for class_name in object_common_dict['color'].keys():
        if class_name == 'White/Beige':
            color_name = 'white or beige'
        elif class_name == 'Gray/Silver':
            color_name = 'gray or silver'
        else:
            color_name = class_name.lower()
        object_type_id = object_common_dict['color'][class_name]
        color_type_dict = {}
        for object_id,object_type in object_type_id:
            if object_type not in color_type_dict.keys():
                color_type_dict[object_type] = []
            color_type_dict[object_type].append(object_id)
        for object_type in color_type_dict.keys():
            if object_type == "object":
                continue
            token_ = f"all the {object_type} that have {color_name} in color"
            text = random.choice(BEGIN_STATE) + random.choice(DEFAULT_COMMAND) + " " + token_+". "
            target_list = [(id_,object_type) for id_ in color_type_dict[object_type]]
            object_id_list = [t[0] for t in target_list]
            reference_list.append(generate_sample(text=text, scan_id=scan_id,target_list=target_list, anchor_list=[], distractor_ids=[],tokens_dict={token_:object_id_list},sub_class='VG_Single_Attribute_Common',gen_class=f'commonrefer-color_mix_type',need_refine = False))
            
    for class_name in object_common_dict['material'].keys():
        if class_name == 'Fabric/Feather':
            material_name = 'fabric or feather'
        elif class_name == 'Wood/Paper':
            material_name = 'wood or paper'
        elif class_name == 'Concrete/Stone':
            material_name = 'concrete or stone'
        else:
            material_name = class_name.lower()
        object_type_id = object_common_dict['material'][class_name]
        material_type_dict = {}
        for object_id,object_type in object_type_id:
            if object_type not in material_type_dict.keys():
                material_type_dict[object_type] = []
            material_type_dict[object_type].append(object_id)
        for object_type in material_type_dict.keys():
            if object_type == "object":
                continue
            token_ = f"all the {object_type} that have {material_name} in material"
            text = random.choice(BEGIN_STATE) + random.choice(DEFAULT_COMMAND) + " " + token_+". "
            target_list = [(id_,object_type) for id_ in material_type_dict[object_type]]
            object_id_list = [t[0] for t in target_list]
            reference_list.append(generate_sample(text=text, scan_id=scan_id,target_list=target_list, anchor_list=[], distractor_ids=[],tokens_dict={token_:object_id_list},sub_class='VG_Single_Attribute_Common',gen_class=f'commonrefer-material_mix_type',need_refine = False))
    
    
    return reference_list

def get_relation_from_base(anchor_id, anchor_type, annotation_data):
    object_ids = annotation_data['object_ids']
    object_types = annotation_data['object_types']
    object_bboxes = annotation_data['bboxes']
    _index = list(object_ids).index(anchor_id)
    anchor_bbox = object_bboxes[_index]
    anchor_volumn = anchor_bbox[3] * anchor_bbox[4] * anchor_bbox[5]
    
    # 1. higher or lower
    higher_list = []
    lower_list = []

    for _index in range(len(object_bboxes)):

        object_bbox = object_bboxes[_index]
        object_id = object_ids[_index]
        object_type = object_types[_index]
        if object_id == anchor_id:
            continue

        if object_bbox[2] > anchor_bbox[2] + anchor_bbox[5] / 2:
            higher_list.append((object_id, object_type))
        if object_bbox[2] < anchor_bbox[2] - anchor_bbox[5] / 2:
            lower_list.append((object_id, object_type))

    # 2. farest or closest

    far_dis = 0
    close_dis = math.inf
    far_id = close_id = anchor_id
    
    for _index in range(len(object_bboxes)):

        object_bbox = object_bboxes[_index]
        object_id = object_ids[_index]
        object_type = object_types[_index]
        if object_id == anchor_id:
            continue
        dis = np.linalg.norm(np.array(anchor_bbox[:3]) - np.array(object_bbox[:3]))
        if dis>far_dis:
            far_dis = dis
            far_id = object_id
        if dis<close_dis:
            close_dis = dis
            close_id = object_id
            
    anchor_dict = dict()
    anchor_dict['higher'] = higher_list
    anchor_dict['lower'] = lower_list
    anchor_dict['farest'] = (far_id, object_types[list(object_ids).index(far_id)])
    anchor_dict['closest'] = (close_id, object_types[list(object_ids).index(close_id)])

    return anchor_dict    
                   
def generate_anchor_base_reference(object_attribute_dict,object_common_dict,annotation_data,ex_scene_id):
    
    # use all the only type to refer others
    reference_list = []
    
    object_ids = annotation_data[ex_scene_id]['object_ids']
    object_types =annotation_data[ex_scene_id]['object_types']
   
    if len(object_ids)<2:
        return reference_list
    object_type_id_dict = {}
    for object_id,object_type in zip(object_ids,object_types):
        if object_type not in object_type_id_dict.keys():
            object_type_id_dict[object_type] = []
        object_type_id_dict[object_type].append(object_id)
    
    anchor_list = []
    for object_type in object_type_id_dict.keys():
        if object_type == "object":
            continue
        if len(object_type_id_dict[object_type])==1:
            anchor_list.append((object_type_id_dict[object_type][0],object_type))
            
    # OO-relation need not downsample
    anchor_OO_list = anchor_list
    for anchor_id,anchor_type in anchor_OO_list:
        if (anchor_id,anchor_type) not in object_attribute_dict.keys() or "common_attributes" not in object_attribute_dict[(anchor_id,anchor_type)]:
            continue
        common_dict = object_attribute_dict[(anchor_id,anchor_type)]["common_attributes"]
        if len(common_dict['color'])==1:
            target_list = object_common_dict['color'][common_dict['color'][0]]
            if len(target_list)<2:
                continue
            target_list = [(id_,object_type) for id_,object_type in target_list if id_!= anchor_id]
            object_id_list = [t[0] for t in target_list]
            name_ = random.choice(DEFAULT_NAME)
            token_ = f"all the {name_} that have a color similar to "
            text = random.choice(BEGIN_STATE) + random.choice(DEFAULT_COMMAND) + " " + token_+f"the {anchor_type}. "
            reference_list.append(generate_sample(text=text, scan_id=ex_scene_id,target_list=target_list, anchor_list=[(anchor_id,anchor_type)], distractor_ids=[],tokens_dict={token_:object_id_list,anchor_type:[anchor_id]},sub_class='VG_Inter_Attribute_OO',gen_class=f'Anchorbase-compare_color',need_refine = False))
            
        if len(common_dict['material'])==1:
            target_list = object_common_dict['material'][common_dict['material'][0]]
            if len(target_list)<2:
                continue
            target_list = [(id_,object_type) for id_,object_type in target_list if id_!= anchor_id]
            object_id_list = [t[0] for t in target_list]
            name_ = random.choice(DEFAULT_NAME)
            token_ = f"all the {name_} that have a material similar to "
            text = random.choice(BEGIN_STATE) + random.choice(DEFAULT_COMMAND) + " " + token_+f"the {anchor_type}. "
            reference_list.append(generate_sample(text=text, scan_id=ex_scene_id,target_list=target_list, anchor_list=[(anchor_id,anchor_type)], distractor_ids=[],tokens_dict={token_:object_id_list,anchor_type:[anchor_id]},sub_class='VG_Inter_Attribute_OO',gen_class=f'Anchorbase-compare_material',need_refine = False))
            
        if len(common_dict['shape'])>0:
            target_list = object_common_dict['shape'][common_dict['shape']]
            if len(target_list)<2:
                continue
            target_list = [(id_,object_type) for id_,object_type in target_list if id_!= anchor_id]
            object_id_list = [t[0] for t in target_list]
            name_ = random.choice(DEFAULT_NAME)
            token_ = f"all the {name_} that have a shape similar to "
            text = random.choice(BEGIN_STATE) + random.choice(DEFAULT_COMMAND) + " " + token_+f"the {anchor_type}. "
            reference_list.append(generate_sample(text=text, scan_id=ex_scene_id,target_list=target_list, anchor_list=[(anchor_id,anchor_type)], distractor_ids=[],tokens_dict={token_:object_id_list,anchor_type:[anchor_id]},sub_class='VG_Inter_Attribute_OO',gen_class=f'Anchorbase-compare_shape',need_refine = False))
    
            
    
    
    # OR-relation need downsample
    anchor_OR_list = random.sample(anchor_list,min(anchor_OR_sample,len(anchor_list)))
    for anchor_id,anchor_type in anchor_OR_list:
        anchor_dict = get_relation_from_base(anchor_id, anchor_type, annotation_data[ex_scene_id])
        for relation_type in anchor_dict.keys():
            
            if relation_type == "higher" or relation_type == "lower":
                object_list = anchor_dict[relation_type]
                name_ = random.choice(DEFAULT_NAME)
                token_ = f"all the {name_} {relation_type} than "
            else:
                object_list = [anchor_dict[relation_type]]
                name_ = random.choice(['item','entity','object','thing'])
                token_ = f"the {name_} {relation_type} to "
            object_id_list = [t[0] for t in object_list]
            text = random.choice(DEFAULT_COMMAND) + " " + token_+"the " + anchor_type +". "
            reference_list.append(generate_sample(text=text, scan_id=ex_scene_id,target_list=object_list, anchor_list=[(anchor_id,anchor_type)], distractor_ids=[],tokens_dict={token_:object_id_list,anchor_type:[anchor_id]},sub_class='VG_Inter_Space_OO',gen_class=f'Anchorbase-template_space',need_refine = False))    
    
    return [reference for reference in reference_list if len(reference['target_id'])>0]        
def generate_space_relation_reference(scene_sr3d_dict, annotation_data, scene_id):
    

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
    
    
    
    object_types = annotation_data[scene_id]['object_types']
    object_ids = annotation_data[scene_id]['object_ids']
    reference_list = []
    wrong_list = ['<>stick', '<>way', 'frame', 'knob', '<>sill', 'cutter', '<>t', '<>pet', 'dryer', 'paper', '<>ry']
    
    for raw_space_item in raw_space_list:

        vg_text, target_dict, all_ids = process_text(raw_space_item["utterance"])
        template = get_tamplate(raw_space_item["utterance"])
  
        
        # check if the item is valid
        skip_flag = False
        
        # 1. check if the templete is valid
        for wrong_word in wrong_list:
            if wrong_word in template:
                skip_flag = True
        if len(template) == 0 or len(vg_text) == 0:
            skip_flag = True
        
        # 2. check if the target object is valid
        for _id in all_ids:
            if _id not in object_ids:
                skip_flag = True
                continue
            _type = object_types[list(object_ids).index(_id)]
            if _type in EXCLUDED_OBJECTS:
                skip_flag = True
                
        if skip_flag:
            continue
        target_type = object_types[list(object_ids).index(raw_space_item["target_id"])]
        target_id = raw_space_item["target_id"]
        
        anchor_ids  = raw_space_item["anchor_ids"]
        anchor_types = [object_types[list(object_ids).index(anchor_id)] for anchor_id in anchor_ids]
        
        tokens_dict = {}
        tokens_dict[target_type] = [target_id]
        for anchor_id,anchor_type in zip(anchor_ids,anchor_types):
            tokens_dict[anchor_type] = [anchor_id]
      
        reference_dict = generate_sample(text=vg_text, 
                                         scan_id = scene_id,
                                         target_list=[(target_id,target_type)], 
                                         anchor_list=[t for t in zip(anchor_ids,anchor_types)], 
                                         distractor_ids=raw_space_item["distractor_ids"],
                                         tokens_dict=tokens_dict,
                                         sub_class='VG_Inter_Space_OO',
                                         gen_class=f'sr3d-space',
                                         need_refine = False)
       
        reference_list.append(reference_dict)

    return reference_list

def generate_function_relation_reference(region_anno_dict,region_id,annotation_data,ex_scene_id):
    total_object_ids = annotation_data[ex_scene_id]['object_ids']
    total_object_types =annotation_data[ex_scene_id]['object_types']
    reference_list = []

    function_relations = region_anno_dict[0]
    large_class_dict = region_anno_dict[2]
    
    
    # for multi region role
    for object_list in large_class_dict.keys():
        default_reference = random.choice(DEFAULT_NAME)
        text = f'This is a description of {numberToWords(len(object_list))} {default_reference} in the {region_id.split("_")[1]}: {large_class_dict[object_list]} Please {random.choice(DEFAULT_COMMAND)} them. '
        
        if isinstance(object_list, str):
            object_list = eval(object_list)
        
        target_list = [(int(object_name.split('_')[1][:-1]) , object_name.split('_')[0][1:]) for object_name in object_list if int(object_name.split('_')[1][:-1]) in total_object_ids]
        object_id_list = [int(object_name.split('_')[1][:-1]) for object_name in object_list if int(object_name.split('_')[1][:-1]) in total_object_ids]
        token_ = f'{numberToWords(len(object_list))} {default_reference}'
        reference_list.append(generate_sample(text=text, scan_id = ex_scene_id,target_list=target_list, anchor_list=[], distractor_ids=[],tokens_dict={token_:object_id_list},sub_class='VG_Inter_OR',gen_class=f'region-bigclass',need_refine = True))  
    
    
    # for oo-function
    # if allow to dwonsample, choose the anchor is better.
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
        if anchor_id not in total_object_ids:
            print("ERROR ID in FunctionOO anchor")
            continue
        anchor_function_text_dict = {}
        for target_name, text in anchor_function_relations[anchor_name]:
            target_type, target_id = target_name.split('_')[0][1:], int(target_name.split('_')[1][:-1])
            if target_id not in total_object_ids:
                print("ERROR ID in FunctionOO target")
                continue
            text = text.replace(target_name, 'X')
            text = text.replace(anchor_name, anchor_type)
            text = text[:-1] + f' in the {region_id.split("_")[1]}. Please {random.choice(DEFAULT_COMMAND)} the X. '

            if text not in anchor_function_text_dict:
                anchor_function_text_dict[text] = [(target_id, target_type)]
            else:
                anchor_function_text_dict[text].append((target_id, target_type))
                
        for text in anchor_function_text_dict.keys():
            target_list = anchor_function_text_dict[text]
            target_id_list = [t[0] for t in target_list]
            token_1 = 'X'
            token_2 = anchor_type
            reference_list.append(generate_sample(text=text, scan_id = ex_scene_id,target_list=target_list, anchor_list=[(anchor_id,anchor_type)], distractor_ids=[],tokens_dict={token_1:target_id_list,token_2:[anchor_id]},sub_class='VG_Inter_Attribute_OO',gen_class=f'region-function_relation',need_refine = True))  
    return reference_list

