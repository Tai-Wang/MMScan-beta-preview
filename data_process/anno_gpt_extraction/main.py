import os
import json

import random
import numpy as np
import mmengine
from tqdm import tqdm

from utils.object_attr_template import Common_Descripition,EXAMPLE_EXTRACT1,EXAMPLE_EXTRACT2,REFERING_TYPES,PLURAL_DICT,all_translate
from utils.file_read_check import check_dict_matching,check_object_text_anno_is_valid,read_annotation_pickle
from utils.openai_api import translate,get_content_groups_from_source_groups,mimic_chat_budget
from utils.data_path import *

def get_coarse_type():
    system_prompt = f"You are given a text description of an object. Your task is to identify the attributes of the object. The 11 attributes are: 1. fine grained category, 2. color, 3. texture, 4. material, 5. weight, 6. size, 7. shape, 8. placement (e.g. standing upright, piled up, leaning, lying flat, hanging), 9. state (e.g. open, closed, locked, empty), 10. function, 11. coarse grained category of the object. The coarse grained category means the larger class this object belongs to, 12. other features. For every attribute, you should give a text to describe it, missing attributes can be left blank. Please reply in json, in the following format: {str(example_extraction)}.  The result should be a dict with 12 attributes as the keys."
    
    content_groups = get_content_groups_from_source_groups([[text]])
    print(content_groups)
    messages, token_usage = mimic_chat_budget(content_groups, system_prompt, num_turns_expensive=0,
                                              report_token_usage=True, json_mode=True)
    response = messages[-1]
    if response["role"] != "assistant":
        print(text)
        return {}
    response_dict = json.loads(response["content"])
    
    return response_dict
    

def back_translate_text_anno_json(json_path):
    """
        Back-translate the modified_description field in the json file to English, and store it in the original json file.
        Returns None.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    object_type = json_path.split("user_shujutang_czc")[-1].split('_')[1]
    if json_data.get("modified_description_en", ""):
        return
    if not check_object_text_anno_is_valid(json_data):
        return
    modified_description = json_data.get("modified_description", "")
    if modified_description:
        back_translated_description = translate(modified_description, "zh", "en", object_type_hint=object_type)
        json_data["modified_description_en"] = back_translated_description
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)


def extract_unique_from_text(text):
    """
        Get the unique attributes of the object from the text description.
    """
    example_extraction = {"fine grained category":"Story book",
                          "color":"Black with a little grey",
                          "material":"Wooden and papers",
                          "texture":"",
                          "weight":"A little heavy",
                          "size":"Small",
                          "shape":"Rectangular",
                          "placement":"Just lying flat on the table",
                          "state":"Closed",
                          "function":"allow people to get knowledge from it",
                          "coarse grained category": "Study and office supplies",
                          "other features":""
                          }
    system_prompt = f"You are given a text description of an object. Your task is to identify the attributes of the object. The 11 attributes are: 1. fine grained category, 2. color, 3. texture, 4. material, 5. weight, 6. size, 7. shape, 8. placement (e.g. standing upright, piled up, leaning, lying flat, hanging), 9. state (e.g. open, closed, locked, empty), 10. function, 11. coarse grained category of the object. The coarse grained category means the larger class this object belongs to, 12. other features. For every attribute, you should give a text to describe it, missing attributes can be left blank. Please reply in json, in the following format: {str(example_extraction)}.  The result should be a dict with 12 attributes as the keys."
    
    content_groups = get_content_groups_from_source_groups([[text]])
    print(content_groups)
    messages, token_usage = mimic_chat_budget(content_groups, system_prompt, num_turns_expensive=0,
                                              report_token_usage=True, json_mode=True)
    response = messages[-1]
    if response["role"] != "assistant":
        print(text)
        return {}
    response_dict = json.loads(response["content"])
    
    return response_dict



def extract_common_from_text(text):
    """
        get the common attributes in defined brackets
    """
    example_extraction = {"coarse grained category": "Furniture",
                          "color": "Black",
                          "material": "Wood",
                          "shape": "Rectangular",
                          "weight": "Medium",
                          "size": "Medium",
                          "placement": "Standing upright"
                        }
    system_prompt = f"You are given a text description of an object. Your task is to choose a word I give to identify the attributes of the object. The {len(Common_Descripition)} attributes are: 1. coarse grained category, 2. color, 3. material, 4. weight, 5. size, 6. shape, 7. placement. I will give you a dict, the keys of it are the attribute names, the values of it are lists of word you can choose to match the attribute . The dict is {str(Common_Descripition)}. For every attribute, you should choose a word from its list to match it, if none of them match, it can be left blank. Please reply in json, in the following format: {str(example_extraction)}.  The result should be a dict with 7 attributes as the keys, the words I give as the values."
  
    content_groups = get_content_groups_from_source_groups([[text]])
    messages, token_usage = mimic_chat_budget(content_groups, system_prompt, num_turns_expensive=0,
                                              report_token_usage=True, json_mode=True)
    response = messages[-1]
    if response["role"] != "assistant":
        print(response)
        return {}
    response_dict = json.loads(response["content"])
    return response_dict

def common_check_and_fix(common_dict,sync_dict):
    
    cnt = 0
    for attribute in common_dict.keys():
        
        # ensure the reply is from the list
        if len(common_dict[attribute])>0 and common_dict[attribute] not in Common_Descripition[attribute]:
            cnt+=1
            for possible_answer in Common_Descripition[attribute]:
                if possible_answer.lower() in common_dict[attribute].lower():
                    common_dict[attribute] = possible_answer
            if common_dict[attribute] not in Common_Descripition[attribute]:
                common_dict[attribute] = ''
        # sync with the unique answer
        if sync_dict is not None and len(sync_dict[attribute])==0:
            common_dict[attribute] = ''
    return common_dict,cnt                  
                    
def generate_unique_Desc(json_path):
    
    
    # check the infos
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    object_type = json_path.split("user_shujutang_czc")[-1].split('_')[1]
    if not json_data.get("modified_description_en", ""):
        return
    if not check_object_text_anno_is_valid(json_data):
        return
    if json_data.get("unique_description", ""):
        return
    modified_description_en = json_data.get("modified_description_en", "")
   
    # retry-gpt
    max_try = 3
    while max_try>0:
        result_dict = extract_unique_from_text(modified_description_en)
        if check_dict_matching(result_dict,EXAMPLE_EXTRACT1):
            break
        max_try += -1
        
    # store the result
    if object_type!='object':
        result_dict['type'] = object_type
    json_data["unique_description"] = result_dict
    if max_try>0:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
    else:
        print(f"process {json_path} fail!")
        
        

def generate_common_Desc(json_path):
    
    # check the infos
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    object_type = json_path.split("user_shujutang_czc")[-1].split('_')[1]
    if not json_data.get("modified_description_en", ""):
        return
    if not check_object_text_anno_is_valid(json_data):
        return
    if json_data.get("common_description", ""):
        return
    modified_description_en = json_data.get("modified_description_en", "")
    
    # retry-gpt
    max_try = 3
    while max_try>0:
        result_dict = extract_common_from_text(modified_description_en)
        if check_dict_matching(result_dict,EXAMPLE_EXTRACT2):
            break
        max_try += -1
        
    # store the result
    sync_dict = json_data.get("unique_description")
    result_dict,_ = common_check_and_fix(result_dict, sync_dict)
    if object_type!='object':
        result_dict['type'] = object_type
    
    json_data["common_description"] = result_dict
    
    if max_try>0:

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
    else:
        print(f"process {json_path} fail!")


def Desc_check(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    if not json_data.get("modified_description_en", ""):
        return 0
    if not check_object_text_anno_is_valid(json_data):
        return 0
    sync_dict1 = json_data.get("unique_description")
    sync_dict2 = json_data.get("common_description")
    if sync_dict1 is None or sync_dict2 is None:
        return 0
    return 1
def Desc_Extracting(json_path):
  
    generate_unique_Desc(json_path)
    generate_common_Desc(json_path)



# differ-process

def find_the_unique(text_list,object_type):
    # choose unique object from the list
    object_info = '('
    for k in range(len(text_list)):
        object_info += f"<{object_type}_{k+1}>, "
    object_info = object_info[:-2] + ')'
    system_prompt = f"You are given some texts, which are descripitions of some objects {object_info}. Some of them are similar and some of them are significantly different. 1. Your task is to identify and choose which one can be distinguished well from others, mainly in color/shape/material/placement. 2. You need to give some 'referring phrases' for each {object_type} you choose, such as 'the white and rectanglar chair'/'the picture on the wall with dark color', obviously, every 'referring phrases' should only refer to only one of the {object_type}. 3. Remember, you do not have to choose all objects, but choose the objects that are significantly different from others, so the referring phrases of it should only match itself. 4. My expected output is a json file, which is a dict with the <{object_type}_id> you choose as the keys and the referring phrases as the values, such as"
    system_prompt +=" {'<chair_1>': 'the white and rectanglar chair', '<chair_3>': 'the wooden circlar chair'}. "
    
    input_dict = {}
    for k in range(len(text_list)):
        input_dict[f"<{object_type}_{k+1}>"] = text_list[k]
    print(input_dict)
    content_groups = get_content_groups_from_source_groups([[str(input_dict)]])
    messages, token_usage = mimic_chat_budget(content_groups, system_prompt, num_turns_expensive=0,
                                              report_token_usage=True, json_mode=True)
    response = messages[-1]
    if response["role"] != "assistant":
        print(response)
        return {}
    response_dict = json.loads(response["content"])
    return response_dict


def generate_differ_pair(text1,text2,object1,object2):
    # compare two objects in some attributes
    query_attribute = ['color','shape','material','placement']
    system_prompt = f"You are given two descriptions of two objects. 1. Your task is to think of these questions: "
    for attribute in query_attribute:
        system_prompt += f"Are they similar in the {attribute}? "
    system_prompt+= "2. Don't be too strict, answer 'Yes' as long as the difference is not significant. You need to take responsibility for your answers, answer 'Not sure' if you are not sure."
    system_prompt+= "3. Placement refers to the layout of the object:Don't judge based on its spatial relationship with other objects. If they are: 1. both standing upright. 2. both lying or leaning on some surface 3. both hanging or mounted on the wall, they are not significantly different in placement. Lying and leaning are regarded as the some placement (resting on surfaces)."
    system_prompt+= "4. For every answer, you should answer 'Yes' or 'No' and you should give me the reason. Don't use the words like 'descibed as' in the reason, make it simple and less than 20 words. You need to give me a json file in this format: {'color': Yes/No/Not sure * Reason, 'shape': Yes/No/Not sure * Reason, ...}.(e.g.{'color': 'Yes * because the table is black and the chair is the same.'})"

    
    text = f"The type of first object is {object1}, the description is: {text1}; the type of second object is {object2}, the description is: {text2}.  Your reason should use 'the {object1}' and 'the {object2}' as the subjects and don't use the words like 'descibed as'."
    
    content_groups = get_content_groups_from_source_groups([[text]])
    messages, token_usage = mimic_chat_budget(content_groups, system_prompt, num_turns_expensive=0,
                                              report_token_usage=True, json_mode=True)
    response = messages[-1]
    if response["role"] != "assistant":
        print(response)
        return {}
    response_dict = json.loads(response["content"])
    return response_dict  
   
def generate_qa_pair(task_name):
    scan_id,path1,path2,object_id1,object_id2,object1,object2 = task_name.split('**')
    with open(path1, "r") as f:
        object_data1= json.load(f)
    with open(path2, "r") as f:
        object_data2= json.load(f)
    
    text1 = object_data1['modified_description_en']
    text2 = object_data2['modified_description_en']
    result_dict = generate_differ_pair(text1,text2,object1,object2)
    if len(result_dict)>0:
        os.makedirs(EXTRACT_PATH+'/qa_pairs/'+scan_id,exist_ok=True)
        with open(EXTRACT_PATH+'/qa_pairs/'+scan_id+'/'+object_id1+'__'+object_id2+'.json', "w") as f:
            json.dump(result_dict, f)

def get_list_from_text(text):
    text = text[1:-1]
    text_list = text.split(",")
    new_list = []
    for text_ in text_list:
        new_list.append(text_.strip())
    return new_list


if __name__ == '__main__':
    num_proc=30
    
    # 1. extrect atttributes
    DATA_ROOT = OBJ_DATA_ROOT
    scene_ids = os.listdir(DATA_ROOT)
    tasks = []
    for scene_id in tqdm(scene_ids):
        scene_dir = os.path.join(DATA_ROOT, scene_id)
        object_text_annos_dir = os.path.join(scene_dir, "corpora_object", "user_shujutang_czc")
        object_text_anno_paths = os.listdir(object_text_annos_dir)
        object_text_anno_paths = [os.path.join(object_text_annos_dir, p) for p in object_text_anno_paths if p.endswith(".json")]
        tasks.extend(object_text_anno_paths)
    
    mmengine.track_parallel_progress(Desc_Extracting, tasks, nproc=num_proc)

    # 2. find the unqiue/difference

    Sample_Size = 30        
    os.makedirs(EXTRACT_PATH+'/qa_pairs',exist_ok=True)
    tasks = []
    
    for ex_scene_id in tqdm(os.listdir(OBJ_DATA_ROOT)):
        anno_objects = []
        object_path = f"{OBJ_DATA_ROOT}/{ex_scene_id}/corpora_object/user_shujutang_czc"
        for json_path in os.listdir(object_path):
            with open(object_path+'/'+json_path, "r") as f:
                object_data= json.load(f)
            object_id,object_type = int(json_path.split("_")[0]),json_path.split("_")[1]
            if 'modified_description_en' not in object_data.keys():
                continue
            anno_objects.append({"path":object_path+'/'+json_path,"type":object_type,"id":object_id})
        if os.path.exists(EXTRACT_PATH+'/qa_pairs/'+ex_scene_id):
            print(len(os.listdir(EXTRACT_PATH+'/qa_pairs/'+ex_scene_id)))
            if len(os.listdir(EXTRACT_PATH+'/qa_pairs/'+ex_scene_id))>=Sample_Size:
                continue
        choice_tasks = []
        for index_1 in range(len(anno_objects)):
            for index_2 in range(index_1+1,len(anno_objects)):
                choice_tasks.append(ex_scene_id+"**"+anno_objects[index_1]["path"]+'**'+anno_objects[index_2]["path"]+'**'+str(anno_objects[index_1]["id"])+'**'+str(anno_objects[index_2]["id"])+'**'+anno_objects[index_1]["type"]+'**'+anno_objects[index_2]["type"])
        tasks+=random.sample(choice_tasks,min(Sample_Size,len(choice_tasks)))
    mmengine.track_parallel_progress(generate_qa_pair, tasks, nproc=num_proc)

