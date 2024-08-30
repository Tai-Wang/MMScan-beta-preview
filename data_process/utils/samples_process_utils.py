import os
def find_matches(sentence, target_dict):
    """
    Find all occurrences of string s in sentence.
    Returns a list of lists [[start, end], [start, end],...] of the matches.
    """
    matches = dict()

    for ss in target_dict.keys():
        s = ss.lower()
        start = 0
        while True:
            start = sentence.lower().find(s, start)
            if start == -1:
                break
            end = start + len(s)
            for object_id in target_dict[ss]:
                _id = int(object_id)
                if _id not in matches.keys():
                    matches[_id] = [[start, end]]
                else:
                    matches[_id].append([start, end])
            start += 1

    return matches
def remove_(text):
    start = 0
    end = len(text)
    while start<len(text)-1 and (text[start]==' ' or text[start]=='['):
        start += 1
    while end > start and (text[end-1]==' ' or text[end-1]==']'):
        end -= 1
    return text[start:end]

def get_out_phrases(text):
    if '[' not in text or ']' not in text:
        return []
    return [remove_(phrase) for phrase in text.split('&') if len(remove_(phrase))>0]
    

def check_vg_process(user_text,gpt_text,id_list=None):
    """_summary_

    Args:
        user_text (_type_): intputs
        gpt_text (_type_): _outputs

    Returns:
        _type_: bool
        
    (1) The number of token_positives in user_text and gpt_text should be the same.
    
    (2) token_positives should appear in the gpt_text.
    
    """
    if '*' not in user_text or '*' not in gpt_text:
        return '',[]
    text1,token_positive_list1 = user_text.split('*')
    text2,token_positive_list2 = gpt_text.split('*')
    token_positive_list1 = get_out_phrases(token_positive_list1)
    token_positive_list2 = get_out_phrases(token_positive_list2)

    
    if len(token_positive_list1)!=len(token_positive_list2):
        return '',[]
    for token_positive in token_positive_list2:
        if token_positive.lower() not in text2.lower():
            return '',[]
    if id_list is not None and  len(id_list)!=len(token_positive_list2):
        print("*******************")
        print(user_text,gpt_text)
        print(token_positive_list2,id_list)
        
    return remove_(text2),token_positive_list2

def check_qa_process(user_text,gpt_text):
    """_summary_

    Args:
        user_text (_type_): intputs
        gpt_text (_type_): _outputs

    Returns:
        _type_: bool
        
    (1) The number of answers in user_text and gpt_text should be the same.
    
    """
    
    if '*' not in user_text or '*' not in gpt_text or len(gpt_text.split('*'))>2:
        return '',[]

    question1,answers1 = user_text.split('*')
    question2,answers2 = gpt_text.split('*')
    
    answers_list1 = get_out_phrases(answers1)
    answers_list2 = get_out_phrases(answers2)
    
    if len(answers_list1)!=len(answers_list2):
        return '',[]
    for index_ in range(len(answers_list2)):
        if "Yes, there is." in answers_list1[index_] or "No, there isn't." in answers_list1[index_]:
            answers_list2[index_] = answers_list1[index_]
    return remove_(question2),answers_list2

example_vg = {"sub_class": "VG_Single_EQ", "scan_id": "1mp3d_0000_region2", "target_id": [4], "distractor_ids": [], "text": "Select all the doorframes and doors. ", "target": ["doorframe"], "anchors": [], "anchor_ids": [], "tokens_positive": {"4": [[15, 25],[30,35]],"5": [[15, 25]]}, "ID": "VG_Single_EQ__1mp3d_0000_region2__1"}

def encode_vg_item(vg_dict):
    """_summary_

    Input the VG dict, return the texts and matching ids
    
    """
    tokens_positive_dict = vg_dict["tokens_positive"]
    text = vg_dict["text"]
    tokens_positive_trans_dict = dict()
    for object_id in tokens_positive_dict.keys():
        for interval in tokens_positive_dict[object_id]:
            start,end = interval
            sub_text = text[start:end]
            if sub_text not in tokens_positive_trans_dict:
                tokens_positive_trans_dict[sub_text] = []
            if int(object_id) not in tokens_positive_trans_dict[sub_text]:
                tokens_positive_trans_dict[sub_text].append(int(object_id))
    text_list = []
    id_list = []
    for sub_text in tokens_positive_trans_dict.keys():
        text_list.append(sub_text)
        id_list.append(tokens_positive_trans_dict[sub_text])
    return text_list,id_list
    

def decode_vg_item(text, text_list,id_list):
    """_summary_

    Input the processed text list and the id_list, return the tokens positive dict
    
    """
    tokens_positive_trans_dict = dict()

    assert len(text_list)==len(id_list)
    for _index in range(len(text_list)):
        tokens_positive_trans_dict[text_list[_index]] = id_list[_index]
    tokens_positive_dict = dict()
    for sub_text in tokens_positive_trans_dict.keys():
        assert sub_text.lower() in text.lower()
    return find_matches(text, tokens_positive_trans_dict)
    

if __name__ == '__main__':
    # vg_user_text = 'This black comfortable chair. Please find table front of it. * [black comfortable chair & table]'
    # vg_gpt_text = 'This is a black comfortable chair. Please find the table in front of it. * [a black comfortable chair &  the table]'
    # qa_user_text = 'I will give you some inforation of object, what is color of it? * [It black. & Dark. ]'
    # qa_gpt_text = 'I will give you some inforation of an object, what is the color of it? * [Its color is black. & It has a dark color. ]'
    
    # #print(check_vg_process(vg_user_text,vg_gpt_text))
    
    # print(check_qa_process(qa_user_text,qa_gpt_text))
    input = 'aaaaaaaa * [dddd]'
    print(get_out_phrases(input.split('*')[1]))
