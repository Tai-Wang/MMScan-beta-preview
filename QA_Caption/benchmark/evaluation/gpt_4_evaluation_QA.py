import threading
import requests
import json
from time import sleep
import random
import re
import os.path as osp
from tqdm import tqdm

API_KEY = ''
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

system = '''Evaluate a model-generated QA result against a human-generated answer for a 3D model. Begin by identifying all key points in the human answer, then compare each of these attributes in the model-generated answer, providing reasons for each evaluation. Following the examples provided, complete the last row of the table:'''
template = '''| Question                                                     | Human Answer                                                 | Model Answer                                                 | Total Key Points | Correct | Ambiguous | Error | Reason        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------- | ------- | --------- | ----- | ------------------------------------------------------------ |
| What is the purpose of the object                            | Hanging clothes                                              | Providing storage space                                      | 1                | 1       | 0         | 0     | A place for hanging clothes also provides storage space.     |
| What is the usage of the object                              | Support the feet                                             | Suitable for daily use                                       | 1                | 0       | 1         | 0     | "Suitable for daily use" is ambiguous and does not specify support for the feet. |
| There is a toilet region in this room, these objects are in it. What is the location and function description of this region? | This is a bathroom that focuses on cleanliness and grooming activities. It includes facilities such as a sink, toilet, and a storage vanity. | This area appears to be a bathroom or toilet area, specifically designed for personal hygiene and sanitation. | 3                | 2       | 0         | 1     | The model identifies the area as a bathroom and notes its usage, but misses specific objects like the sink and vanity. |
| Are these two objects the same in shape? Why?                | No, the chair is irregular and the lamp is cylindrical.      | Yes, the chair and the chair are both irregular.             | 2                | 1       | 0         | 1     | The model correctly identifies one chair but fails to recognize another lamp and its shape. |
| {question}                                                   | {human_answer}                                               | {model_answer}                                               |                  |         |           |       |                                                              |
Rules:
1. Focus on comparing the two answers.
2. Only include the completed last row of the table in your response, excluding the header.'''


def generate_chat_completion(messages, model="gpt-4o-2024-05-13", temperature=1, max_tokens=None):# gpt-3.5-turbo-0125 gpt-4-1106-preview
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    try:
        response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))
    except:
        sleep(20)
        return generate_chat_completion(messages, 
                                        model=model, 
                                        temperature=temperature, 
                                        max_tokens=max_tokens)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(response.status_code)
        sleep(20)
        return generate_chat_completion(messages, 
                                        model=model, 
                                        temperature=temperature, 
                                        max_tokens=max_tokens)

def process(data_list,idx,output_path):
    processed_json = []
    bar = tqdm(data_list.keys())
    for i in bar:
        if data_list[i]["gt"][0].lower() == data_list[i]["pred"][0].lower():
            processed_json.append({
                    "gpt_pred":"",
                    "total":1,
                    "correct":1,
                    "ambiguous":0,
                    "error":0,
                    "reason":"same",
                    "info":data_list[i],
                    "qs_type":i.split("-")[0]
                })
            continue
        elif bool(re.search(r'\byes\b|\bno\b', data_list[i]["gt"][0].lower()))  and \
                len(data_list[i]["gt"][0])<10: # HACK: avoide Yes or no in a long scentence
            if bool(re.search(r'\bno\b', data_list[i]["gt"][0].lower())) and bool(re.search(r'\bno\b', data_list[i]["pred"][0].lower())) or\
                bool(re.search(r'\byes\b', data_list[i]["gt"][0].lower())) and bool(re.search(r'\byes\b', data_list[i]["pred"][0].lower())):
                processed_json.append({
                    "gpt_pred":"skip",
                    "total":1,
                    "correct":1,
                    "ambiguous":0,
                    "error":0,
                    "reason":"same",
                    "info":data_list[i],
                    "qs_type":i.split("-")[0]
                })
            else:
                processed_json.append({
                    "gpt_pred":"skip",
                    "total":1,
                    "correct":0,
                    "ambiguous":0,
                    "error":1,
                    "reason":"",
                    "info":data_list[i],
                    "qs_type":i.split("-")[0]
                })
            print("pass one query")
            continue

        messages=[{"role": "system", "content":system},
                  {"role":"user", "content":template.format(question=data_list[i]["question"],
                                                            human_answer=data_list[i]["gt"][0],
                                                            model_answer=data_list[i]["pred"][0]
                                                            )}]

        chat = None
        for _ in range(0,3):
            try:
                chat = generate_chat_completion(messages=messages)
                split = chat.split("|")
                split = [i.strip() for i in split]
                assert len(split) == 10
                processed_json.append({
                    "gpt_pred":chat,
                    "total":int(split[4]),
                    "correct":int(split[5]),
                    "ambiguous":int(split[6]),
                    "error":int(split[7]),
                    "reason":split[8],
                    "info":data_list[i],
                    "qs_type":i.split("-")[0]
                })
                print(" ============================================================== ")
                print("|".join(split[2:9]))
                print(" ============================================================== ")
                break
            except Exception as e:
                print(e)
                pass
    json.dump(processed_json,open(osp.join(output_path,f"gpt_eval_QA_rank_{idx}.json",'w')))
    return processed_json

if __name__ == "__main__":

    data = json.load(open("path/to/eval/test.json"))
    output_path = '.'
    num_samples = 4000

    random.seed(0)
    list_data = list(data.items())
    sampled_items = random.sample(list_data, min(num_samples,len(list_data)))
    sampled_dict = dict(sampled_items)
    num_threads = 8

    keys = list(sampled_dict.keys())
    segment_size = len(keys) // num_threads
    threads = []

    for i in range(num_threads):
        start = i * segment_size
        end = start + segment_size if i < num_threads - 1 else len(keys)
        # Create a sub-dictionary for each thread
        sub_dict = {key: sampled_dict[key] for key in keys[start:end]}
        thread = threading.Thread(target=process, args=(sub_dict, i, output_path))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
