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

template = '''Evaluate a model-generated object caption result against a  human-generated caption for a 3D language model. Focus on the following aspects:  object type(the center object in the caption), color, shape, position, function, and design. If the human caption does not include any of these aspects, mark this attribute as 0; otherwise, use 1 or -1 to indicate matching or incorrect(missing).  Following the examples provided, complete the last row of the table:

| human caption                                                | model caption                                                | object type | color | shape | position | function | design |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------- | ----- | ----- | -------- | -------- | ------ |
| In the center of the room stands a spacious arched doorway, crafted from dark wood, exuding a strong sense of elegance. It is in excellent condition. Compared to the other items in the room, this door is quite large, vertically placed against the beige wall. The door is functional, providing access to another room. Its design features include dark wood veneer and an arched top, adding a touch of refinement to the room. | The doorframe is a prominent feature in the room. It is rectangular in shape and painted in a light beige or off-white color. Made of wood, it exudes a warm and inviting feeling. The doorframe is relatively large, taking up a significant portion of the wall space. In terms of functionality, it serves as a typical doorframe, connecting one room to another. Compared to similar items, this doorframe is well-maintained with no obvious signs of damage or wear. Its design is simple yet elegant, blending seamlessly into the overall aesthetic of the room. | -1           | -1    | 0     | -1       | 1        | -1     |
| This is a single object, a chair. The chair is white and made of plastic. It is positioned slightly tilted to the right on a light-colored floor. The chair is empty and well-maintained. Its design is simple, without armrests or a backrest. Compared to similar items, this chair is of medium size. Its function is to provide a comfortable seat, and it effectively fulfills this purpose. | This is a white, rectangular box. The box is made of material such as plastic or stainless steel, and it's in good condition with no obvious signs of wear or damage. It's placed vertically on the floor, and there are no other items next to it. Compared to similar items, this box is of medium size. Its design is simple and practical, without any extra features or decorations. | -1          | 1     | 0     | -1       | -1       | -1     |
| {human_caption}                                              | {model_caption}                                              |             |       |       |          |          |        |

Rules:
1. First, carefully read the human caption to identify the six key attributes, then compare them with the model caption.
2. Format the 6 scores separated by "|" (beginning and ending with "|"). Respond with only the formatted scores. '''

def generate_chat_completion(messages, model="gpt-4o-2024-05-13", temperature=1, max_tokens=None):# gpt-3.5-turbo-0125 gpt-4-1106-preview gpt-4o-2024-05-13
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
        messages=[{"role": "system", "content":template.format(human_caption=data_list[i]["gt"][0],
                                                                model_caption=data_list[i]["pred"][0]
                                                                )}]
        chat = None
        for _ in range(0,3):
            try:
                chat = generate_chat_completion(messages=messages)
                split = chat.split("|")
                split = [i.strip() for i in split]
                print(len(split),split)
                assert len(split) == 8
                processed_json.append({
                    "gt":data_list[i]["gt"],
                    "perd":data_list[i]["pred"][0],
                    "gpt_pred":chat,
                    "object_type":int(split[1]),
                    "color":int(split[2]),
                    "shape":int(split[3]),
                    "position":int(split[4]),
                    "function":int(split[5]),
                    "design":int(split[6]),
                })
                for i in processed_json[-1]:
                    print(i,processed_json[-1][i])
                break
            except Exception as e:
                print(e)
                pass
    json.dump(processed_json,open(osp.join(output_path,f"gpt_eval_cap_rank_{idx}.json"),'w'))
    return processed_json

if __name__ == "__main__":
    data = json.load(open("path/to/cap_dict.json"))
    output_path = "."
    random.seed(0)
    num_samples = 2000
    list_data = list(data.items())
    sampled_items = random.sample(list_data, min(num_samples,len(list_data)))
    sampled_dict = dict(sampled_items)

    keys = list(sampled_dict.keys())
    segment_size = len(keys) // num_threads
    threads = []

    for i in range(num_threads):
        start = i * segment_size
        end = start + segment_size if i < num_threads - 1 else len(keys)
        # Create a sub-dictionary for each thread
        sub_dict = {key: sampled_dict[key] for key in keys[start:end]}
        thread = threading.Thread(target=process, args=(sub_dict, i,output_path))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
