from glob import glob
import json
from copy import deepcopy
import re
from tqdm import tqdm
import os.path as osp
path_2_eval = "/mnt/petrelfs/yangshuai1/rep/embodiedscanL_LLM_evaluation/gpt_eval_ll3da_cap_*.json"

all_files = glob(osp.join(path_2_eval))

data = []
for file in all_files:
    data += json.load(open(file))

template = {
    "object_type":0,
    "color": 0,
    "shape": 0,
    "position": 0,
    "function": 0,
    "design": 0,
}

c = deepcopy(template)
p = deepcopy(template)
n = deepcopy(template)

for i in data:
    for k in i:
        if i[k]!=0 and isinstance(i[k],int):
            c[k]+=1
            if i[k]==-1:
                n[k]+=1
            else:
                p[k]+=1

overall_c = 0
overall_s = 0 
h_s = 0
c_s = 0
for k in c:
    overall_c+=c[k]
    overall_s+=p[k]#-n[k]
    h_s+=n[k]
    c_s+=p[k]
    print((p[k])/c[k]) # -n[k]

print(overall_s/overall_c)



    
