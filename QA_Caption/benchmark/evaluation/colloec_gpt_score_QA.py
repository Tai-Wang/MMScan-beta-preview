from glob import glob
import json
from copy import deepcopy
import re
from tqdm import tqdm
import os.path as osp

def main(path_2_eval):
    all_files = glob(osp.join(path_2_eval,"*.json"))

    data = []
    for file in all_files:
        data += json.load(open(file))

    template = {
        "total":0,
        "correct":0,
        "ambiguous":0,
        "error":0,
    }

    metrics = ['STa', 'STs', 'OOa', 'OOs', 'OR', 'overall', 'Advanced']

    deep_copies = {var: deepcopy(template) for var in metrics}

    STa = deep_copies['STa']
    STs = deep_copies['STs']
    OOa = deep_copies['OOa']
    OOs = deep_copies['OOs']
    OR = deep_copies['OR']
    overall = deep_copies['overall']
    Advanced = deep_copies['Advanced']

    for l in tqdm(data):
        eval_type = l["qs_type"]
        if "Attribute_OO" in eval_type:
            target = OOa
        elif "Space_OO" in eval_type:
            target = OOs
        elif "EQ" in eval_type or "Single_Attribute" in eval_type:
            target = STa
        elif "OR" in eval_type:
            target = OR
        elif "Single_Space" in eval_type:
            target = STs
        elif "Advanced" in eval_type:
            target = Advanced
        else:
            raise NotImplementedError

        target["total"]+=l["total"]
        target["correct"]+=l["correct"]
        target["ambiguous"]+=l["ambiguous"]
        target["error"]+=l["error"]

    for k in overall:
        overall[k] = STa[k] + STs[k] +  OOa[k] + OOs[k] + OR[k] + Advanced[k]
    print(" ======== STa ========= ")
    print(STa["correct"]/(STa["correct"]+STa["ambiguous"]+STa["error"]))

    print(" ======== STs ========= ")
    print(STs["correct"]/(STs["correct"]+STs["ambiguous"]+STs["error"]))

    print(" ======== OOa ========= ")
    print(OOa["correct"]/(OOa["correct"]+OOa["ambiguous"]+OOa["error"]))

    print(" ======== OOs ========= ")
    print(OOs["correct"]/(OOs["correct"]+OOs["ambiguous"]+OOs["error"]))

    print(" ======== OR ========= ")
    print(OR["correct"]/(OR["correct"]+OR["ambiguous"]+OR["error"]))

    print(" ======== Advanced ========= ")
    print(Advanced["correct"]/(Advanced["correct"]+Advanced["ambiguous"]+Advanced["error"]))

    print(" ======== overall ========= ")
    print(overall["correct"]/(overall["correct"]+overall["ambiguous"]+overall["error"]))

if __name__ == "__main__":
    path_2_eval = "gpt_ll3da_zero_shot"
    main(path_2_eval)