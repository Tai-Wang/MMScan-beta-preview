import os
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
import json
import argparse
from glob import glob
from lavis.common.registry import registry
from tqdm import tqdm

# ======== Step 0: Configurations >>>>>>>>
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="object", type=str, choices=["object", "room"])
parser.add_argument("-v", "--visualize", action="store_true")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = "weights/pretrain_blip2_sam_flant5xl_v2.pth"
assert os.path.exists(ckpt_path), "Please specify the checkpoint path."

anno_files = glob("../../../data/QAs/*.json")
with open(os.path.join(f'../../../data/mmscan_anno/test_scene_ids.txt' ), 'r') as f:
    scan_names = f.read().splitlines()
to_skip = []
for l in anno_files:
    extrac = l.split("/")[-1].split(".")[0].split("_")
    extrac = "_".join(extrac[:-1])
    if extrac not in scan_names:
        to_skip.append(l)
for l in to_skip:
    anno_files.remove(l)

mp3d_mapping = json.load(open("../../../data/mp3d_mapping.json"))
trscan_mapping = json.load(open("../../../data/3rscan_mapping.json"))

finished_scenes = os.listdir("outputs")
finished_scenes_name = []
for i in finished_scenes:
    if "skip" not in i:
        finished_scenes_name.append(i.replace(".json",""))
print(finished_scenes_name)
anno_files = anno_files[len(anno_files)//2:]


# ======== Step 1: Load model from checkpoint >>>>>>>>
print("Loading model from checkpoint...")
model_cfg = {
    "arch": "blip2_t5",
    "model_type": "pretrain_flant5xl",
    "use_grad_checkpoint": False,
}
model_cfg = OmegaConf.create(model_cfg)
model = registry.get_model_class(model_cfg.arch).from_pretrained(model_type=model_cfg.model_type)
checkpoint = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(checkpoint["model"], strict=False)
model.eval()
model.to(DEVICE)

processor_cfg = {"name": "blip_question", "prompt": ""}
processor_cfg = OmegaConf.create(processor_cfg)
text_processor = registry.get_processor_class(processor_cfg.name).from_config(processor_cfg)

# ======== Step 3: Inference >>>>>>>>


for scene_path in tqdm(anno_files):

    out_json = []
    embodied_scan_anno = json.load(open(scene_path))
    scene_name = scene_path.split("/")[-1].replace("_QA.json","")
    print(scene_name)
    if scene_name in finished_scenes_name:
        continue
    if "3rscan" in scene_name:
        house_name = scene_name
        for k in trscan_mapping:
            if trscan_mapping[k] == house_name:
                mapped_house = trscan_mapping[k]
                break
        feat_root = f"3dLLM/3rscan_step3_feat/{scene_name}"
        if os.path.exists(os.path.join(feat_root,"pcd_feat.pt")):
            pc_feature =  torch.tensor(torch.load(os.path.join(feat_root,"pcd_feat.pt")))
            pc_points = torch.tensor(torch.load(os.path.join(feat_root,"pcd_pos.pt")))
        else:
            print(f"skip {mapped_house}")
            continue
        
    elif "mp3d" in scene_name:
        house_name = "_".join(scene_name.split("_")[:2])
        for k in mp3d_mapping:
            if mp3d_mapping[k] == house_name:
                mapped_house = k
                break
        feat_root = f"3dLLM/mp3d_step3_feat/{mapped_house}"
        if os.path.exists(os.path.join(feat_root,"pcd_feat.pt")):
            pc_feature =  torch.tensor(torch.load(os.path.join(feat_root,"pcd_feat.pt")))
            pc_points = torch.tensor(torch.load(os.path.join(feat_root,"pcd_pos.pt")))
        else:
            print(f"skip {mapped_house}")
            continue
    else:
        try:
            pc_feature =  torch.load(os.path.join("3dLLM/scannnet/voxelized_features_sam_nonzero_preprocess",f"{scene_name}.pt"))
            pc_points = torch.from_numpy(np.load(open(os.path.join(f"3dLLM/scannnet/voxelized_voxels_sam_nonzero_preprocess",f"{scene_name}.npy"),'rb')))
        except:
            json.dump(out_json,open(f"outputs/{scene_name}_skiped.json",'w'))
            continue

    pc_feature = pc_feature.to(DEVICE).unsqueeze(0)  # (1, N, 1408)
    pc_points = pc_points.long().to(DEVICE).unsqueeze(0)  # (1, N, 3)

    model_inputs = {"text_input": "", "pc_feat": pc_feature, "pc": pc_points}

    for query in embodied_scan_anno:
        model_inputs = {"text_input": str(query["question"]), "pc_feat": pc_feature, "pc": pc_points}
        model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items() if isinstance(v, torch.Tensor)}
        model_inputs["text_input"]=str(query["question"])
        model_outputs = model.predict_answers(
            samples=model_inputs,
            max_len=50,
            length_penalty=1.2,
            repetition_penalty=1.5,
        )
        model_outputs = model_outputs[0]
        print(query["answers"],model_outputs)
        out_json.append({
            "sub_class":query["sub_class"],
            'ID':query['ID'],
            "answer":model_outputs,
            "gt_answer":query["answers"]
        })
    json.dump(out_json,open(f"outputs/{scene_name}.json",'w'))