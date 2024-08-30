<br />
<p align="center">
  <h1 align="center">3D-LLM: Injecting the 3D World into Large Language Models (NeurIPS 2023 Spotlight) </h1>
  <p align="center">
    <a href="https://evelinehong.github.io">Yining Hong</a>,
    <a href="https://haoyuzhen.com">Haoyu Zhen</a>,
    <a href="https://peihaochen.github.io">Peihao Chen</a>,
    <a href="https://zsh2000.github.io">Shuhong Zheng</a>,
    <a href="https://yilundu.github.io">Yilun Du</a>,
    <a href="https://zfchenunique.github.io">Zhenfang Chen</a>,
    <a href="https://people.csail.mit.edu/ganchuang">Chuang Gan</a>
  </p>
  <p align="center">
    <a href='https://arxiv.org/abs/2307.12981'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://vis-www.cs.umass.edu/3dllm/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
  <p align="center">
    <img src="figs/pipeline.png" alt="Logo" width="80%">
  </p>
</p>

3D-LLM is the first Large Language Model that could take 3D representations as inputs. It is able to handle both object (e.g., objaverse) and scene data (e.g., scannet & hm3d). 

## Installation
Install [salesforce-lavis](https://github.com/salesforce/LAVIS)

```shell
$ conda create -n lavis python=3.8
$ conda activate lavis

$ git clone https://github.com/salesforce/LAVIS.git SalesForce-LAVIS
$ cd SalesForce-LAVIS
$ pip install -e .

$ pip install positional_encodings
```

## Checkpoints
### Pretraining Checkpoints
[Pretrained checkpoints](https://drive.google.com/file/d/1tiis8mOdZGBzmR7vgZtRE4Ni_2FE4nTr/view?usp=drive_link) are released (Please use v2!)

### Finetuning Checkpoints
[Finetuning checkpoints](https://drive.google.com/drive/folders/1RKP1cz6R6H8YziEc4f3MHW9dCCXQChbA?usp=drive_link) for [ScanQA](https://drive.google.com/file/d/1sPynAO8pI_RPR4pwWTrx8weDTdMPsqtW/view?usp=drive_link), [SQA3d](https://drive.google.com/file/d/1Ka9TWv6cs6h-pPaaQG1auIiQma2xbNFk/view?usp=drive_link), and [3DMV_VQA](https://drive.google.com/file/d/1_h2wPPGO64HY5LUcA1bD8DlZx3WCsY8b/view?usp=drive_link) are released.
The results are better than preprint-version paper. We will update the camera-ready paper to the arxiv soon.

## Quick Start: Inference
Download the objaverse subset features [here](https://drive.google.com/file/d/1mJZONfWREfIUAPYXP65D65uS2EoplAfR/view?usp=drive_link). Download the [pretrained checkpoints](https://drive.google.com/drive/folders/1urI2I3S8SgLD8L9brl4ae1Mul_yhCxJe?usp=drive_link). For more details, please refer to `3DLLM_BLIP2-base/DEMO.md`.
```
$ cd 3DLLM_BLIP2-base
$ conda activate lavis

python inference.py # for objects
python inference.py --mode room # for scenes
```
TODO: huggingface auto load checkpoint.

## Finetuning
Finetuning config yaml files that need to be changed are in [this directory](https://github.com/UMass-Foundation-Model/3D-LLM/tree/main/3DLLM_BLIP2-base/lavis/projects/blip2/train)
1. Download the [pretrained checkpoints](https://drive.google.com/file/d/1tiis8mOdZGBzmR7vgZtRE4Ni_2FE4nTr/view?usp=drive_link). Modify the "resume_checkpoint_path" path in the yaml files
2. Download the [questions](https://drive.google.com/drive/folders/14MDiDl6Cch_B27Q0aZgdElhAEOBBpn2o?usp=drive_link), modify the "annotations" path in the yaml files
3. Download the [scannet features](https://drive.google.com/drive/folders/1H1SKZsK_XpMkXntwhY4BqSxsGCRP8cgp?usp=drive_link)  or [3dmv-vqa features](https://drive.google.com/drive/folders/1NdFKKn_IZxGezi6fXA60rF1uxTOmhOet?usp=drive_link). Modify the path (both train and val) in lavis/datasets/datasets/threedvqa_datasets.py
4.
```
$ cd 3DLLM_BLIP2-base

$ conda activate lavis

python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/<finetune_yaml_file>
```
You can also load the finetuning checkpoints [here](https://drive.google.com/drive/folders/1RKP1cz6R6H8YziEc4f3MHW9dCCXQChbA?usp=drive_link).

5.**Calculating scores**
```
cd calculate_scores
python calculate_score_<task>.py --folder <your result dir> --epoch <your epoch>
```
please also modify the feature and question path in the scripts

TODO: huggingface auto load checkpoint.


## Data
All data will be gradually released in [Google Drive](https://drive.google.com/drive/folders/188Yd7tmiUfyct-dVMpkQ8q_tnqkb-4bo?usp=sharing) and [Huggingface](https://huggingface.co/datasets/ShuhongZheng/3D-LLM) (All files are released in Google Drive first and then Huggingface. Please refer to the Google Drive for file structure)

### Pretraining Data
We are still cleaning the grounding & navigation part. All other pre-training data are released.
#### Object Data
  &emsp;Language annotations of object data released [here](https://drive.google.com/file/d/17K1ZGb1HpvPj36a84GefGQK7DOkRp_32/view?usp=sharing).

  &emsp;For downloading Objaverse data, please refer to [Objaverse website](https://objaverse.allenai.org/).
  
  &emsp;To get 3D features and point clouds of the Objaverse data, please refer to [Step1](https://github.com/UMass-Foundation-Model/3D-LLM/blob/main/README.md#step1-render-images-from-different-views-of-a-scene) and [Step3](https://github.com/UMass-Foundation-Model/3D-LLM/blob/main/README.md#step3-3d-feature-construction-from-rendered-images) of 3DLanguage Data generation - ChatCaptioner based

  &emsp;A small set of objaverse features is released [here](https://drive.google.com/file/d/1mJZONfWREfIUAPYXP65D65uS2EoplAfR/view?usp=drive_link).
  
  &emsp;TODO: We will probably release the whole set of Objaverse 3D features 
#### Scene Data
 &emsp; [Language data released here](https://drive.google.com/file/d/18AVyAWwMwqP8nyZQfTpEMw1Wfhghjw0b/view?usp=drive_link). 
 
 &emsp; 3D features and point clouds (~250G) are released [here](https://drive.google.com/drive/folders/1bJpK00UFRZGH7tdx3PaqDrrTfHNP6B2K?usp=drive_link). However, if you want to explore generating the features yourself, please refer to the Three-step 3D Feature Extraction part [here](https://github.com/UMass-Foundation-Model/3D-LLM#three-step-3d-feature-extraction). Please use v2 to be consistent with the checkpoints (and also result in better performances).

 &emsp;chat: 73103. task: 84531

### Finetuning Data
3D features and point clouds of Scannet (used for finetuning ScanQA and SQA3D) are released in [here](https://drive.google.com/drive/folders/1CsEt48jj5uCyelGcXXJBkGH86QYeCE8D?usp=drive_link). 3D features and point clouds of 3DMV-VQA are released [here](https://drive.google.com/drive/folders/1NdFKKn_IZxGezi6fXA60rF1uxTOmhOet?usp=drive_link) (3DMV-VQA data will be further updated for a clearer structure).

All questions can be found [here](https://drive.google.com/drive/folders/14MDiDl6Cch_B27Q0aZgdElhAEOBBpn2o?usp=drive_link).



## 3DLanguage Data Generation

### ChatCaptioner based / Three-step 3D Feature Extraction (Objaverse)
#### Step1: render images from different views of a scene
Follow the instruction in ``3DLanguage_data/ChatCaptioner_based/objaverse_render/README.md`` for installation. 

The following code will render images of a objaverse scene (e.g. f6e9ec5953854dff94176c36b877c519). The rendered images will be saved at ``3DLanguage_data/ChatCaptioner_based/objaverse_render/output``.
(Please refer to ``3DLanguage_data/ChatCaptioner_based/objaverse_render/README.md`` for more details about the command)

```shell
$ cd ./3DLanguage_data/ChatCaptioner_based/objaverse_render

$ {path/to/blender} -b -P render.py -noaudio --disable-crash-handler -- --uid f6e9ec5953854dff94176c36b877c519

```

#### Step2: generate caption for this objaverse scene
Installation: 

Please follow [ChatCaptioner](https://github.com/Vision-CAIR/ChatCaptioner) to install the environment/


The following code will read the rended images of an objaverse scene (e.g., f6e9ec5953854dff94176c36b877c519) and generate scene caption at ``3DLanguage_data/ChatCaptioner_based/output``

```shell
$ cd ./3DLanguage_data/ChatCaptioner_based

$ python chatcaption.py --specific_scene f6e9ec5953854dff94176c36b877c519
```

#### Step3: 3D feature construction from rendered images

Follow the instruction in ``3DLanguage_data/ChatCaptioner_based/gen_features/README.md`` for extracting 3D features from rendered images.
```shell
$ cd ./3DLanguage_data/ChatCaptioner_based/gen_features
```


### Box-Demonstration-Instruction based
TODO

### Revision based
TODO

## Three-step 3D Feature Extraction (Scene)
This section is for constructing 3D features for scene data. If you already downloaded our released scene data, please skip this section.
### First step
Installation: 

Please follow [Mask2Former](https://github.com/facebookresearch/Mask2Former) to install the environment and download the [pretrained weight](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl) to the current directory
if extracting the masks with [Mask2Former](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf).

Please follow [Segment Anything](https://github.com/facebookresearch/segment-anything) to install the environment and download the [pretrained weight](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) to the current directory if extracting the masks with [SAM](https://arxiv.org/abs/2304.02643).

Extract masks with Mask2Former:

```shell
$ cd ./three_steps_3d_feature/first_step

$ python maskformer_mask.py --scene_dir_path DATA_DIR_WITH_RGB_IMAGES --save_dir_path DIR_YOU_WANT_TO_SAVE_THE_MASKS
```

Extract masks with Segment Anything:

```shell
$ cd ./three_steps_3d_feature/first_step

$ python sam_mask.py --scene_dir_path DATA_DIR_WITH_RGB_IMAGES --save_dir_path DIR_YOU_WANT_TO_SAVE_THE_MASKS
```

After the first step, we are expected to obtain a directory of masks (specified by ``--save_dir_path``) that contains extracted masks for
multi-view images of the scenes.

### Second step
Note: BLIP features are for LAVIS(BLIP2), CLIP features are for open-flamingo.

Installation: The same as the following ``3D-LLM_BLIP2-based`` section to install [salesforce-lavis](https://github.com/salesforce/LAVIS).

There are four options: (1) Extract CLIP feature with Mask2Former masks; (2) Extract CLIP feature with SAM masks;
(3) Extract BLIP feature with Mask2Former masks; (4) Extract BLIP feature with SAM masks.

Extract 2D CLIP features with Mask2Former masks:
```shell
$ cd ./three_steps_3d_feature/second_step/

$ python clip_maskformer.py --scene_dir_path DATA_DIR_WITH_RGB_IMAGES --mask_dir_path MASK_DIR_FROM_1ST_STEP --save_dir_path DIR_YOU_WANT_TO_SAVE_THE_FEAT
```

For the other options, the scripts are in similar format.

After the second step, we are expected to obtain a directory of features (specified by ``--save_dir_path``) that contains 2D features for
multi-view images of the scenes.

### Third step
#### Direct Reconstruction
Installation:

Please install the [Habitat environment](https://github.com/facebookresearch/habitat-lab/tree/challenge-2022).

Reconstruct 3D feature from multi-view 2D features:

```shell
$ cd ./three_steps_3d_feature/third_step/

$ python sam_mask.py --data_dir_path DATA_DIR_WITH_RGB_IMAGES --depth_dir_path DATA_DIR_WITH_DEPTH_IMAGES --feat_dir_path FEATURE_DIR_FROM_2ND_STEP
```

After the third step, we are expected to obtain two files (``pcd_pos.pt`` and ``pcd_feat.pt``) for each room inside the corresponding RGB directory.
``pcd_pos.pt`` contains the point positions of the 3D point cloud (shape: ``N * 3``). ``pcd_feat.pt`` contains the point features of the 3D point cloud (shape: ``N * n_dim``).
``N`` is the number of sampled points in the point cloud (default: 300000) and ``n_dim`` is the feature dimension (1024 for CLIP feature, 1408 for BLIP feature).

#### GradSLAM (Feature Fusion)
Refer to [Concept Fusion](https://github.com/concept-fusion/concept-fusion).

We will also release our reproduced version of Concept Fusion for our feature generation (we reproduced the paper before their official release).

#### Neural Field
Please refer to [3D-CLR](https://github.com/evelinehong/3D-CLR-Official) repository.

## Pre-training

```shell
$ cd 3DLLM_BLIP2-base

$ conda activate lavis
# use facebook/opt-2.7b:
$ TODO
# use flant5
$ python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/pretrain.yaml
```
## 3D-LLM_flamingo-based
TODO.

## Citation

If you find our work useful, please consider citing:

```
@article{3dllm,
 author = {Hong, Yining and Zhen, Haoyu and Chen, Peihao and Zheng, Shuhong and Du, Yilun and Chen, Zhenfang and Gan, Chuang},
 title = {3D-LLM: Injecting the 3D World into Large Language Models},
 journal = {NeurIPS},
 year = {2023},
} 
```

### Acknowledgements

https://github.com/salesforce/LAVIS

https://github.com/facebookresearch/Mask2Former

https://github.com/facebookresearch/segment-anything

https://github.com/mlfoundations/open_flamingo

https://github.com/concept-fusion/concept-fusion

https://github.com/evelinehong/3D-CLR-Official
