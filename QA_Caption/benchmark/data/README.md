# Prepare MMScan Data

Based on the EmbodiedScan's data structure, please follow the directory structure shown below to organize the MMScan's annotations.

```
data
├── mmscan_anno
│   ├── object_caption.json -> path_of_object_caption_json 
│   ├── QAs -> path_of_QA_json_dir
│   ├── region_caption.json -> path_of_region_caption_json
│   ├── test_scene_ids.txt  # list of scene ids for testing
│   ├── train_scene_ids.txt # list of scene ids for training
│   └── val_scene_ids.txt  # list of scene ids for validation
├── mmscan_info
│   ├── 3rscan_mapping.json -> path_of_the_3rscan_id_mapping
│   ├── embodiedscan_infos_train_full.pkl -> path_of_ embodiedscan_infos_train_full
│   ├── embodiedscan_infos_val_full.pkl -> path_of_ embodiedscan_infos_val_full
│   └── mp3d_mapping.json -> path_of_the_mp3d_id_mapping
└── mmscan_scenes -> path_of_global_aligned_point_clouds

```